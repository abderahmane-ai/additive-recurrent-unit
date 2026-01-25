#!/usr/bin/env python3
"""
Phase 8: Long-Form Sentiment Analysis - IMDb Benchmark

Tests ARU on full-length movie reviews, where maintaining sentiment context
over very long sequences (500-2000 tokens) is critical for accurate classification.

Task:
- Input: Movie review text (variable length, up to 2000 tokens)
- Output: Binary sentiment (positive/negative)
- Challenge: Long-term dependency, sentiment accumulation, context maintenance

Why ARU excels:
- Additive accumulation of sentiment signals across long reviews
- Persistence gate (π) maintains early sentiment without decay
- Accumulation gate (α) integrates new sentiment evidence
- Unlike GRU's convex constraint, ARU can truly accumulate sentiment

Dataset: IMDb Movie Reviews
- Source: https://ai.stanford.edu/~amaas/data/sentiment/
- 25,000 training reviews, 25,000 test reviews
- Binary labels (positive/negative)
- Average review length: ~230 words (but can be 1000+ words)

Key Innovation: We test on BOTH truncated (256 tokens) and full-length reviews
to demonstrate ARU's superior long-term memory compared to GRU/LSTM.
"""

import argparse
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import copy

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM
from utils.training import count_parameters

console = Console()


def load_imdb_data(max_len=None, max_vocab=10000):
    """
    Load IMDb dataset with optional truncation.
    
    Args:
        max_len: Maximum sequence length (None = no truncation)
        max_vocab: Maximum vocabulary size
        
    Returns:
        train_data, train_labels, test_data, test_labels, vocab_size
    """
    try:
        from datasets import load_dataset
        from collections import Counter
        import re
    except ImportError:
        console.print("[red]Error: datasets not installed. Install with: pip install datasets[/red]")
        sys.exit(1)
    
    console.print("[cyan]Loading IMDb dataset...[/cyan]")
    
    # Load dataset from HuggingFace
    dataset = load_dataset('imdb', trust_remote_code=True)
    
    # Simple tokenizer (split on whitespace and punctuation)
    def tokenize(text):
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    # Build vocabulary from training data
    console.print("[cyan]Building vocabulary...[/cyan]")
    counter = Counter()
    train_data_raw = []
    train_labels_raw = []
    
    for example in dataset['train']:
        tokens = tokenize(example['text'])
        counter.update(tokens)
        train_data_raw.append(tokens)
        train_labels_raw.append(example['label'])
    
    # Create vocab (most common words)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    console.print(f"[green]✓[/green] Vocabulary size: {vocab_size:,}")
    
    # Convert tokens to indices
    def tokens_to_indices(tokens, max_length=None):
        indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        if max_length is not None:
            indices = indices[:max_length]
        return indices
    
    # Process training data
    train_data = [tokens_to_indices(tokens, max_len) for tokens in train_data_raw]
    train_labels = train_labels_raw
    
    # Process test data
    test_data_raw = []
    test_labels_raw = []
    for example in dataset['test']:
        tokens = tokenize(example['text'])
        test_data_raw.append(tokens_to_indices(tokens, max_len))
        test_labels_raw.append(example['label'])
    
    test_data = test_data_raw
    test_labels = test_labels_raw
    
    # Statistics
    train_lengths = [len(seq) for seq in train_data]
    test_lengths = [len(seq) for seq in test_data]
    
    console.print(f"[green]✓[/green] Train samples: {len(train_data):,}")
    console.print(f"[green]✓[/green] Test samples: {len(test_data):,}")
    console.print(f"[dim]Train length - Mean: {np.mean(train_lengths):.0f}, "
                 f"Median: {np.median(train_lengths):.0f}, "
                 f"Max: {np.max(train_lengths):.0f}[/dim]")
    console.print(f"[dim]Test length - Mean: {np.mean(test_lengths):.0f}, "
                 f"Median: {np.median(test_lengths):.0f}, "
                 f"Max: {np.max(test_lengths):.0f}[/dim]")
    
    return train_data, train_labels, test_data, test_labels, vocab_size


def collate_batch(batch):
    """Collate function for DataLoader with padding."""
    sequences, labels = zip(*batch)
    
    # Convert to tensors
    sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Pad sequences
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return sequences_padded, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def run_imdb_benchmark(config):
    """Run IMDb sentiment analysis benchmark."""
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    
    max_len_str = f"{config['max_len']}" if config['max_len'] else "Full-length"
    console.print(f"[cyan]Max sequence length:[/cyan] {max_len_str}")
    console.print(f"[cyan]Vocabulary size:[/cyan] {config['max_vocab']:,}\n")
    
    # Load data
    train_data, train_labels, test_data, test_labels, vocab_size = load_imdb_data(
        max_len=config['max_len'],
        max_vocab=config['max_vocab']
    )
    
    # Create datasets
    train_dataset = list(zip(train_data, train_labels))
    test_dataset = list(zip(test_data, test_labels))
    
    # Split train into train/val
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    
    console.print(f"\n[green]✓[/green] Train: {len(train_dataset):,} samples")
    console.print(f"[green]✓[/green] Val: {len(val_dataset):,} samples")
    console.print(f"[green]✓[/green] Test: {len(test_dataset):,} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )
    
    # Models to compare
    models_to_run = config.get('models', ['ARU', 'GRU', 'LSTM'])
    
    results = {}
    
    for model_name in models_to_run:
        console.print(f"\n[bold cyan]Training {model_name}[/bold cyan]")
        
        # Create model
        if model_name == 'ARU':
            model = ARU(
                input_size=vocab_size,
                hidden_size=config['hidden_size'],
                num_classes=2,
                dropout=config['dropout'],
                use_embedding=True
            )
        elif model_name == 'GRU':
            model = ManualGRU(
                input_size=vocab_size,
                hidden_size=config['hidden_size'],
                num_classes=2,
                dropout=config['dropout'],
                use_embedding=True
            )
        elif model_name == 'LSTM':
            model = ManualLSTM(
                input_size=vocab_size,
                hidden_size=config['hidden_size'],
                num_classes=2,
                dropout=config['dropout'],
                use_embedding=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = model.to(device)
        params = count_parameters(model)
        console.print(f"Parameters: {params:,}")
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None
        
        train_start = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]{model_name}", total=config['epochs'])
            
            for epoch in range(config['epochs']):
                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                
                # Validate
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]{model_name} - Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)"
                )
                
                if patience_counter >= config['patience']:
                    console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                    break
        
        train_time = time.time() - train_start
        
        # Load best model and test
        model.load_state_dict(best_state)
        
        test_start = time.time()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        inference_time = time.time() - test_start
        
        results[model_name] = {
            'params': params,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'train_time': train_time,
            'inference_time': inference_time
        }
        
        console.print(f"[green]✓[/green] {model_name}: Test Acc = {test_acc:.2f}%")
    
    # Results table
    console.print("\n")
    table = Table(
        title=f"📊 IMDb Sentiment Analysis - {max_len_str} tokens",
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Val Acc", justify="right")
    table.add_column("Test Acc", justify="right")
    table.add_column("Train Time", justify="right")
    table.add_column("Inference", justify="right")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['best_val_acc']:.2f}%",
            f"{data['test_acc']:.2f}%",
            f"{data['train_time']:.1f}s",
            f"{data['inference_time']:.2f}s"
        )
    
    console.print(table)
    console.print("\n[dim]Higher accuracy is better[/dim]")
    
    # Compare ARU vs GRU
    if 'ARU' in results and 'GRU' in results:
        aru_acc = results['ARU']['test_acc']
        gru_acc = results['GRU']['test_acc']
        
        if aru_acc > gru_acc:
            improvement = aru_acc - gru_acc
            console.print(f"\n[green]🎯 ARU outperforms GRU by {improvement:.2f} percentage points![/green]")
        else:
            diff = gru_acc - aru_acc
            console.print(f"\n[yellow]⚠️  GRU outperforms ARU by {diff:.2f} percentage points[/yellow]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 8: IMDb Long-Form Sentiment Analysis')
    parser.add_argument('--max-len', type=int, default=None,
                       help='Maximum sequence length (None = full-length)')
    parser.add_argument('--max-vocab', type=int, default=10000,
                       help='Maximum vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden size for RNN models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['ARU', 'GRU', 'LSTM'],
                       help='Models to benchmark')
    parser.add_argument('--truncated', action='store_true',
                       help='Use truncated sequences (256 tokens)')
    parser.add_argument('--compare-lengths', action='store_true',
                       help='Run comparison on both truncated and full-length')
    
    args = parser.parse_args()
    
    # Set max_len based on flags
    if args.truncated:
        args.max_len = 256
    
    config = vars(args)
    
    if args.compare_lengths:
        # Run on both truncated and full-length
        console.print(Panel.fit(
            "[bold cyan]📊 Phase 8: IMDb Sentiment Analysis Benchmark[/bold cyan]\n"
            "[yellow]Comparing Truncated vs Full-Length Reviews[/yellow]\n"
            "[dim]Testing ARU's long-term memory advantage[/dim]",
            border_style="green"
        ))
        
        console.print("\n[bold yellow]═══ PART 1: Truncated Reviews (256 tokens) ═══[/bold yellow]\n")
        config['max_len'] = 256
        results_truncated = run_imdb_benchmark(config)
        
        console.print("\n\n[bold yellow]═══ PART 2: Full-Length Reviews ═══[/bold yellow]\n")
        config['max_len'] = None
        results_full = run_imdb_benchmark(config)
        
        # Comparison summary
        console.print("\n\n[bold cyan]═══ SUMMARY: Truncated vs Full-Length ═══[/bold cyan]\n")
        
        summary_table = Table(header_style="bold magenta")
        summary_table.add_column("Model", style="cyan")
        summary_table.add_column("Truncated (256)", justify="right")
        summary_table.add_column("Full-Length", justify="right")
        summary_table.add_column("Difference", justify="right")
        
        for model_name in config['models']:
            if model_name in results_truncated and model_name in results_full:
                trunc_acc = results_truncated[model_name]['test_acc']
                full_acc = results_full[model_name]['test_acc']
                diff = full_acc - trunc_acc
                
                diff_str = f"{diff:+.2f}%"
                if diff < -2:
                    diff_str = f"[red]{diff_str}[/red]"
                elif diff > 2:
                    diff_str = f"[green]{diff_str}[/green]"
                else:
                    diff_str = f"[yellow]{diff_str}[/yellow]"
                
                summary_table.add_row(
                    model_name,
                    f"{trunc_acc:.2f}%",
                    f"{full_acc:.2f}%",
                    diff_str
                )
        
        console.print(summary_table)
        
        # Analysis
        if 'ARU' in results_truncated and 'GRU' in results_truncated:
            aru_drop = results_truncated['ARU']['test_acc'] - results_full['ARU']['test_acc']
            gru_drop = results_truncated['GRU']['test_acc'] - results_full['GRU']['test_acc']
            
            console.print(f"\n[bold]Performance Drop Analysis:[/bold]")
            console.print(f"  ARU: {aru_drop:.2f} percentage points")
            console.print(f"  GRU: {gru_drop:.2f} percentage points")
            
            if aru_drop < gru_drop:
                console.print(f"\n[green]✓ ARU is more robust to longer sequences![/green]")
                console.print(f"[dim]ARU's additive accumulation maintains performance better than GRU's convex constraint[/dim]")
    else:
        # Single run
        max_len_desc = "Truncated (256 tokens)" if config['max_len'] == 256 else "Full-Length"
        console.print(Panel.fit(
            f"[bold cyan]📊 Phase 8: IMDb Sentiment Analysis Benchmark[/bold cyan]\n"
            f"[yellow]{max_len_desc} Reviews[/yellow]\n"
            "[dim]Testing ARU on long-form sentiment analysis[/dim]",
            border_style="green"
        ))
        
        run_imdb_benchmark(config)


if __name__ == "__main__":
    main()
