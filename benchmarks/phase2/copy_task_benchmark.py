#!/usr/bin/env python3
"""
Phase 2: Copy Task Benchmark - Long-Term Memory Test

This benchmark validates ARU (Additive Recurrent Unit) on the classic Copy Task,
which requires models to memorize and reproduce sequences after a delay period.

Task Structure:
1. Input sequence of T symbols (e.g., [3, 7, 2, 5, 1])
2. Blank delay period of D timesteps (all zeros)
3. Copy marker signal (special symbol 9)
4. Model must output the original sequence

Key ARU Advantage: The three-gate architecture with high persistence (π ≈ 1) 
enables perfect accumulation and retention of information across the delay period.
This is a pure test of long-term memory without confounding factors like local
patterns or linguistic structure.

Example (T=5, D=10):
Input:  [3, 7, 2, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0]
Target: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 2, 5, 1]
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import copy
import matplotlib.pyplot as plt
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM, ManualRNN
from utils.training import count_parameters

console = Console()

COPY_TASK_CONFIG = {
    'seq_length': 10,  # Length of sequence to copy
    'delay_length': 50,  # Delay period (blank timesteps)
    'num_symbols': 8,  # Vocabulary size (0=blank, 1-8=symbols, 9=copy marker)
    'hidden_size': 128,
    'batch_size': 128,
    'epochs': 50,
    'lr': 0.001,
    'dropout': 0.0,  # No dropout for this task
    'patience': 10,
    'num_sequences': 20000,  # Training sequences
}

def generate_copy_task_data(num_sequences, seq_length, delay_length, num_symbols, seed=42):
    """
    Generate copy task dataset.
    
    Args:
        num_sequences: Number of sequences to generate
        seq_length: Length of sequence to memorize
        delay_length: Number of blank timesteps before copy
        num_symbols: Vocabulary size (1 to num_symbols)
        
    Returns:
        inputs: (num_sequences, total_length) - input sequences
        targets: (num_sequences, total_length) - target sequences
    """
    np.random.seed(seed)
    
    # Total length: sequence + delay + copy_marker + sequence
    total_length = seq_length + delay_length + 1 + seq_length
    
    inputs = np.zeros((num_sequences, total_length), dtype=np.int64)
    targets = np.zeros((num_sequences, total_length), dtype=np.int64)
    
    for i in range(num_sequences):
        # Generate random sequence (symbols 1 to num_symbols)
        sequence = np.random.randint(1, num_symbols + 1, size=seq_length)
        
        # Input: [sequence, zeros (delay), copy_marker, zeros]
        inputs[i, :seq_length] = sequence
        inputs[i, seq_length + delay_length] = num_symbols + 1  # Copy marker (9 if num_symbols=8)
        
        # Target: [zeros, zeros (delay), zeros, sequence]
        targets[i, seq_length + delay_length + 1:] = sequence
    
    return torch.from_numpy(inputs), torch.from_numpy(targets)

def compute_copy_accuracy(predictions, targets, seq_length, delay_length):
    """
    Compute accuracy on the copied sequence only (ignore other timesteps).
    
    Args:
        predictions: (batch, total_length) - predicted symbols
        targets: (batch, total_length) - target symbols
        seq_length: Length of sequence to copy
        delay_length: Delay period length
        
    Returns:
        accuracy: Percentage of correctly copied sequences
        per_symbol_accuracy: Percentage of correctly copied symbols
    """
    # Extract only the output region (after copy marker)
    start_idx = seq_length + delay_length + 1
    end_idx = start_idx + seq_length
    
    pred_output = predictions[:, start_idx:end_idx]
    target_output = targets[:, start_idx:end_idx]
    
    # Sequence accuracy: all symbols correct
    correct_sequences = (pred_output == target_output).all(dim=1).sum().item()
    sequence_accuracy = 100.0 * correct_sequences / predictions.size(0)
    
    # Per-symbol accuracy
    correct_symbols = (pred_output == target_output).sum().item()
    total_symbols = pred_output.numel()
    symbol_accuracy = 100.0 * correct_symbols / total_symbols
    
    return sequence_accuracy, symbol_accuracy

def train_epoch_copy(model, inputs, targets, criterion, optimizer, device, batch_size, config):
    """Train one epoch on copy task."""
    model.train()
    
    num_sequences = inputs.size(0)
    indices = torch.randperm(num_sequences)
    
    total_loss = 0
    total_seq_acc = 0
    total_sym_acc = 0
    num_batches = 0
    
    for i in range(0, num_sequences, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_inputs = inputs[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - return all states for sequence-to-sequence
        outputs = model(batch_inputs, return_all_states=True)
        
        # Reshape for loss computation
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = batch_targets.reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute accuracy
        _, predicted = outputs.max(dim=-1)
        seq_acc, sym_acc = compute_copy_accuracy(
            predicted, batch_targets, 
            config['seq_length'], config['delay_length']
        )
        
        total_seq_acc += seq_acc
        total_sym_acc += sym_acc
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_seq_acc = total_seq_acc / num_batches
    avg_sym_acc = total_sym_acc / num_batches
    
    return avg_loss, avg_seq_acc, avg_sym_acc

def evaluate_copy(model, inputs, targets, criterion, device, batch_size, config):
    """Evaluate on copy task."""
    model.eval()
    
    num_sequences = inputs.size(0)
    
    total_loss = 0
    total_seq_acc = 0
    total_sym_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch_inputs = inputs[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            outputs = model(batch_inputs, return_all_states=True)
            
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = batch_targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            total_loss += loss.item()
            
            _, predicted = outputs.max(dim=-1)
            seq_acc, sym_acc = compute_copy_accuracy(
                predicted, batch_targets,
                config['seq_length'], config['delay_length']
            )
            
            total_seq_acc += seq_acc
            total_sym_acc += sym_acc
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_seq_acc = total_seq_acc / num_batches
    avg_sym_acc = total_sym_acc / num_batches
    
    return avg_loss, avg_seq_acc, avg_sym_acc

def run_copy_task_benchmark(config: dict, seed: int = 42):
    """Run Copy Task benchmark."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}\n")
    
    # Generate data
    console.print("[cyan]Generating copy task dataset...[/cyan]")
    inputs, targets = generate_copy_task_data(
        config['num_sequences'],
        config['seq_length'],
        config['delay_length'],
        config['num_symbols'],
        seed=seed
    )
    
    # Split into train/val/test
    train_size = int(0.7 * config['num_sequences'])
    val_size = int(0.15 * config['num_sequences'])
    
    train_inputs = inputs[:train_size]
    train_targets = targets[:train_size]
    
    val_inputs = inputs[train_size:train_size + val_size]
    val_targets = targets[train_size:train_size + val_size]
    
    test_inputs = inputs[train_size + val_size:]
    test_targets = targets[train_size + val_size:]
    
    total_length = config['seq_length'] + config['delay_length'] + config['seq_length']
    
    console.print(f"[green]✓[/green] Generated copy task dataset")
    console.print(f"  Sequence length: {config['seq_length']}")
    console.print(f"  Delay length: {config['delay_length']}")
    console.print(f"  Total length: {total_length} timesteps")
    console.print(f"  Vocabulary: {config['num_symbols']} symbols + blank + marker")
    console.print(f"  Train: {len(train_inputs):,} | Val: {len(val_inputs):,} | Test: {len(test_inputs):,}")
    
    # Input size = num_symbols + 2 (blank=0, symbols=1-8, marker=9)
    input_size = config['num_symbols'] + 2
    output_size = config['num_symbols'] + 2
    
    models = {
        'ARU': ARU(
            input_size,
            config['hidden_size'],
            num_classes=output_size,
            dropout=config['dropout'],
            use_embedding=True
        ),
        'GRU': ManualGRU(
            input_size,
            config['hidden_size'],
            num_classes=output_size,
            dropout=config['dropout'],
            use_embedding=True
        ),
        'LSTM': ManualLSTM(
            input_size,
            config['hidden_size'],
            num_classes=output_size,
            dropout=config['dropout'],
            use_embedding=True
        ),
        'RNN': ManualRNN(
            input_size,
            config['hidden_size'],
            num_classes=output_size,
            dropout=config['dropout'],
            use_embedding=True
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            
            params = count_parameters(model)
            console.print(f"\n[bold cyan]Training {name}[/bold cyan]")
            console.print(f"[yellow]Parameters:[/yellow] {params:,}")
            
            # Format model specs
            model_str = str(model)
            lines = model_str.split('\n')
            console.print(f"[green]Specs:[/green] {lines[0]}")
            for line in lines[1:]:
                console.print(f"  {line}")
            
            best_val_seq_acc = 0.0
            best_model_state = None
            patience_counter = 0
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]{name}", total=config['epochs'])
                
                for epoch in range(1, config['epochs'] + 1):
                    train_loss, train_seq_acc, train_sym_acc = train_epoch_copy(
                        model, train_inputs, train_targets, criterion, optimizer, device,
                        config['batch_size'], config
                    )
                    val_loss, val_seq_acc, val_sym_acc = evaluate_copy(
                        model, val_inputs, val_targets, criterion, device,
                        config['batch_size'], config
                    )
                    
                    if val_seq_acc > best_val_seq_acc:
                        best_val_seq_acc = val_seq_acc
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    progress.update(
                        task, advance=1,
                        description=f"[cyan]{name} - Seq: {val_seq_acc:.1f}% Sym: {val_sym_acc:.1f}% (Best: {best_val_seq_acc:.1f}%)"
                    )
                    
                    if patience_counter >= config['patience']:
                        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                        break
            
            train_time = time.time() - start_time
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            test_loss, test_seq_acc, test_sym_acc = evaluate_copy(
                model, test_inputs, test_targets, criterion, device,
                config['batch_size'], config
            )
            
            # Show sample predictions to demonstrate what the model learned
            console.print(f"\n[bold yellow]Testing {name} on new data - Sample Predictions:[/bold yellow]")
            with torch.no_grad():
                sample_inputs = test_inputs[:3].to(device)
                sample_targets = test_targets[:3].to(device)
                sample_outputs = model(sample_inputs, return_all_states=True)
                _, sample_preds = sample_outputs.max(dim=-1)
                
                start_idx = config['seq_length'] + config['delay_length'] + 1
                end_idx = start_idx + config['seq_length']
                
                for i in range(3):
                    input_seq = sample_inputs[i, :config['seq_length']].cpu().numpy()
                    target_seq = sample_targets[i, start_idx:end_idx].cpu().numpy()
                    pred_seq = sample_preds[i, start_idx:end_idx].cpu().numpy()
                    
                    match = "✓" if np.array_equal(target_seq, pred_seq) else "✗"
                    console.print(f"  {match} Sample {i+1}:")
                    console.print(f"     Input:  {input_seq}")
                    console.print(f"     Target: {target_seq}")
                    console.print(f"     Pred:   {pred_seq}")
            
            model.train()
            
            results[name] = {
                'params': params,
                'best_val_seq_acc': best_val_seq_acc,
                'test_seq_acc': test_seq_acc,
                'test_sym_acc': test_sym_acc,
                'train_time': train_time
            }
            
            console.print(
                f"[green]✓[/green] {name} completed - "
                f"Seq Acc: {test_seq_acc:.1f}% | Sym Acc: {test_sym_acc:.1f}%"
            )
        
        except Exception as e:
            console.print(f"[bold red]Error training {name}:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        console.print("[bold red]No models completed training successfully[/bold red]")
        return
    
    # Results table
    console.print("\n")
    table = Table(
        title=f"📊 Copy Task Results (T={config['seq_length']}, D={config['delay_length']})",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Sequence Acc", justify="right", style="blue")
    table.add_column("Symbol Acc", justify="right", style="blue")
    table.add_column("Train Time", justify="right", style="yellow")
    
    for name, data in results.items():
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['test_seq_acc']:.1f}%",
            f"{data['test_sym_acc']:.1f}%",
            f"{data['train_time']:.0f}s"
        )
    
    console.print(table)
    
    winner = max(results.items(), key=lambda x: x[1]['test_seq_acc'])
    console.print(f"\n[bold green]🏆 Winner (Sequence Acc): {winner[0]} ({winner[1]['test_seq_acc']:.1f}%)[/bold green]")
    
    if 'ARU' in results and 'GRU' in results:
        aru_acc = results['ARU']['test_seq_acc']
        gru_acc = results['GRU']['test_seq_acc']
        diff = aru_acc - gru_acc
        console.print(f"\n[cyan]ARU vs GRU:[/cyan] {diff:+.1f}% difference")
        if diff > 5:
            console.print(
                "[green]ARU's three-gate architecture enables superior long-term memory![/green]"
            )
    
    console.print(
        f"\n[bold yellow]💡 Key Insight:[/bold yellow] The copy task requires perfect\n"
        f"   memorization across {config['delay_length']} blank timesteps. ARU's high persistence\n"
        f"   gate (π ≈ 1) enables lossless information retention, while GRU's\n"
        f"   convex combination constraint causes gradual information decay."
    )

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Copy Task Benchmark with ARU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seq-length', type=int, default=10, help='Sequence length to copy')
    parser.add_argument('--delay-length', type=int, default=50, help='Delay period length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    args = parser.parse_args()
    
    if args.seq_length:
        COPY_TASK_CONFIG['seq_length'] = args.seq_length
    if args.delay_length:
        COPY_TASK_CONFIG['delay_length'] = args.delay_length
    if args.epochs:
        COPY_TASK_CONFIG['epochs'] = args.epochs
    
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]ARU Phase 2: Copy Task Benchmark[/bold cyan]\n"
        "[yellow]Long-Term Memory Test - Memorize and Reproduce Sequences[/yellow]\n"
        f"[dim]Seq Length: {COPY_TASK_CONFIG['seq_length']} | "
        f"Delay: {COPY_TASK_CONFIG['delay_length']} | "
        f"Total: {COPY_TASK_CONFIG['seq_length'] * 2 + COPY_TASK_CONFIG['delay_length']} timesteps[/dim]",
        border_style="blue"
    ))
    
    run_copy_task_benchmark(COPY_TASK_CONFIG, seed=args.seed)

if __name__ == "__main__":
    main()
