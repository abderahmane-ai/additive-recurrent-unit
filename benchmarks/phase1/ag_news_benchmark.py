#!/usr/bin/env python3
"""
Phase 1: AG News Classification Benchmark with ARU

This benchmark validates ARU (Additive Recurrent Unit) against baselines on AG News
classification (4 classes). Features robust checkpointing and comprehensive evaluation.

ARU's additive capability allows it to simultaneously maintain category-specific
features while accumulating evidence from new tokens - impossible in GRU.
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
import seaborn as sns
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM, ManualRNN
from utils.data import load_ag_news_dataset
from utils.training import train_epoch, evaluate, count_parameters, test_and_print_samples

console = Console()

AG_NEWS_CONFIG = {
    'max_vocab_size': 5000,
    'max_len': 100,
    'hidden_size': 128,
    'batch_size': 256,
    'epochs': 8,
    'lr': 0.002,
    'dropout': 0.1,
    'train_samples': 60000,
    'test_samples': 7600,
    'patience': 3,
}

def run_ag_news_benchmark(config: dict, seed: int = 42):
    """Run AG News benchmark with Best Model Checkpointing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}\n")
    
    try:
        X_train, y_train, X_test, y_test, vocab = load_ag_news_dataset(
            max_vocab_size=config['max_vocab_size'],
            max_len=config['max_len'],
            train_samples=config['train_samples'],
            test_samples=config['test_samples']
        )
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        return

    vocab_size = len(vocab)
    num_classes = 4
    
    full_train_dataset = TensorDataset(X_train, y_train)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    console.print(f"Data split: Train: {train_size:,} | Val: {val_size:,} | Test: {len(X_test):,}")

    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == 'cuda' else {}

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'], shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'], **kwargs
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config['batch_size'], **kwargs
    )
    
    models = {
        'ARU': ARU(
            vocab_size, 
            config['hidden_size'], 
            num_classes=num_classes, 
            dropout=config['dropout']
        ),
        'GRU': ManualGRU(vocab_size, config['hidden_size'], num_classes=num_classes, dropout=config['dropout']),
        'LSTM': ManualLSTM(vocab_size, config['hidden_size'], num_classes=num_classes, dropout=config['dropout']),
        'RNN': ManualRNN(vocab_size, config['hidden_size'], num_classes=num_classes, dropout=config['dropout']),
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            # LSTM often needs lower learning rate
            lr = config['lr'] * 0.5 if name == 'LSTM' else config['lr']
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if name == 'LSTM':
                console.print(f"[dim]Using reduced LR for LSTM: {lr}[/dim]")
            
            params = count_parameters(model)
            console.print(f"\n[bold cyan]Training {name}[/bold cyan]")
            console.print(f"[yellow]Parameters:[/yellow] {params:,}")
            
            # Format model specs with proper indentation
            model_str = str(model)
            lines = model_str.split('\n')
            console.print(f"[green]Specs:[/green] {lines[0]}")
            for line in lines[1:]:
                console.print(f"  {line}")
            
            best_val_acc = 0.0
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
                    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=1.0)
                    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    progress.update(task, advance=1, description=f"[cyan]{name} - Epoch {epoch} - Train: {train_acc:.1f}% Val: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) Loss: {train_loss:.3f}")
                    
                    if patience_counter >= config['patience']:
                        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                        break
            
            train_time = time.time() - start_time
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            
            # Test on new data and show what the model learned
            console.print(f"\n[bold yellow]Testing {name} on new data...[/bold yellow]")
            class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
            test_and_print_samples(model, test_loader, device, num_samples=5, class_names=class_names)
            
            model.train()

            results[name] = {
                'params': params,
                'best_val_acc': best_val_acc,
                'test_acc': test_acc,
                'train_time': train_time
            }
            
            console.print(f"[green]✓[/green] {name} completed - Test Acc: {test_acc:.2f}% | Best Val Acc: {best_val_acc:.2f}%")
            
            if test_acc < 10.0:
                console.print(f"[yellow]⚠ Warning: {name} has suspiciously low accuracy ({test_acc:.2f}%). Check for bugs.[/yellow]")


        
        except Exception as e:
            console.print(f"[bold red]Error training {name}:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        console.print("[bold red]No models completed training successfully[/bold red]")
        return
    
    # Final Results Table
    console.print("\n")
    table = Table(title="📊 AG News Results (ARU)", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Best Val Acc", justify="right", style="blue")
    table.add_column("Test Acc", justify="right", style="blue")
    table.add_column("Train Time", justify="right", style="yellow")
    
    for name, data in results.items():
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['best_val_acc']:.2f}%",
            f"{data['test_acc']:.2f}%",
            f"{data['train_time']:.1f}s"
        )
    
    console.print(table)
    
    winner = max(results.items(), key=lambda x: x[1]['test_acc'])
    console.print(f"\n[bold green]🏆 Winner (Test Acc): {winner[0]} ({winner[1]['test_acc']:.2f}%)[/bold green]")
    
    # Performance comparison
    if 'ARU' in results and 'GRU' in results:
        aru_acc = results['ARU']['test_acc']
        gru_acc = results['GRU']['test_acc']
        diff = aru_acc - gru_acc
        console.print(f"\n[cyan]ARU vs GRU:[/cyan] {diff:+.2f}% difference")
        if diff > 0:
            console.print("[dim]ARU's additive accumulation enables better feature integration[/dim]")

def main():
    parser = argparse.ArgumentParser(description='Phase 1: AG News Benchmark with ARU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]ARU Phase 1: AG News Benchmark[/bold cyan]\n"
        "[yellow]Multi-class Classification with Additive Accumulation[/yellow]",
        border_style="blue"
    ))
    
    run_ag_news_benchmark(AG_NEWS_CONFIG, seed=args.seed)

if __name__ == "__main__":
    main()