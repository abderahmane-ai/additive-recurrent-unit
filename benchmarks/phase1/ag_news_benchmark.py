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
    """Run AG News benchmark with multiple seeds for statistical significance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}\n")
    
    num_runs = config.get('num_runs', 1)
    console.print(f"[cyan]Number of runs:[/cyan] {num_runs}\n")
    
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
    
    console.print(f"Vocab size: {vocab_size:,} | Classes: {num_classes}")
    
    # Store results across all runs
    all_results = {
        'ARU': {'test_acc': [], 'val_acc': [], 'train_time': []},
        'GRU': {'test_acc': [], 'val_acc': [], 'train_time': []},
        'LSTM': {'test_acc': [], 'val_acc': [], 'train_time': []},
        'RNN': {'test_acc': [], 'val_acc': [], 'train_time': []},
    }
    
    # Run each model multiple times
    for run_idx in range(num_runs):
        run_seed = seed + 1000 * (run_idx + 1)
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Run {run_idx + 1}/{num_runs} (seed={run_seed})[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
        
        # Set seed for this run
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)
        
        full_train_dataset = TensorDataset(X_train, y_train)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        generator = torch.Generator().manual_seed(run_seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )
        
        if run_idx == 0:
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
        
        for name, model in models.items():
            try:
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                # LSTM often needs lower learning rate
                lr = config['lr'] * 0.5 if name == 'LSTM' else config['lr']
                optimizer = optim.Adam(model.parameters(), lr=lr)
                
                params = count_parameters(model)
                
                if run_idx == 0:
                    console.print(f"\n[bold cyan]Training {name}[/bold cyan] ({params:,} params)")
                    if name == 'LSTM':
                        console.print(f"[dim]Using reduced LR for LSTM: {lr}[/dim]")
                else:
                    console.print(f"\n[cyan]Training {name}[/cyan]")
                
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
                        
                        progress.update(task, advance=1, description=f"[cyan]{name} - Val: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")
                        
                        if patience_counter >= config['patience']:
                            break
                
                train_time = time.time() - start_time
                
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                model.eval()
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                
                # Store results
                all_results[name]['test_acc'].append(test_acc)
                all_results[name]['val_acc'].append(best_val_acc)
                all_results[name]['train_time'].append(train_time)
                all_results[name]['params'] = params
                
                console.print(f"[green]OK[/green] {name}: Test={test_acc:.2f}%, Val={best_val_acc:.2f}%")
                
                if test_acc < 10.0:
                    console.print(f"[yellow]⚠ Warning: {name} has suspiciously low accuracy ({test_acc:.2f}%)[/yellow]")
            
            except Exception as e:
                console.print(f"[bold red]Error training {name}:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute statistics and display results
    console.print("\n" + "="*80)
    console.print("[bold magenta]STATISTICAL RESULTS[/bold magenta]")
    console.print("="*80 + "\n")
    
    # Results table with statistics
    table = Table(title=f"AG News Results (N={num_runs} runs)", header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test Acc", justify="right")
    table.add_column("Val Acc", justify="right")
    table.add_column("Time (s)", justify="right")
    
    for name in ['RNN', 'GRU', 'ARU', 'LSTM']:
        if not all_results[name]['test_acc']:
            continue
        
        data = all_results[name]
        test_mean = np.mean(data['test_acc'])
        test_std = np.std(data['test_acc'], ddof=1) if num_runs > 1 else 0.0
        val_mean = np.mean(data['val_acc'])
        val_std = np.std(data['val_acc'], ddof=1) if num_runs > 1 else 0.0
        time_mean = np.mean(data['train_time'])
        
        if num_runs > 1:
            test_str = f"{test_mean:.2f} ± {test_std:.2f}"
            val_str = f"{val_mean:.2f} ± {val_std:.2f}"
        else:
            test_str = f"{test_mean:.2f}"
            val_str = f"{val_mean:.2f}"
        
        table.add_row(
            name,
            f"{data['params']:,}",
            test_str,
            val_str,
            f"{time_mean:.1f}"
        )
    
    console.print(table)
    
    # Statistical significance tests
    if num_runs > 1:
        console.print("\n[bold cyan]Statistical Significance Tests (vs GRU baseline)[/bold cyan]\n")
        
        from utils.stats import format_comparison_table
        
        # Test accuracy comparison
        test_acc_dict = {name: data['test_acc'] for name, data in all_results.items() if data['test_acc']}
        console.print("[yellow]Test Accuracy Comparison:[/yellow]")
        console.print(format_comparison_table(test_acc_dict, "Test Acc", baseline='GRU', lower_is_better=False))
    
    # Performance comparison
    if all_results['ARU']['test_acc'] and all_results['GRU']['test_acc']:
        aru_mean = np.mean(all_results['ARU']['test_acc'])
        gru_mean = np.mean(all_results['GRU']['test_acc'])
        diff = aru_mean - gru_mean
        console.print(f"\n[cyan]ARU vs GRU:[/cyan] {diff:+.2f}% difference")
        if diff > 0:
            console.print("[dim]ARU's additive accumulation enables better feature integration[/dim]")
    
    console.print(f"\n[green]Benchmark complete![/green] Results ready for manual report update.")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Phase 1: AG News Benchmark with ARU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs with different seeds')
    args = parser.parse_args()
    
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]ARU Phase 1: AG News Benchmark[/bold cyan]\n"
        f"[yellow]Multi-class Classification | Runs: {args.num_runs}[/yellow]",
        border_style="blue"
    ))
    
    config = AG_NEWS_CONFIG.copy()
    config['num_runs'] = args.num_runs
    
    run_ag_news_benchmark(config, seed=args.seed)

if __name__ == "__main__":
    main()