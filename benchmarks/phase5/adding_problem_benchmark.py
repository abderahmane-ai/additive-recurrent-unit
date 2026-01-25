#!/usr/bin/env python3
"""
Phase 5: Adding Problem Benchmark

A classic synthetic benchmark for testing long-term dependency learning.
Fast to run, clearly demonstrates RNN capabilities.

Task:
- Input: Two sequences of length T
  1. Random numbers in [0, 1]
  2. Mask with exactly two 1s (rest 0s), marking which numbers to add
- Output: Sum of the two marked numbers

Example (T=10):
  Numbers: [0.3, 0.7, 0.2, 0.9, 0.1, 0.5, 0.8, 0.4, 0.6, 0.2]
  Mask:    [1,   0,   0,   0,   0,   0,   0,   1,   0,   0  ]
  Target:  0.3 + 0.4 = 0.7

Challenge: The model must remember the first marked number across
potentially hundreds of timesteps until it sees the second marker.

Baseline: Predicting 1.0 (expected sum of two uniform [0,1] numbers)
gives MSE ≈ 0.167. Models must beat this to show learning.

Why ARU excels: The three-gate architecture allows:
- High persistence (π ≈ 1) to maintain the first number
- Selective accumulation (α) when markers appear
- Clean addition without interference from unmarked timesteps
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
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM, ManualRNN
from utils.training import count_parameters

console = Console()


def generate_adding_data(n_samples, seq_length, seed=None):
    """Generate adding problem dataset."""
    if seed is not None:
        np.random.seed(seed)
    
    # Random numbers in [0, 1]
    numbers = np.random.uniform(0, 1, (n_samples, seq_length)).astype(np.float32)
    
    # Mask with exactly two 1s per sequence
    masks = np.zeros((n_samples, seq_length), dtype=np.float32)
    for i in range(n_samples):
        # First marker in first half, second in second half
        idx1 = np.random.randint(0, seq_length // 2)
        idx2 = np.random.randint(seq_length // 2, seq_length)
        masks[i, idx1] = 1.0
        masks[i, idx2] = 1.0
    
    # Target: sum of marked numbers
    targets = (numbers * masks).sum(axis=1, keepdims=True).astype(np.float32)
    
    # Input: stack numbers and masks as 2 channels
    inputs = np.stack([numbers, masks], axis=2).astype(np.float32)
    
    return inputs, targets


def create_adding_model(model_class, hidden_size, is_aru=False):
    """Create model for adding problem (regression, 2D input)."""
    if is_aru:
        model = model_class(
            input_size=2,
            hidden_size=hidden_size,
            num_classes=1,
            dropout=0.0,
            use_embedding=False
        )
    else:
        model = model_class(
            input_size=2,
            hidden_size=hidden_size,
            num_classes=1,
            dropout=0.0,
            use_embedding=False
        )
    return model


def run_adding_benchmark(config: dict, seed: int = 42):
    """Run adding problem benchmark with multiple seeds for statistical significance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    
    seq_length = config['seq_length']
    num_runs = config.get('num_runs', 1)
    console.print(f"[cyan]Sequence length:[/cyan] {seq_length}")
    console.print(f"[cyan]Number of runs:[/cyan] {num_runs}")
    console.print(f"[cyan]Baseline MSE (predict 1.0):[/cyan] ~0.167\n")
    
    # Generate data (same for all runs)
    console.print("[cyan]Generating data...[/cyan]")
    train_x, train_y = generate_adding_data(config['train_samples'], seq_length, seed)
    val_x, val_y = generate_adding_data(config['val_samples'], seq_length, seed + 1)
    test_x, test_y = generate_adding_data(config['test_samples'], seq_length, seed + 2)
    
    console.print(f"[green]✓[/green] Generated {config['train_samples']:,} train, {config['val_samples']:,} val, {config['test_samples']:,} test samples\n")
    
    # Models
    all_models = [
        ('ARU', ARU, True),
        ('GRU', ManualGRU, False),
        ('LSTM', ManualLSTM, False),
        ('RNN', ManualRNN, False)
    ]
    
    if config.get('model_filter'):
        models = [(n, c, a) for n, c, a in all_models if n == config.get('model_filter')]
    else:
        models = all_models
    
    # Store results across all runs
    all_results = {name: {'test_mse': [], 'train_time': []} for name, _, _ in models}
    
    # Run each model multiple times
    for run_idx in range(num_runs):
        run_seed = seed + 1000 * (run_idx + 1)
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Run {run_idx + 1}/{num_runs} (seed={run_seed})[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
        
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)
        
        train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        for name, model_class, is_aru in models:
            try:
                model = create_adding_model(model_class, config['hidden_size'], is_aru).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=config['lr'])
                
                params = count_parameters(model)
                
                if run_idx == 0:
                    console.print(f"\n[bold cyan]Training {name}[/bold cyan] ({params:,} params)")
                else:
                    console.print(f"\n[cyan]Training {name}[/cyan]")
                
                best_val_mse = float('inf')
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
                    
                    for epoch in range(config['epochs']):
                        model.train()
                        train_loss = 0
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            out = model(x)
                            loss = criterion(out, y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                            train_loss += loss.item()
                        
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for x, y in val_loader:
                                x, y = x.to(device), y.to(device)
                                out = model(x)
                                val_loss += criterion(out, y).item()
                        val_mse = val_loss / len(val_loader)
                        
                        if val_mse < best_val_mse:
                            best_val_mse = val_mse
                            best_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        progress.update(
                            task, advance=1,
                            description=f"[cyan]{name} - MSE: {val_mse:.4f} (Best: {best_val_mse:.4f})"
                        )
                        
                        if patience_counter >= config['patience']:
                            break
                
                train_time = time.time() - start_time
                
                model.load_state_dict(best_state)
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        test_loss += criterion(out, y).item()
                test_mse = test_loss / len(test_loader)
                
                all_results[name]['test_mse'].append(test_mse)
                all_results[name]['train_time'].append(train_time)
                all_results[name]['params'] = params
                
                status = "✓ SOLVED" if test_mse < 0.01 else ("Learning" if test_mse < 0.167 else "Failed")
                console.print(f"[green]OK[/green] {name}: MSE={test_mse:.4f} [{status}]")
                
            except Exception as e:
                console.print(f"[red]Error training {name}:[/red] {e}")
                import traceback
                traceback.print_exc()
    
    # Compute statistics and display results
    console.print("\n" + "="*80)
    console.print("[bold magenta]STATISTICAL RESULTS[/bold magenta]")
    console.print("="*80 + "\n")
    
    table = Table(title=f"Adding Problem Results (T={seq_length}, N={num_runs} runs)", header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MSE", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Time (s)", justify="right")
    
    for name in ['RNN', 'GRU', 'ARU', 'LSTM']:
        if name not in all_results or not all_results[name]['test_mse']:
            continue
        
        data = all_results[name]
        mse_mean = np.mean(data['test_mse'])
        mse_std = np.std(data['test_mse'], ddof=1) if num_runs > 1 else 0.0
        time_mean = np.mean(data['train_time'])
        
        solved = mse_mean < 0.01
        status = "✓ Solved" if solved else ("Learning" if mse_mean < 0.167 else "Failed")
        color = "green" if solved else ("yellow" if mse_mean < 0.167 else "red")
        
        if num_runs > 1:
            mse_str = f"{mse_mean:.4f} ± {mse_std:.4f}"
        else:
            mse_str = f"{mse_mean:.4f}"
        
        table.add_row(
            name,
            f"{data['params']:,}",
            mse_str,
            f"[{color}]{status}[/{color}]",
            f"{time_mean:.1f}"
        )
    
    console.print(table)
    console.print("\n[dim]Baseline (predict 1.0): MSE ≈ 0.167 | Solved: MSE < 0.01[/dim]")
    
    if num_runs > 1:
        console.print("\n[bold cyan]Statistical Significance Tests (vs GRU baseline)[/bold cyan]\n")
        from utils.stats import format_comparison_table
        mse_dict = {name: data['test_mse'] for name, data in all_results.items() if data['test_mse']}
        console.print(format_comparison_table(mse_dict, "MSE", baseline='GRU', lower_is_better=True))
    
    if all_results['ARU']['test_mse'] and all_results['GRU']['test_mse']:
        aru_mean = np.mean(all_results['ARU']['test_mse'])
        gru_mean = np.mean(all_results['GRU']['test_mse'])
        if aru_mean < gru_mean:
            improvement = (gru_mean - aru_mean) / gru_mean * 100
            console.print(f"\n[green]ARU beats GRU by {improvement:.1f}% lower MSE[/green]")
    
    console.print(f"\n[green]Benchmark complete![/green]")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Phase 5: Adding Problem Benchmark')
    parser.add_argument('--seq-length', type=int, default=200, help='Sequence length (default: 200)')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--model', type=str, choices=['ARU', 'GRU', 'LSTM', 'RNN'], help='Single model')
    parser.add_argument('--long', action='store_true', help='Use T=500 (harder)')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs with different seeds')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    args = parser.parse_args()
    
    # Reduce samples for longer sequences
    if args.long:
        train_samples, val_samples, test_samples = 5000, 500, 500
    else:
        train_samples, val_samples, test_samples = 10000, 1000, 1000
    
    config = {
        'seq_length': 500 if args.long else args.seq_length,
        'hidden_size': args.hidden_size,
        'batch_size': 128,
        'epochs': args.epochs,
        'lr': 0.001,
        'patience': 10,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'model_filter': args.model,
        'num_runs': args.num_runs,
    }
    
    console.print(Panel.fit(
        "[bold cyan]Phase 5: Adding Problem[/bold cyan]\n"
        f"[yellow]Sequence Length: {config['seq_length']} | Runs: {config['num_runs']}[/yellow]\n"
        "[dim]Sum two marked numbers - tests long-term memory[/dim]",
        border_style="blue"
    ))
    
    run_adding_benchmark(config, seed=args.seed)


if __name__ == "__main__":
    main()
