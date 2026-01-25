#!/usr/bin/env python3
"""
Phase 3: Counting Task Benchmark

Tests the ability to count occurrences of a signal in a sequence.
This is the ideal task for ARU's additive accumulation design.

Task:
- Input: Sequence of random values with some marked as "count me" (value=1)
- Output: Total count of marked values

Example (T=50):
  Input:  [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, ...]  (1s scattered randomly)
  Target: 3 (count of 1s in the sequence)

Why ARU excels:
- Pure additive accumulation: h_t = h_{t-1} + v_t when π≈1, α≈1
- Each "1" should add exactly +1 to the hidden state
- No interference from "0" values when α≈0
- This is literally what ARU was designed for

Baseline: Predicting mean count gives high MSE. Models must learn to count.
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


def generate_counting_data(n_samples, seq_length, density=0.1, seed=None):
    """
    Generate counting task dataset.
    
    Args:
        n_samples: Number of sequences
        seq_length: Length of each sequence
        density: Probability of a "1" at each position (default 10%)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate binary sequences with given density
    inputs = (np.random.rand(n_samples, seq_length) < density).astype(np.float32)
    
    # Target is the count of 1s
    targets = inputs.sum(axis=1, keepdims=True).astype(np.float32)
    
    # Reshape inputs for RNN: (batch, seq_len, 1)
    inputs = inputs[:, :, np.newaxis]
    
    return inputs, targets


def run_counting_benchmark(config: dict, seed: int = 42):
    """Run counting task benchmark with multiple seeds for statistical significance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    
    seq_length = config['seq_length']
    density = config['density']
    expected_count = seq_length * density
    num_runs = config.get('num_runs', 1)
    
    console.print(f"[cyan]Sequence length:[/cyan] {seq_length}")
    console.print(f"[cyan]Signal density:[/cyan] {density:.0%}")
    console.print(f"[cyan]Expected count:[/cyan] ~{expected_count:.1f}")
    console.print(f"[cyan]Number of runs:[/cyan] {num_runs}")
    
    # Generate data (same for all runs for fair comparison)
    console.print("\n[cyan]Generating data...[/cyan]")
    train_x, train_y = generate_counting_data(
        config['train_samples'], seq_length, density, seed
    )
    val_x, val_y = generate_counting_data(
        config['val_samples'], seq_length, density, seed + 1
    )
    test_x, test_y = generate_counting_data(
        config['test_samples'], seq_length, density, seed + 2
    )
    
    # Compute baseline MSE (predicting mean)
    mean_count = train_y.mean()
    baseline_mse = ((test_y - mean_count) ** 2).mean()
    console.print(f"[yellow]Baseline MSE (predict mean={mean_count:.1f}):[/yellow] {baseline_mse:.4f}")
    
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    console.print(f"[green]OK[/green] Generated {config['train_samples']:,} train, {config['val_samples']:,} val, {config['test_samples']:,} test\n")
    
    # Models (ordered for fair comparison)
    all_models = [
        ('RNN', ManualRNN, False),
        ('GRU', ManualGRU, False),
        ('ARU', ARU, True),
        ('LSTM', ManualLSTM, False),
    ]
    
    if config.get('model_filter'):
        models = [(n, c, a) for n, c, a in all_models if n == config['model_filter']]
    else:
        models = all_models
    
    # Store results across all runs
    all_results = {name: {'test_mse': [], 'test_mae': [], 'train_time': []} for name, _, _ in models}
    
    # Run each model multiple times
    for run_idx in range(num_runs):
        run_seed = seed + 1000 * (run_idx + 1)
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Run {run_idx + 1}/{num_runs} (seed={run_seed})[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
        
        for name, model_class, is_aru in models:
            try:
                # Set seed for this run
                torch.manual_seed(run_seed)
                np.random.seed(run_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(run_seed)
                
                model = model_class(
                    input_size=1,
                    hidden_size=config['hidden_size'],
                    num_classes=1,
                    dropout=0.0,
                    use_embedding=False
                ).to(device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=config['lr'])
                
                params = count_parameters(model)
                
                if run_idx == 0:
                    console.print(f"[bold cyan]Training {name}[/bold cyan] ({params:,} params)")
                else:
                    console.print(f"[cyan]Training {name}[/cyan]")
                
                best_val_mse = float('inf')
                best_state = None
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
                        # Train
                        model.train()
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            optimizer.zero_grad()
                            out = model(x)
                            loss = criterion(out, y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        
                        # Validate
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for x, y in val_loader:
                                x, y = x.to(device), y.to(device)
                                val_loss += criterion(model(x), y).item()
                        val_mse = val_loss / len(val_loader)
                        
                        if val_mse < best_val_mse:
                            best_val_mse = val_mse
                            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
                
                # Test
                model.load_state_dict(best_state)
                model.to(device)
                model.eval()
                
                test_loss = 0
                all_preds, all_targets = [], []
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        pred = model(x)
                        test_loss += criterion(pred, y).item()
                        all_preds.append(pred.cpu())
                        all_targets.append(y.cpu())
                
                test_mse = test_loss / len(test_loader)
                all_preds = torch.cat(all_preds)
                all_targets = torch.cat(all_targets)
                
                # Compute MAE (more interpretable for counting)
                mae = (all_preds - all_targets).abs().mean().item()
                
                # Store results
                all_results[name]['test_mse'].append(test_mse)
                all_results[name]['test_mae'].append(mae)
                all_results[name]['train_time'].append(train_time)
                all_results[name]['params'] = params
                
                console.print(f"[green]OK[/green] {name}: MSE={test_mse:.4f}, MAE={mae:.2f}\n")
                
            except Exception as e:
                console.print(f"[red]Error training {name}:[/red] {e}")
                import traceback
                traceback.print_exc()
    
    # Compute statistics and display results
    console.print("\n" + "="*80)
    console.print("[bold magenta]STATISTICAL RESULTS[/bold magenta]")
    console.print("="*80 + "\n")
    
    # Results table with statistics
    table = Table(title=f"Counting Task Results (T={seq_length}, density={density:.0%}, N={num_runs} runs)", 
                  header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MSE", justify="right")
    table.add_column("Test MAE", justify="right")
    table.add_column("Time (s)", justify="right")
    
    for name in ['RNN', 'GRU', 'ARU', 'LSTM']:
        if name not in all_results or not all_results[name]['test_mse']:
            continue
        
        data = all_results[name]
        mse_mean = np.mean(data['test_mse'])
        mse_std = np.std(data['test_mse'], ddof=1) if num_runs > 1 else 0.0
        mae_mean = np.mean(data['test_mae'])
        mae_std = np.std(data['test_mae'], ddof=1) if num_runs > 1 else 0.0
        time_mean = np.mean(data['train_time'])
        
        if num_runs > 1:
            mse_str = f"{mse_mean:.4f} ± {mse_std:.4f}"
            mae_str = f"{mae_mean:.2f} ± {mae_std:.2f}"
        else:
            mse_str = f"{mse_mean:.4f}"
            mae_str = f"{mae_mean:.2f}"
        
        table.add_row(
            name,
            f"{data['params']:,}",
            mse_str,
            mae_str,
            f"{time_mean:.1f}"
        )
    
    console.print(table)
    console.print(f"\n[dim]Baseline MSE (predict mean): {baseline_mse:.4f}[/dim]")
    
    # Statistical significance tests
    if num_runs > 1:
        console.print("\n[bold cyan]Statistical Significance Tests (vs GRU baseline)[/bold cyan]\n")
        
        from utils.stats import format_comparison_table
        
        # MSE comparison
        mse_dict = {name: data['test_mse'] for name, data in all_results.items() if data['test_mse']}
        console.print("[yellow]Test MSE Comparison:[/yellow]")
        console.print(format_comparison_table(mse_dict, "MSE", baseline='GRU', lower_is_better=True))
        
        # MAE comparison
        console.print("\n[yellow]Test MAE Comparison:[/yellow]")
        mae_dict = {name: data['test_mae'] for name, data in all_results.items() if data['test_mae']}
        console.print(format_comparison_table(mae_dict, "MAE", baseline='GRU', lower_is_better=True))
    
    if 'ARU' in all_results and all_results['ARU']['test_mae']:
        aru_mae_mean = np.mean(all_results['ARU']['test_mae'])
        console.print(f"\n[bold]ARU average counting error: {aru_mae_mean:.2f}[/bold]")
        if aru_mae_mean < 1.0:
            console.print("[green]Excellent! ARU counts nearly perfectly.[/green]")
        elif aru_mae_mean < 2.0:
            console.print("[green]Good counting accuracy.[/green]")
    
    console.print(f"\n[green]Benchmark complete![/green] Results ready for manual report update.")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Counting Task Benchmark')
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--density', type=float, default=0.1, help='Signal density (0-1)')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden size')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs')
    parser.add_argument('--model', type=str, choices=['ARU', 'GRU', 'LSTM', 'RNN'], help='Single model')
    parser.add_argument('--long', action='store_true', help='Use T=300 (harder)')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs with different seeds')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    args = parser.parse_args()
    
    config = {
        'seq_length': 300 if args.long else args.seq_length,
        'density': args.density,
        'hidden_size': args.hidden_size,
        'batch_size': 128,
        'epochs': args.epochs,
        'lr': 0.001,
        'patience': 8,
        'train_samples': 8000,
        'val_samples': 1000,
        'test_samples': 1000,
        'model_filter': args.model,
        'num_runs': args.num_runs,
    }
    
    console.print(Panel.fit(
        "[bold cyan]Phase 3: Counting Task[/bold cyan]\n"
        f"[yellow]Sequence Length: {config['seq_length']} | Density: {config['density']:.0%} | Runs: {config['num_runs']}[/yellow]\n"
        "[dim]Count occurrences of 1s - tests additive accumulation[/dim]",
        border_style="blue"
    ))
    
    run_counting_benchmark(config, seed=args.seed)


if __name__ == "__main__":
    main()
