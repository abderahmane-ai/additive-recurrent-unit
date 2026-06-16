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
from utils.training import count_parameters, set_seed

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
    """Run Copy Task benchmark with multiple seeds for statistical significance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}\n")
    
    num_runs = config.get('num_runs', 1)
    console.print(f"[cyan]Number of runs:[/cyan] {num_runs}\n")
    
    # Generate data (same for all runs)
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
    console.print(f"  Train: {len(train_inputs):,} | Val: {len(val_inputs):,} | Test: {len(test_inputs):,}\n")
    
    # Input size = num_symbols + 2 (blank=0, symbols=1-8, marker=9)
    input_size = config['num_symbols'] + 2
    output_size = config['num_symbols'] + 2
    
    # Store results across all runs
    all_results = {
        'ARU': {'test_seq_acc': [], 'test_sym_acc': [], 'train_time': []},
        'GRU': {'test_seq_acc': [], 'test_sym_acc': [], 'train_time': []},
        'LSTM': {'test_seq_acc': [], 'test_sym_acc': [], 'train_time': []},
        'RNN': {'test_seq_acc': [], 'test_sym_acc': [], 'train_time': []},
    }
    
    # Run each model multiple times
    for run_idx in range(num_runs):
        run_seed = seed + 1000 * (run_idx + 1)
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]Run {run_idx + 1}/{num_runs} (seed={run_seed})[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]\n")
        
        # Set seed for reproducibility
        set_seed(run_seed)

        models = {
            'ARU': ARU(input_size, config['hidden_size'], num_classes=output_size, dropout=config['dropout'], use_embedding=True),
            'GRU': ManualGRU(input_size, config['hidden_size'], num_classes=output_size, dropout=config['dropout'], use_embedding=True),
            'LSTM': ManualLSTM(input_size, config['hidden_size'], num_classes=output_size, dropout=config['dropout'], use_embedding=True),
            'RNN': ManualRNN(input_size, config['hidden_size'], num_classes=output_size, dropout=config['dropout'], use_embedding=True),
        }
        
        for name, model in models.items():
            try:
                model = model.to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=config['lr'])
                
                params = count_parameters(model)
                
                if run_idx == 0:
                    console.print(f"\n[bold cyan]Training {name}[/bold cyan] ({params:,} params)")
                else:
                    console.print(f"\n[cyan]Training {name}[/cyan]")
                
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
                            description=f"[cyan]{name} - Seq: {val_seq_acc:.1f}% (Best: {best_val_seq_acc:.1f}%)"
                        )
                        
                        if patience_counter >= config['patience']:
                            break
                
                train_time = time.time() - start_time
                
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                
                model.eval()
                test_loss, test_seq_acc, test_sym_acc = evaluate_copy(
                    model, test_inputs, test_targets, criterion, device,
                    config['batch_size'], config
                )
                
                all_results[name]['test_seq_acc'].append(test_seq_acc)
                all_results[name]['test_sym_acc'].append(test_sym_acc)
                all_results[name]['train_time'].append(train_time)
                all_results[name]['params'] = params
                
                console.print(f"[green]OK[/green] {name}: Seq={test_seq_acc:.1f}%, Sym={test_sym_acc:.1f}%")
            
            except Exception as e:
                console.print(f"[bold red]Error training {name}:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute statistics and display results
    console.print("\n" + "="*80)
    console.print("[bold magenta]STATISTICAL RESULTS[/bold magenta]")
    console.print("="*80 + "\n")
    
    table = Table(
        title=f"Copy Task Results (T={config['seq_length']}, D={config['delay_length']}, N={num_runs} runs)",
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Sequence Acc", justify="right")
    table.add_column("Symbol Acc", justify="right")
    table.add_column("Time (s)", justify="right")
    
    for name in ['RNN', 'GRU', 'ARU', 'LSTM']:
        if not all_results[name]['test_seq_acc']:
            continue
        
        data = all_results[name]
        seq_mean = np.mean(data['test_seq_acc'])
        seq_std = np.std(data['test_seq_acc'], ddof=1) if num_runs > 1 else 0.0
        sym_mean = np.mean(data['test_sym_acc'])
        sym_std = np.std(data['test_sym_acc'], ddof=1) if num_runs > 1 else 0.0
        time_mean = np.mean(data['train_time'])
        
        if num_runs > 1:
            seq_str = f"{seq_mean:.1f} ± {seq_std:.1f}"
            sym_str = f"{sym_mean:.1f} ± {sym_std:.1f}"
        else:
            seq_str = f"{seq_mean:.1f}"
            sym_str = f"{sym_mean:.1f}"
        
        table.add_row(
            name,
            f"{data['params']:,}",
            seq_str,
            sym_str,
            f"{time_mean:.0f}"
        )
    
    console.print(table)
    
    if num_runs > 1:
        console.print("\n[bold cyan]Statistical Significance Tests (vs GRU baseline)[/bold cyan]\n")
        from utils.stats import format_comparison_table
        
        seq_dict = {name: data['test_seq_acc'] for name, data in all_results.items() if data['test_seq_acc']}
        console.print("[yellow]Sequence Accuracy Comparison:[/yellow]")
        console.print(format_comparison_table(seq_dict, "Seq Acc", baseline='GRU', lower_is_better=False))
        
        console.print("\n[yellow]Symbol Accuracy Comparison:[/yellow]")
        sym_dict = {name: data['test_sym_acc'] for name, data in all_results.items() if data['test_sym_acc']}
        console.print(format_comparison_table(sym_dict, "Sym Acc", baseline='GRU', lower_is_better=False))
    
    if all_results['ARU']['test_seq_acc'] and all_results['GRU']['test_seq_acc']:
        aru_mean = np.mean(all_results['ARU']['test_seq_acc'])
        gru_mean = np.mean(all_results['GRU']['test_seq_acc'])
        diff = aru_mean - gru_mean
        console.print(f"\n[cyan]ARU vs GRU:[/cyan] {diff:+.1f}% difference")
        if diff > 5:
            console.print("[green]ARU's three-gate architecture enables superior long-term memory![/green]")
    
    console.print(f"\n[green]Benchmark complete![/green]")
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Copy Task Benchmark with ARU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--seq-length', type=int, default=10, help='Sequence length to copy')
    parser.add_argument('--delay-length', type=int, default=50, help='Delay period length')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs with different seeds')
    args = parser.parse_args()
    
    if args.seq_length:
        COPY_TASK_CONFIG['seq_length'] = args.seq_length
    if args.delay_length:
        COPY_TASK_CONFIG['delay_length'] = args.delay_length
    if args.epochs:
        COPY_TASK_CONFIG['epochs'] = args.epochs
    
    COPY_TASK_CONFIG['num_runs'] = args.num_runs
    
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]ARU Phase 2: Copy Task Benchmark[/bold cyan]\n"
        f"[yellow]Long-Term Memory Test | Runs: {args.num_runs}[/yellow]\n"
        f"[dim]Seq Length: {COPY_TASK_CONFIG['seq_length']} | "
        f"Delay: {COPY_TASK_CONFIG['delay_length']} | "
        f"Total: {COPY_TASK_CONFIG['seq_length'] * 2 + COPY_TASK_CONFIG['delay_length']} timesteps[/dim]",
        border_style="blue"
    ))
    
    run_copy_task_benchmark(COPY_TASK_CONFIG, seed=args.seed)

if __name__ == "__main__":
    main()
