#!/usr/bin/env python3
"""
Phase 4: Sparse Event Counting Benchmark with ARU

This benchmark tests ARU on a pure counting task that explicitly requires
additive accumulation. The task involves counting rare events distributed
across long sequences (500-2000 timesteps).

Core Challenge: Count the total occurrences of 3-5 different event types
scattered sparsely (1-2% density) across the sequence. The model must:
1. Detect each event type independently
2. Maintain all counts simultaneously without interference
3. Sum counts at the end to produce final prediction

Why ARU Should Excel:
- High π (≈1): Maintains running counts without decay
- High υ (≈1): Adds +1 to appropriate counter when event detected
- True additive accumulation: count_t = count_{t-1} + detected_event_t
- GRU/LSTM forced averaging makes precise counting difficult

Example Task:
- Input: [0,0,1,0,0,0,2,0,0,3,0,0,1,0,...] (length 1000)
- Events: type-1 appears 12 times, type-2 appears 8 times, type-3 appears 15 times
- Target: Total count = 35
- Or multi-class: [12, 8, 15] for per-event-type counts
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

COUNTING_CONFIG = {
    'hidden_size': 64,
    'batch_size': 128,  # Larger batches for faster training
    'epochs': 25,  # Reduced from 50
    'lr': 0.005,  # Higher learning rate for faster convergence
    'dropout': 0.1,
    'patience': 8,  # Reduced patience
    'sequence_length': 500,  # Shorter sequences
    'num_sequences': 5000,  # Reduced from 10000
    'num_event_types': 3,
    'event_density': 0.03,  # Higher density (3% instead of 1.5%)
    'max_count': 50,
    'task_type': 'total_count',
}

def generate_counting_sequence(length, num_event_types, density, seed=None):
    """
    Generate a sparse event sequence for counting.
    
    Args:
        length: Sequence length
        num_event_types: Number of different event types (1 to num_event_types)
        density: Probability of event at each timestep
        seed: Random seed for reproducibility
        
    Returns:
        sequence: Array of integers (0=no event, 1-N=event type)
        counts: Array of counts per event type
        total_count: Total number of events
    """
    if seed is not None:
        np.random.seed(seed)
    
    sequence = np.zeros(length, dtype=np.int64)
    counts = np.zeros(num_event_types, dtype=np.int64)
    
    # Decide which positions have events
    event_mask = np.random.random(length) < density
    num_events = event_mask.sum()
    
    if num_events > 0:
        # Assign random event types to event positions
        event_positions = np.where(event_mask)[0]
        event_types = np.random.randint(1, num_event_types + 1, size=num_events)
        sequence[event_positions] = event_types
        
        # Count each event type
        for event_type in range(1, num_event_types + 1):
            counts[event_type - 1] = (event_types == event_type).sum()
    
    total_count = counts.sum()
    
    return sequence, counts, total_count

def generate_dataset(config):
    """
    Generate counting dataset.
    
    Returns:
        sequences: (num_sequences, seq_length) integer array
        targets: (num_sequences,) or (num_sequences, num_event_types) depending on task
    """
    console.print(f"[cyan]Generating {config['num_sequences']} counting sequences...[/cyan]")
    
    sequences = []
    targets = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating sequences", total=config['num_sequences'])
        
        for i in range(config['num_sequences']):
            seq, counts, total = generate_counting_sequence(
                config['sequence_length'],
                config['num_event_types'],
                config['event_density'],
                seed=None
            )
            sequences.append(seq)
            
            if config['task_type'] == 'total_count':
                targets.append(total)
            else:  # per_type_count
                targets.append(counts)
            
            progress.update(task, advance=1)
    
    sequences = np.array(sequences, dtype=np.int64)
    targets = np.array(targets, dtype=np.float32)
    
    console.print(f"[green]✓[/green] Generated {config['num_sequences']} sequences")
    console.print(f"  Sequence length: {config['sequence_length']} timesteps")
    console.print(f"  Event types: {config['num_event_types']}")
    console.print(f"  Event density: {config['event_density']*100:.1f}%")
    console.print(f"  Task: {config['task_type']}")
    
    # Statistics
    if config['task_type'] == 'total_count':
        console.print(f"  Avg count per sequence: {targets.mean():.1f} (std: {targets.std():.1f})")
        console.print(f"  Count range: [{targets.min():.0f}, {targets.max():.0f}]")
    
    return sequences, targets

def load_counting_dataset(config):
    """Load or generate counting dataset."""
    sequences, targets = generate_dataset(config)
    
    sequences_tensor = torch.from_numpy(sequences)
    targets_tensor = torch.from_numpy(targets)
    
    num_sequences = len(sequences)
    train_size = int(0.7 * num_sequences)
    valid_size = int(0.15 * num_sequences)
    
    train_sequences = sequences_tensor[:train_size]
    train_targets = targets_tensor[:train_size]
    
    valid_sequences = sequences_tensor[train_size:train_size + valid_size]
    valid_targets = targets_tensor[train_size:train_size + valid_size]
    
    test_sequences = sequences_tensor[train_size + valid_size:]
    test_targets = targets_tensor[train_size + valid_size:]
    
    console.print(f"[green]✓[/green] Split dataset")
    console.print(f"  Train: {len(train_sequences)} sequences")
    console.print(f"  Valid: {len(valid_sequences)} sequences")
    console.print(f"  Test: {len(test_sequences)} sequences")
    
    return (train_sequences, train_targets), (valid_sequences, valid_targets), (test_sequences, test_targets)

def compute_counting_metrics(predictions, targets):
    """
    Compute counting metrics.
    
    Args:
        predictions: Model predictions (counts)
        targets: Ground truth counts
        
    Returns:
        Dictionary of metrics
    """
    preds_np = predictions.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Mean Absolute Error
    mae = np.abs(preds_np - targets_np).mean()
    
    # Mean Squared Error
    mse = ((preds_np - targets_np) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # Exact match accuracy (for integers)
    preds_rounded = np.round(preds_np)
    exact_match = (preds_rounded == targets_np).mean() * 100
    
    # Within-1 accuracy
    within_1 = (np.abs(preds_rounded - targets_np) <= 1).mean() * 100
    
    # Within-2 accuracy
    within_2 = (np.abs(preds_rounded - targets_np) <= 2).mean() * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'exact_match': exact_match,
        'within_1': within_1,
        'within_2': within_2,
    }

def train_epoch_counting(model, sequences, targets, criterion, optimizer, device, batch_size=64):
    """Train epoch for counting task."""
    model.train()
    
    num_sequences = sequences.size(0)
    indices = torch.randperm(num_sequences)
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    for i in range(0, num_sequences, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_sequences = sequences[batch_indices].to(device)
        batch_targets = targets[batch_indices].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_sequences)
        
        # Handle different output shapes
        if outputs.dim() > 1 and outputs.size(-1) == 1:
            outputs = outputs.squeeze(-1)
        
        # Ensure outputs are positive (ReLU) since counts are non-negative
        outputs = torch.nn.functional.relu(outputs)
        
        loss = criterion(outputs, batch_targets)
        
        # Check for NaN
        if torch.isnan(loss):
            console.print(f"[red]NaN detected in loss! Outputs: {outputs[:3]}, Targets: {batch_targets[:3]}[/red]")
            continue
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        all_predictions.append(outputs.detach())
        all_targets.append(batch_targets)
    
    if num_batches == 0:
        return float('nan'), {'mae': float('nan'), 'rmse': float('nan'), 'exact_match': 0, 'within_1': 0, 'within_2': 0}
    
    avg_loss = total_loss / num_batches
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_counting_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics

def evaluate_counting(model, sequences, targets, criterion, device, batch_size=64):
    """Evaluate counting model."""
    model.eval()
    
    num_sequences = sequences.size(0)
    
    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, num_sequences, batch_size):
            batch_sequences = sequences[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            
            outputs = model(batch_sequences)
            
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            
            # Ensure outputs are positive
            outputs = torch.nn.functional.relu(outputs)
            
            loss = criterion(outputs, batch_targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.append(outputs)
            all_targets.append(batch_targets)
    
    avg_loss = total_loss / num_batches
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_counting_metrics(all_predictions, all_targets)
    
    return avg_loss, metrics

def run_counting_benchmark(config: dict, seed: int = 42, aru_only: bool = False):
    """Run Sparse Event Counting benchmark."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}\n")
    
    try:
        (train_sequences, train_targets), (valid_sequences, valid_targets), (test_sequences, test_targets) = \
            load_counting_dataset(config)
    except Exception as e:
        console.print(f"[bold red]Error loading dataset:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Input size = number of event types + 1 (for no-event)
    input_size = config['num_event_types'] + 1
    
    # Output size depends on task
    if config['task_type'] == 'total_count':
        output_size = 1
    else:
        output_size = config['num_event_types']
    
    models = {
        'ARU': ARU(
            input_size,
            config['hidden_size'],
            num_classes=output_size,
            dropout=config['dropout'],
            use_embedding=True,
            persistence_init=3.0,  # Higher init for maintaining counts
            accumulation_init=-1.0,  # Start with lower accumulation gate
        ),
    }
    
    if not aru_only:
        models.update({
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
        })
    
    results = {}
    
    for name, model in models.items():
        try:
            model = model.to(device)
            
            criterion = nn.MSELoss()  # Regression task
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=4
            )
            
            params = count_parameters(model)
            console.print(f"\n[bold cyan]Training {name}[/bold cyan]")
            console.print(f"[yellow]Parameters:[/yellow] {params:,}")
            
            # Format model specs with proper indentation
            model_str = str(model)
            lines = model_str.split('\n')
            console.print(f"[green]Specs:[/green] {lines[0]}")
            for line in lines[1:]:
                console.print(f"  {line}")
            
            best_val_mae = float('inf')
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
                    train_loss, train_metrics = train_epoch_counting(
                        model, train_sequences, train_targets, criterion, optimizer, device,
                        batch_size=config['batch_size']
                    )
                    val_loss, val_metrics = evaluate_counting(
                        model, valid_sequences, valid_targets, criterion, device,
                        batch_size=config['batch_size']
                    )
                    
                    scheduler.step(val_metrics['mae'])
                    
                    if val_metrics['mae'] < best_val_mae:
                        best_val_mae = val_metrics['mae']
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    progress.update(
                        task, advance=1,
                        description=f"[cyan]{name} - Val MAE: {val_metrics['mae']:.2f} (Best: {best_val_mae:.2f})"
                    )
                    
                    if patience_counter >= config['patience']:
                        console.print(f"[yellow]Early stopping at epoch {epoch}[/yellow]")
                        break
            
            train_time = time.time() - start_time
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            test_loss, test_metrics = evaluate_counting(
                model, test_sequences, test_targets, criterion, device,
                batch_size=config['batch_size']
            )
            
            # Show sample predictions on new test data
            console.print(f"\n[bold yellow]Testing {name} on new counting sequences:[/bold yellow]")
            with torch.no_grad():
                sample_seqs = test_sequences[:5].to(device)
                sample_targets = test_targets[:5].to(device)
                sample_outputs = model(sample_seqs)
                sample_preds = sample_outputs.squeeze(-1)
                
                for i in range(5):
                    true_count = sample_targets[i].item()
                    pred_count = sample_preds[i].item()
                    error = abs(pred_count - true_count)
                    status = "✓" if error < 0.5 else "✗"
                    
                    # Count actual events in sequence (non-zero values)
                    events = (sample_seqs[i] > 0).sum().item()
                    
                    console.print(f"  {status} Sample {i+1}: Events={events}, "
                                f"True count={true_count:.0f}, "
                                f"Predicted={pred_count:.1f}, "
                                f"Error={error:.2f}")
            
            model.train()
            
            results[name] = {
                'params': params,
                'best_val_mae': best_val_mae,
                'test_mae': test_metrics['mae'],
                'test_rmse': test_metrics['rmse'],
                'test_exact_match': test_metrics['exact_match'],
                'test_within_1': test_metrics['within_1'],
                'test_within_2': test_metrics['within_2'],
                'train_time': train_time
            }
            
            console.print(
                f"[green]✓[/green] {name} completed - "
                f"Test MAE: {test_metrics['mae']:.2f} | "
                f"Exact: {test_metrics['exact_match']:.1f}%"
            )
            
        except Exception as e:
            console.print(f"[bold red]Error training {name}:[/bold red] {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        console.print("[bold red]No models completed training successfully[/bold red]")
        return
    
    console.print("\n")
    table = Table(
        title="🔢 Sparse Event Counting Results",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right", style="green")
    table.add_column("Test MAE", justify="right", style="yellow")
    table.add_column("Test RMSE", justify="right", style="yellow")
    table.add_column("Exact Match", justify="right", style="blue")
    table.add_column("Within ±1", justify="right", style="blue")
    table.add_column("Train Time", justify="right", style="magenta")
    
    for name, data in results.items():
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['test_mae']:.2f}",
            f"{data['test_rmse']:.2f}",
            f"{data['test_exact_match']:.1f}%",
            f"{data['test_within_1']:.1f}%",
            f"{data['train_time']:.0f}s"
        )
    
    console.print(table)
    
    winner = min(results.items(), key=lambda x: x[1]['test_mae'])
    console.print(f"\n[bold green]🏆 Winner (Lowest MAE): {winner[0]} ({winner[1]['test_mae']:.2f})[/bold green]")
    
    if 'ARU' in results and 'GRU' in results:
        aru_mae = results['ARU']['test_mae']
        gru_mae = results['GRU']['test_mae']
        improvement = ((gru_mae - aru_mae) / gru_mae) * 100
        console.print(f"\n[cyan]ARU vs GRU:[/cyan] {improvement:+.1f}% improvement (lower is better)")
        if improvement > 10:
            console.print(
                "[green]ARU's additive accumulation (π≈1, υ≈1) enables precise counting "
                "by maintaining running sums![/green]"
            )
        elif improvement > 5:
            console.print(
                "[dim]ARU shows advantage in maintaining accurate counts over long sequences[/dim]"
            )
    
    console.print(
        f"\n[bold yellow]💡 Key Insight:[/bold yellow] This is a PURE counting task.\n"
        f"   ARU can achieve additive accumulation: count_t = count_{{t-1}} + event_t\n"
        f"   when ρ ≈ 1 (no reset), π ≈ 1 (maintain count), and α ≈ 1 (add detected event).\n"
        f"   GRU is constrained by z + (1-z) = 1, forcing weighted averaging\n"
        f"   instead of true addition, making precise counting harder."
    )

def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Sparse Event Counting Benchmark with ARU'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--sequence-length', type=int, default=500,
                       help='Sequence length')
    parser.add_argument('--num-sequences', type=int, default=5000,
                       help='Total number of sequences to generate')
    parser.add_argument('--event-density', type=float, default=0.03,
                       help='Event density (probability of event per timestep)')
    parser.add_argument('--aru-only', action='store_true',
                       help='Train only ARU (skip baselines for faster testing)')
    args = parser.parse_args()
    
    if args.epochs:
        COUNTING_CONFIG['epochs'] = args.epochs
    if args.sequence_length:
        COUNTING_CONFIG['sequence_length'] = args.sequence_length
    if args.num_sequences:
        COUNTING_CONFIG['num_sequences'] = args.num_sequences
    if args.event_density:
        COUNTING_CONFIG['event_density'] = args.event_density
    
    console.clear()
    mode_str = " (ARU Only)" if args.aru_only else ""
    console.print(Panel.fit(
        f"[bold cyan]ARU Phase 4: Sparse Event Counting Benchmark{mode_str}[/bold cyan]\n"
        "[yellow]Pure Additive Accumulation Task[/yellow]\n"
        f"[dim]Epochs: {COUNTING_CONFIG['epochs']} | "
        f"Sequence Length: {COUNTING_CONFIG['sequence_length']} | "
        f"Event Density: {COUNTING_CONFIG['event_density']*100:.1f}%[/dim]",
        border_style="blue"
    ))
    
    run_counting_benchmark(COUNTING_CONFIG, seed=args.seed, aru_only=args.aru_only)

if __name__ == "__main__":
    main()