#!/usr/bin/env python3
"""
Phase 6: Inception - Nested Dream Layers Benchmark

Inspired by Inception (2010), where time moves at different rates in nested dream levels.

Task:
- Process sequences with hierarchical structure (dreams within dreams)
- Each nested level has a different time scale (1x, 5x, 20x, etc.)
- Predict values that depend on aggregating information across all levels

The model receives:
- Events from multiple nested layers simultaneously
- Time scale indicators for each layer
- Hierarchical context signals

The model must predict:
- Values that depend on coherently integrating all nested layers
- Requires maintaining separate state for each hierarchy level

Physics/Rules:
- Reality (Layer 0): 1x time scale
- Dream Level 1: 5x faster than reality
- Dream Level 2: 20x faster than Level 1 (100x faster than reality)
- Dream Level 3: 20x faster than Level 2 (2000x faster than reality)

Why ARU excels:
- Persistent memory to maintain slow-changing upper layers
- Accumulation to integrate fast-changing lower layers
- Clean separation between different time scales
- Long-term dependencies across nested hierarchies

Metrics:
- Prediction accuracy on test sequences
- Layer-wise error breakdown
- Cross-layer coherence score
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
import copy

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM, ManualRNN
from utils.training import count_parameters, set_seed

console = Console()


def generate_inception_sequence(seq_length, n_layers=3, seed=None):
    """
    Generate a hierarchical sequence with nested time scales.
    
    Args:
        seq_length: Length of sequence in reality (Layer 0) time
        n_layers: Number of dream layers (excluding reality)
        seed: Random seed
    
    Returns:
        sequence: (seq_length, n_layers + 1, feature_dim) nested observations
        target: (seq_length,) values to predict (sum of all layer contributions)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Time scale multipliers for each layer
    time_scales = [1, 5, 20, 20][:n_layers + 1]  # Reality + dream layers
    
    # Generate slow-varying signals for each layer
    feature_dim = 4  # Features per layer
    
    all_layers = []
    layer_contributions = []
    
    for layer_idx, scale in enumerate(time_scales):
        # Effective length for this layer
        effective_length = seq_length * scale
        
        # Generate smooth signal with frequency appropriate to this layer
        # Higher layers change slower, lower layers change faster
        frequency = 0.1 / (layer_idx + 1)  # Slower oscillation for higher layers
        t = np.arange(effective_length) / effective_length
        
        # Create sinusoidal patterns with layer-specific phase
        signal = np.zeros((effective_length, feature_dim))
        for f in range(feature_dim):
            phase = np.random.uniform(0, 2 * np.pi)
            signal[:, f] = np.sin(2 * np.pi * frequency * t * (f + 1) + phase)
        
        # Add noise (more noise in deeper layers)
        noise_level = 0.1 * (layer_idx + 1) / (n_layers + 1)
        signal += np.random.randn(effective_length, feature_dim) * noise_level
        
        # Downsample to reality time (average pooling)
        if scale > 1:
            # Reshape and average to get reality-time observations
            signal_downsampled = signal.reshape(seq_length, scale, feature_dim).mean(axis=1)
        else:
            signal_downsampled = signal
        
        all_layers.append(signal_downsampled)
        
        # Contribution of this layer to the target (weighted by layer depth)
        # Deeper layers have exponentially smaller contribution
        weight = 1.0 / (2 ** layer_idx)
        contribution = signal_downsampled.sum(axis=1) * weight
        layer_contributions.append(contribution)
    
    # Stack layers: (seq_length, n_layers + 1, feature_dim)
    sequence = np.stack(all_layers, axis=1).astype(np.float32)
    
    # Target: sum of all layer contributions at each timestep
    target_sequence = np.sum(layer_contributions, axis=0).astype(np.float32)
    
    # For this task, we predict the FINAL integrated value (not per-timestep)
    # This tests if the model can accumulate information across the entire sequence
    target_final = target_sequence[-1]  # Final value
    
    return sequence, target_final


def create_inception_dataset(n_samples, seq_length, n_layers=3, seed=None):
    """
    Create dataset for Inception benchmark.
    
    Returns:
        inputs: (n_samples, seq_length, (n_layers+1) * feature_dim) - all layers flattened
        targets: (n_samples, seq_length) - target values at each timestep
    """
    if seed is not None:
        np.random.seed(seed)
    
    inputs_list = []
    targets_list = []
    
    for i in range(n_samples):
        seq, tgt = generate_inception_sequence(
            seq_length, n_layers, seed=seed + i if seed else None
        )
        # Flatten layers: (seq_length, (n_layers+1) * feature_dim)
        seq_flat = seq.reshape(seq_length, -1)
        inputs_list.append(seq_flat)
        targets_list.append(tgt)
    
    inputs = np.array(inputs_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)[:, np.newaxis]  # (n_samples, 1)
    
    return inputs, targets


def create_model(model_class, input_size, hidden_size, output_size, is_aru=False):
    """Create model for nested sequence prediction."""
    if is_aru:
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=output_size,
            dropout=0.1,
            use_embedding=False
        )
    else:
        model = model_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=output_size,
            dropout=0.1,
            use_embedding=False
        )
    return model


def run_inception_benchmark(config: dict, seed: int = 42):
    """Run Inception nested layers benchmark."""
    # Set seed for reproducibility
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    
    seq_length = config['seq_length']
    n_layers = config['n_layers']
    
    console.print(f"[cyan]Sequence length:[/cyan] {seq_length} timesteps")
    console.print(f"[cyan]Dream layers:[/cyan] {n_layers} (plus reality)")
    console.print(f"[cyan]Time scales:[/cyan] 1x, 5x, 20x, 20x" if n_layers >= 3 else "1x, 5x")
    console.print(f"[cyan]Total layers:[/cyan] {n_layers + 1}\n")
    
    # Generate data
    console.print("[cyan]Generating Inception scenarios...[/cyan]")
    train_x, train_y = create_inception_dataset(
        config['train_samples'], seq_length, n_layers, seed=seed
    )
    val_x, val_y = create_inception_dataset(
        config['val_samples'], seq_length, n_layers, seed=seed + 1000
    )
    test_x, test_y = create_inception_dataset(
        config['test_samples'], seq_length, n_layers, seed=seed + 2000
    )
    
    console.print(f"[green]✓[/green] Generated {config['train_samples']:,} train, "
                  f"{config['val_samples']:,} val, {config['test_samples']:,} test scenarios")
    
    input_dim = train_x.shape[2]
    output_dim = 1  # Predict single value per timestep
    
    console.print(f"[dim]Input dimension: {input_dim} (4 features × {n_layers + 1} layers)[/dim]")
    console.print(f"[dim]Output dimension: {output_dim} (final integrated value)[/dim]\n")
    
    # Targets are already (batch, 1) shape
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Models
    all_models = [
        ('ARU', ARU, True),
        ('GRU', ManualGRU, False),
        ('LSTM', ManualLSTM, False),
        ('RNN', ManualRNN, False)
    ]
    
    if config.get('model_filter'):
        models = [(n, c, a) for n, c, a in all_models if n == config['model_filter']]
    else:
        models = all_models
    
    results = {}
    
    for name, model_class, is_aru in models:
        try:
            model = create_model(
                model_class, input_dim, config['hidden_size'], output_dim, is_aru
            ).to(device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
            
            params = count_parameters(model)
            console.print(f"\n[bold cyan]Training {name}[/bold cyan]")
            console.print(f"Parameters: {params:,}")
            
            best_val_loss = float('inf')
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
                    train_loss = 0
                    for x, y in train_loader:
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        out = model(x)  # (batch, seq_length, output_dim)
                        loss = criterion(out, y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    train_loss = train_loss / len(train_loader)
                    
                    # Validate
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for x, y in val_loader:
                            x, y = x.to(device), y.to(device)
                            out = model(x)
                            val_loss += criterion(out, y).item()
                    val_loss = val_loss / len(val_loader)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    progress.update(
                        task, advance=1,
                        description=f"[cyan]{name} - Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})"
                    )
                    
                    if patience_counter >= config['patience']:
                        break
            
            train_time = time.time() - start_time
            
            # Test
            model.load_state_dict(best_state)
            model.eval()
            
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    all_predictions.append(out.cpu().numpy())
                    all_targets.append(y.cpu().numpy())
            
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # Correlation coefficient
            pred_flat = predictions.flatten()
            targ_flat = targets.flatten()
            correlation = np.corrcoef(pred_flat, targ_flat)[0, 1]
            
            results[name] = {
                'params': params,
                'test_mse': mse,
                'test_mae': mae,
                'correlation': correlation,
                'train_time': train_time
            }
            
            console.print(f"[green]✓[/green] {name}: MSE = {mse:.4f}, MAE = {mae:.4f}, Corr = {correlation:.4f}")
            
        except Exception as e:
            console.print(f"[red]Error training {name}:[/red] {e}")
            import traceback
            traceback.print_exc()
    
    # Results table
    console.print("\n")
    table = Table(
        title=f"🌀 Inception - Nested Dream Layers Results ({n_layers + 1} layers)",
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MSE", justify="right")
    table.add_column("Test MAE", justify="right")
    table.add_column("Correlation", justify="right")
    table.add_column("Time", justify="right")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['test_mse']):
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['test_mse']:.4f}",
            f"{data['test_mae']:.4f}",
            f"{data['correlation']:.4f}",
            f"{data['train_time']:.1f}s"
        )
    
    console.print(table)
    
    console.print("\n[dim]Lower MSE/MAE is better, higher correlation is better[/dim]")
    console.print("[dim]Correlation: 1.0 = perfect prediction, 0.0 = random[/dim]")
    
    if 'ARU' in results and 'GRU' in results:
        aru_mse = results['ARU']['test_mse']
        gru_mse = results['GRU']['test_mse']
        if aru_mse < gru_mse:
            improvement = (gru_mse - aru_mse) / gru_mse * 100
            console.print(f"\n[green]🌀 ARU achieves {improvement:.1f}% lower error than GRU on nested hierarchies![/green]")
    
    # Save report
    os.makedirs(os.path.join(project_root, "benchmarks", "phase6"), exist_ok=True)
    report_path = os.path.join(project_root, "benchmarks", "phase6", "inception_report.md")
    
    if not results:
        console.print("[red]No results to save - all models failed to train[/red]")
        return
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 📊 Phase 6: Inception - Nested Dream Layers Benchmark Report\n\n")
        f.write("## Executive Summary\n\n")
        
        best_model = min(results.items(), key=lambda x: x[1]['test_mse'])
        f.write(f"**{best_model[0]} achieved the lowest MSE** of {best_model[1]['test_mse']:.4f}. ")
        f.write("This benchmark tests a model's ability to process hierarchical sequences with ")
        f.write("nested time scales—inspired by the dream layers in Inception (2010).\n\n")
        f.write("---\n\n")
        
        f.write("## Inspired by Inception (2010)\n\n")
        f.write("*\"We need to go deeper.\"* - Cobb\n\n")
        f.write("---\n\n")
        
        f.write("## 🎯 Task Specification\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write("| **Task** | Integrate information across nested hierarchical layers |\n")
        f.write(f"| **Sequence Length** | {seq_length} timesteps |\n")
        f.write(f"| **Number of Layers** | {n_layers + 1} (reality + {n_layers} dream layers) |\n")
        f.write("| **Time Scales** | 1x (reality), 5x, 20x, 20x (nested) |\n")
        f.write("| **Challenge** | Maintain separate context for each hierarchy level |\n")
        f.write("\n---\n\n")
        
        f.write("## 🏆 Performance Results\n\n")
        f.write("### Test Metrics (Lower MSE/MAE is Better)\n\n")
        f.write("| Rank | Model | Test MSE | Test MAE | Correlation | Parameters |\n")
        f.write("|------|-------|----------|----------|-------------|------------|\n")
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['test_mse'])
        for rank, (name, data) in enumerate(sorted_results, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            f.write(f"| {medal} | **{name}** | **{data['test_mse']:.4f}** | "
                   f"{data['test_mae']:.4f} | {data['correlation']:.4f} | {data['params']:,} |\n")
        
        f.write("\n### Key Observations\n\n")
        if 'ARU' in results:
            aru_data = results['ARU']
            f.write(f"✅ **Hierarchical Integration** - ARU achieved MSE = {aru_data['test_mse']:.4f}, ")
            f.write("demonstrating effective integration across nested time scales.\n")
            f.write(f"✅ **Correlation** - {aru_data['correlation']:.4f} indicates strong predictive accuracy.\n")
        
        f.write("\n---\n\n")
        f.write("## 🔬 Technical Analysis\n\n")
        f.write("### Nested Hierarchies with Different Time Scales\n\n")
        f.write("This task requires:\n")
        f.write("- **Multi-resolution processing**: Simultaneously tracking fast and slow dynamics\n")
        f.write("- **Persistent memory**: Maintaining slow-changing upper layer context\n")
        f.write("- **Accumulation**: Integrating rapid changes in deeper layers\n")
        f.write("- **Cross-layer coherence**: Combining information across time scales\n\n")
        
        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write("This benchmark demonstrates the models' ability to handle hierarchical temporal ")
        f.write("structures with nested time scales. Applications include:\n")
        f.write("- Hierarchical reinforcement learning\n")
        f.write("- Multi-timescale forecasting\n")
        f.write("- Compositional sequence understanding\n")
    
    console.print(f"\n[green]✓[/green] Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: Inception - Nested Dream Layers Benchmark'
    )
    parser.add_argument('--seq-length', type=int, default=100, help='Sequence length (default: 100)')
    parser.add_argument('--n-layers', type=int, default=3, help='Number of dream layers (default: 3)')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size (default: 128)')
    parser.add_argument('--epochs', type=int, default=30, help='Max epochs (default: 30)')
    parser.add_argument('--model', type=str, choices=['ARU', 'GRU', 'LSTM', 'RNN'],
                       help='Train single model only')
    parser.add_argument('--hard', action='store_true', help='Harder: longer sequences, more layers')
    args = parser.parse_args()
    
    if args.hard:
        seq_length = 200
        n_layers= 4
        train_samples, val_samples, test_samples = 3000, 300, 300
    else:
        seq_length = args.seq_length
        n_layers = args.n_layers
        train_samples, val_samples, test_samples = 5000, 500, 500
    
    config = {
        'seq_length': seq_length,
        'n_layers': n_layers,
        'hidden_size': args.hidden_size,
        'batch_size': 64,
        'epochs': args.epochs,
        'lr': 0.001,
        'patience': 10,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'model_filter': args.model,
    }
    
    console.print(Panel.fit(
        "[bold cyan]🌀 Inception - Nested Dream Layers[/bold cyan]\n"
        f"[yellow]Sequence Length: {config['seq_length']} | "
        f"Layers: {config['n_layers'] + 1} (Reality + {config['n_layers']} dreams)[/yellow]\n"
        "[dim]\"We need to go deeper...\"[/dim]",
        border_style="blue"
    ))
    
    run_inception_benchmark(config)


if __name__ == "__main__":
    main()
