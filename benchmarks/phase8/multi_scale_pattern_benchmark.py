"""
Multi-Scale Pattern Recognition Benchmark

Task: Detect and count patterns at three different timescales simultaneously.

Structure:
    - Fast patterns:   Every 5 steps (e.g., "10101")
    - Medium patterns: Every 20 steps (e.g., "1...1" with 20-step gap)
    - Slow patterns:   Every 80 steps (e.g., "1...1" with 80-step gap)

The model must:
    1. Track all three pattern types concurrently
    2. Output the total count of completed patterns at each timestep
    3. Handle interference between scales

This task is designed to favor hierarchical architectures that can naturally
separate timescales, while single-scale RNNs must balance competing temporal
dynamics in a single hidden state.

Example sequence (simplified):
    t=0-4:   Fast pattern starts
    t=5:     Fast pattern completes → count=1
    t=10:    Another fast pattern → count=2
    t=20:    Medium pattern completes → count=3
    t=80:    Slow pattern completes → count=4

Author: Abderahmane Ainouche
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aru.model import ARU
from aru.hierarchical import HARU
from aru.baselines import ManualGRU as GRU, ManualLSTM as LSTM
from utils.training import train_epoch, evaluate
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


class MultiScalePatternDataset(Dataset):
    """
    Generate sequences with patterns at multiple timescales.
    
    Patterns:
        Fast:   Marker every 5 steps
        Medium: Marker every 20 steps  
        Slow:   Marker every 80 steps
    
    Target: Cumulative count of completed patterns at each timestep.
    """
    
    def __init__(
        self,
        num_sequences: int = 1000,
        seq_length: int = 200,
        fast_period: int = 5,
        medium_period: int = 20,
        slow_period: int = 80,
        noise_prob: float = 0.05,
    ):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.fast_period = fast_period
        self.medium_period = medium_period
        self.slow_period = slow_period
        self.noise_prob = noise_prob
        
        self.sequences = []
        self.targets = []
        
        for _ in range(num_sequences):
            seq, target = self._generate_sequence()
            self.sequences.append(seq)
            self.targets.append(target)
    
    def _generate_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single sequence with multi-scale patterns."""
        # Initialize sequence (0=background, 1=fast, 2=medium, 3=slow, 4=noise)
        seq = torch.zeros(self.seq_length, dtype=torch.long)
        target = torch.zeros(self.seq_length, dtype=torch.float32)
        
        count = 0
        
        # Place pattern markers
        for t in range(self.seq_length):
            # Check for pattern completions
            if t > 0 and t % self.fast_period == 0:
                seq[t] = 1
                count += 1
            
            if t > 0 and t % self.medium_period == 0:
                seq[t] = 2
                count += 1
            
            if t > 0 and t % self.slow_period == 0:
                seq[t] = 3
                count += 1
            
            # Add noise
            if np.random.random() < self.noise_prob and seq[t] == 0:
                seq[t] = 4
            
            target[t] = count
        
        return seq, target
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    device: str,
) -> nn.Module:
    """Create model based on type."""
    if model_type == "ARU":
        return ARU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=None,
            use_embedding=True,
            dropout=0.1,
        ).to(device)
    
    elif model_type == "HARU":
        # Three-layer hierarchy optimized for 5/20/80 periods
        return HARU(
            input_size=input_size,
            hidden_sizes=[64, 128, 256],  # Fast → Medium → Slow
            persistence_inits=[0.5, 2.0, 3.5],  # τ≈2, τ≈7, τ≈30
            accumulation_inits=[1.0, 0.0, -1.0],  # Reactive → Conservative
            num_classes=None,
            use_embedding=True,
            dropout=0.1,
            use_skip_connections=True,
        ).to(device)
    
    elif model_type == "GRU":
        return GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=None,
            use_embedding=True,
            dropout=0.1,
        ).to(device)
    
    elif model_type == "LSTM":
        return LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=None,
            use_embedding=True,
            dropout=0.1,
        ).to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 50,
    lr: float = 0.001,
) -> Dict[str, float]:
    """Train model and return best validation metrics."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Determine output size based on model type
    if hasattr(model, 'hidden_sizes'):
        # HARU with skip connections
        if hasattr(model, 'use_skip_connections') and model.use_skip_connections:
            output_size = sum(model.hidden_sizes)
        else:
            output_size = model.hidden_sizes[-1]
    else:
        # ARU, GRU, LSTM
        output_size = model.hidden_size
    
    # Add output projection for regression
    output_proj = nn.Linear(output_size, 1).to(device)
    
    best_val_loss = float('inf')
    best_metrics = {}
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        output_proj.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            hidden = model(sequences, return_all_states=True)
            predictions = output_proj(hidden).squeeze(-1)
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(output_proj.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        output_proj.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                hidden = model(sequences, return_all_states=True)
                predictions = output_proj(hidden).squeeze(-1)
                
                loss = criterion(predictions, targets)
                mae = torch.abs(predictions - targets).mean()
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'train_loss': train_loss,
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_metrics


def run_benchmark():
    """Run multi-scale pattern recognition benchmark."""
    console.print("\n[bold cyan]Multi-Scale Pattern Recognition Benchmark[/bold cyan]")
    console.print("=" * 70)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: {device}\n")
    
    # Dataset
    console.print("[yellow]Generating datasets...[/yellow]")
    train_dataset = MultiScalePatternDataset(num_sequences=2000, seq_length=200)
    val_dataset = MultiScalePatternDataset(num_sequences=500, seq_length=200)
    test_dataset = MultiScalePatternDataset(num_sequences=500, seq_length=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    console.print(f"Train: {len(train_dataset)} sequences")
    console.print(f"Val:   {len(val_dataset)} sequences")
    console.print(f"Test:  {len(test_dataset)} sequences\n")
    
    # Models to compare
    models_config = {
        "HARU": {"hidden_size": 256},  # Total params comparable
        "ARU": {"hidden_size": 448},   # Match HARU's total hidden capacity
        "GRU": {"hidden_size": 448},
        "LSTM": {"hidden_size": 448},
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        console.print(f"[bold green]Training {model_name}...[/bold green]")
        
        model = create_model(
            model_type=model_name,
            input_size=5,  # 0=background, 1=fast, 2=medium, 3=slow, 4=noise
            hidden_size=config["hidden_size"],
            device=device,
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        console.print(f"Parameters: {num_params:,}")
        
        # Train
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=50,
            lr=0.001,
        )
        
        results[model_name] = {
            'params': num_params,
            **metrics
        }
        
        console.print(f"Best Val MAE: {metrics['val_mae']:.4f}\n")
    
    # Display results
    console.print("\n[bold cyan]Final Results[/bold cyan]")
    console.print("=" * 70)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Val Loss (MSE)", justify="right")
    table.add_column("Val MAE", justify="right")
    table.add_column("Train Loss", justify="right")
    
    # Sort by validation MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['val_mae'])
    
    for model_name, metrics in sorted_results:
        table.add_row(
            model_name,
            f"{metrics['params']:,}",
            f"{metrics['val_loss']:.4f}",
            f"{metrics['val_mae']:.4f}",
            f"{metrics['train_loss']:.4f}",
        )
    
    console.print(table)
    
    # Analysis
    console.print("\n[bold cyan]Analysis[/bold cyan]")
    console.print("=" * 70)
    
    haru_mae = results['HARU']['val_mae']
    aru_mae = results['ARU']['val_mae']
    improvement = ((aru_mae - haru_mae) / aru_mae) * 100
    
    console.print(f"HARU vs ARU: {improvement:+.1f}% improvement")
    console.print(f"HARU MAE: {haru_mae:.4f}")
    console.print(f"ARU MAE:  {aru_mae:.4f}")
    
    if improvement > 10:
        console.print("\n[bold green]✓ HARU shows significant advantage on multi-scale task![/bold green]")
    elif improvement > 0:
        console.print("\n[yellow]HARU shows modest improvement.[/yellow]")
    else:
        console.print("\n[red]HARU did not outperform ARU on this task.[/red]")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_benchmark()
