#!/usr/bin/env python3
"""
Phase 7: Real-World Time Series Forecasting - ETT Benchmark

Tests ARU on the Electricity Transformer Temperature (ETT) dataset,
a standard benchmark for long-term time series forecasting with real-world
industrial sensor data.

Task:
- Input: Past 96 timesteps of 7 features (6 power loads + 1 temperature)
- Output: Predict oil temperature H steps ahead (H = 96, 192, 336, or 720)
- Challenge: Multi-scale patterns, non-stationarity, long-horizon forecasting

Why ARU excels:
- Persistence gate (π) maintains stable long-term trends
- Accumulation gate (α) captures additive seasonal components
- Reset gate (ρ) handles regime changes in real-world data
- Combined architecture models both baseline and dynamic fluctuations

Dataset: ETT (Electricity Transformer Temperature)
- Source: https://github.com/zhouhaoyi/ETDataset
- Standard split: 12 months train / 4 months val / 4 months test
- Normalized using training statistics
"""

import argparse
import time
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from aru import ARU
from aru.baselines import ManualGRU, ManualLSTM
from utils.training import count_parameters

console = Console()


class ETTDataset(Dataset):
    """ETT Dataset for time series forecasting."""
    
    def __init__(self, data, seq_len=96, pred_len=96, features='MS'):
        """
        Args:
            data: DataFrame with columns [date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]
            seq_len: Input sequence length (lookback window)
            pred_len: Prediction horizon (forecast length)
            features: 'MS' (multivariate predict single), 'M' (multivariate predict multivariate)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        
        # Remove date column and convert to numpy
        if 'date' in data.columns:
            data = data.drop(columns=['date'])
        
        self.data = data.values.astype(np.float32)
        
        # Target is always the last column (OT - Oil Temperature)
        self.target_idx = -1
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        """
        Returns:
            x: (seq_len, n_features) - input sequence
            y: (pred_len,) or (pred_len, n_features) - target sequence
        """
        # Input: past seq_len timesteps, all features
        x = self.data[idx:idx + self.seq_len]
        
        # Target: future pred_len timesteps
        if self.features == 'MS':
            # Predict only target variable (OT)
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_idx]
        else:
            # Predict all variables
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        
        return torch.from_numpy(x), torch.from_numpy(y)


def load_ett_data(data_path, dataset_name='ETTh1', seq_len=96, pred_len=96):
    """
    Load and split ETT dataset.
    
    Standard split: 12 months train / 4 months val / 4 months test
    """
    import pandas as pd
    
    # Load data
    file_path = os.path.join(data_path, f'{dataset_name}.csv')
    
    if not os.path.exists(file_path):
        console.print(f"[red]Error: Dataset not found at {file_path}[/red]")
        console.print(f"[yellow]Please download from: https://github.com/zhouhaoyi/ETDataset[/yellow]")
        sys.exit(1)
    
    df = pd.read_csv(file_path)
    
    # Standard split for ETT
    # ETTh1/h2: 17520 points (2 years hourly)
    # Train: 12 months, Val: 4 months, Test: 4 months
    n_train = 12 * 30 * 24  # 8640
    n_val = 4 * 30 * 24     # 2880
    
    train_df = df[:n_train]
    val_df = df[n_train - seq_len:n_train + n_val]
    test_df = df[n_train + n_val - seq_len:]
    
    # Normalize using training statistics
    train_mean = train_df.drop(columns=['date']).mean()
    train_std = train_df.drop(columns=['date']).std()
    
    def normalize(data):
        data = data.copy()
        for col in data.columns:
            if col != 'date':
                data[col] = (data[col] - train_mean[col]) / train_std[col]
        return data
    
    train_df = normalize(train_df)
    val_df = normalize(val_df)
    test_df = normalize(test_df)
    
    # Create datasets
    train_dataset = ETTDataset(train_df, seq_len, pred_len)
    val_dataset = ETTDataset(val_df, seq_len, pred_len)
    test_dataset = ETTDataset(test_df, seq_len, pred_len)
    
    return train_dataset, val_dataset, test_dataset, (train_mean, train_std)


class TimeSeriesModel(nn.Module):
    """Wrapper for RNN models to output predictions."""
    
    def __init__(self, rnn_model, pred_len):
        super().__init__()
        self.rnn = rnn_model
        self.pred_len = pred_len
        
        # Get hidden size from RNN
        if hasattr(rnn_model, 'hidden_size'):
            hidden_size = rnn_model.hidden_size
        else:
            hidden_size = rnn_model.rnn.hidden_size
        
        # Projection layer to generate multi-step predictions
        self.projection = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            out: (batch, pred_len)
        """
        # Get final hidden state from RNN
        out = self.rnn(x)  # (batch, hidden_size)
        
        # Project to prediction horizon
        out = self.projection(out)  # (batch, pred_len)
        
        return out


def create_model(model_name, input_size, hidden_size, pred_len):
    """Create time series forecasting model."""
    
    if model_name == 'ARU':
        rnn = ARU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=hidden_size,  # Output hidden state
            dropout=0.1,
            use_embedding=False
        )
    elif model_name == 'GRU':
        rnn = ManualGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=hidden_size,
            dropout=0.1,
            use_embedding=False
        )
    elif model_name == 'LSTM':
        rnn = ManualLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=hidden_size,
            dropout=0.1,
            use_embedding=False
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = TimeSeriesModel(rnn, pred_len)
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    
    return mse, mae, preds, targets


def run_ett_benchmark(config):
    """Run ETT benchmark."""
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    console.print(f"[cyan]Dataset:[/cyan] {config['dataset']}")
    console.print(f"[cyan]Sequence length:[/cyan] {config['seq_len']}")
    console.print(f"[cyan]Prediction horizon:[/cyan] {config['pred_len']}\n")
    
    # Load data
    console.print("[cyan]Loading ETT dataset...[/cyan]")
    train_dataset, val_dataset, test_dataset, stats = load_ett_data(
        config['data_path'],
        config['dataset'],
        config['seq_len'],
        config['pred_len']
    )
    
    console.print(f"[green]✓[/green] Train: {len(train_dataset):,} samples")
    console.print(f"[green]✓[/green] Val: {len(val_dataset):,} samples")
    console.print(f"[green]✓[/green] Test: {len(test_dataset):,} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Get input size (number of features)
    input_size = train_dataset.data.shape[1]
    console.print(f"[dim]Input features: {input_size}[/dim]\n")
    
    # Models to compare
    models_to_run = config.get('models', ['ARU', 'GRU', 'LSTM'])
    
    results = {}
    
    for model_name in models_to_run:
        console.print(f"\n[bold cyan]Training {model_name}[/bold cyan]")
        
        # Create model
        model = create_model(
            model_name,
            input_size,
            config['hidden_size'],
            config['pred_len']
        ).to(device)
        
        params = count_parameters(model)
        console.print(f"Parameters: {params:,}")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        best_val_loss = float('inf')
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
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                
                # Validate
                val_mse, val_mae, _, _ = evaluate(model, val_loader, device)
                
                if val_mse < best_val_loss:
                    best_val_loss = val_mse
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]{model_name} - Val MSE: {val_mse:.4f} (Best: {best_val_loss:.4f})"
                )
                
                if patience_counter >= config['patience']:
                    console.print(f"[yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                    break
        
        train_time = time.time() - train_start
        
        # Load best model and test
        model.load_state_dict(best_state)
        
        # Test evaluation
        test_start = time.time()
        test_mse, test_mae, preds, targets = evaluate(model, test_loader, device)
        inference_time = time.time() - test_start
        
        results[model_name] = {
            'params': params,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'train_time': train_time,
            'inference_time': inference_time,
            'predictions': preds,
            'targets': targets
        }
        
        console.print(f"[green]✓[/green] {model_name}: Test MSE = {test_mse:.4f}, MAE = {test_mae:.4f}")
    
    # Results table
    console.print("\n")
    table = Table(
        title=f"📊 ETT {config['dataset']} - Horizon {config['pred_len']}",
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MSE", justify="right")
    table.add_column("Test MAE", justify="right")
    table.add_column("Train Time", justify="right")
    table.add_column("Inference", justify="right")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['test_mse']):
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['test_mse']:.4f}",
            f"{data['test_mae']:.4f}",
            f"{data['train_time']:.1f}s",
            f"{data['inference_time']:.2f}s"
        )
    
    console.print(table)
    console.print("\n[dim]Lower MSE/MAE is better[/dim]")
    
    # Compare ARU vs GRU
    if 'ARU' in results and 'GRU' in results:
        aru_mse = results['ARU']['test_mse']
        gru_mse = results['GRU']['test_mse']
        
        if aru_mse < gru_mse:
            improvement = (gru_mse - aru_mse) / gru_mse * 100
            console.print(f"\n[green]🎯 ARU outperforms GRU by {improvement:.1f}%![/green]")
        else:
            diff = (aru_mse - gru_mse) / gru_mse * 100
            console.print(f"\n[yellow]⚠️  GRU outperforms ARU by {diff:.1f}%[/yellow]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Phase 7: ETT Time Series Benchmark')
    parser.add_argument('--data-path', type=str, default='./data/ETT',
                       help='Path to ETT dataset directory')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                       choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
                       help='ETT dataset variant')
    parser.add_argument('--seq-len', type=int, default=96,
                       help='Input sequence length')
    parser.add_argument('--pred-len', type=int, default=96,
                       choices=[96, 192, 336, 720],
                       help='Prediction horizon')
    parser.add_argument('--hidden-size', type=int, default=256,
                       help='Hidden size for RNN models')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['ARU', 'GRU', 'LSTM'],
                       help='Models to benchmark')
    
    args = parser.parse_args()
    
    config = vars(args)
    
    console.print(Panel.fit(
        f"[bold cyan]📊 Phase 7: Real-World Time Series Forecasting[/bold cyan]\n"
        f"[yellow]Dataset: {config['dataset']} | "
        f"Horizon: {config['pred_len']} | "
        f"Seq Length: {config['seq_len']}[/yellow]\n"
        "[dim]Testing ARU on real-world electricity transformer data[/dim]",
        border_style="green"
    ))
    
    run_ett_benchmark(config)


if __name__ == "__main__":
    main()