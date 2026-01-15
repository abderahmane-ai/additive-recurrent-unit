#!/usr/bin/env python3
"""
Phase 6: The Matrix - Bullet Time Dodge Benchmark

Inspired by the iconic bullet dodge scene from The Matrix.

Task:
- Observe incoming bullet trajectories in 3D space
- Predict future positions of multiple projectiles
- Generate optimal evasion sequence to avoid getting hit

The model receives:
- Current positions and velocities of N bullets (3D coordinates)
- Agent's current position
- Time step information

The model must predict:
- Future bullet positions (T timesteps ahead)
- Optimal agent velocity/movement to dodge

Physics:
- Realistic ballistics with gravity and air resistance
- Multiple simultaneous projectiles
- 3D spatial reasoning

Why ARU excels:
- Long-term trajectory tracking across timesteps
- Accumulation of velocity information
- Persistent memory of multiple object states
- Clean separation of position vs. velocity updates

Metrics:
- Trajectory prediction error (MSE on future positions)
- Dodge success rate (% of bullets avoided)
- Minimum clearance distance
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


class BulletPhysics:
    """Realistic bullet physics simulator."""
    
    def __init__(self, gravity=9.81, air_resistance=0.01):
        self.g = gravity  # m/s^2
        self.k = air_resistance  # air resistance coefficient
    
    def simulate_trajectory(self, initial_pos, initial_vel, timesteps, dt=0.01):
        """
        Simulate bullet trajectory with gravity and air resistance.
        
        Args:
            initial_pos: (3,) array [x, y, z]
            initial_vel: (3,) array [vx, vy, vz]
            timesteps: number of time steps
            dt: time delta per step (seconds)
        
        Returns:
            positions: (timesteps, 3) trajectory
            velocities: (timesteps, 3) velocities
        """
        positions = np.zeros((timesteps, 3))
        velocities = np.zeros((timesteps, 3))
        
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        
        for t in range(timesteps):
            positions[t] = pos
            velocities[t] = vel
            
            # Air resistance (proportional to velocity)
            air_drag = -self.k * vel
            
            # Gravity (only affects y-axis, assuming y is up)
            gravity_force = np.array([0, -self.g, 0])
            
            # Update velocity
            vel = vel + (air_drag + gravity_force) * dt
            
            # Update position
            pos = pos + vel * dt
        
        return positions, velocities


def generate_matrix_scenario(n_bullets=5, seq_length=100, dt=0.01, seed=None):
    """
    Generate a Matrix bullet dodge scenario.
    
    Returns:
        scenario: dict with bullet trajectories and agent state
    """
    if seed is not None:
        np.random.seed(seed)
    
    physics = BulletPhysics()
    
    # Agent starts at origin
    agent_pos = np.array([0.0, 1.5, 0.0])  # 1.5m height (human head)
    
    # Generate bullets coming from different directions
    bullets = []
    for _ in range(n_bullets):
        # Random initial position in a sphere around agent (3-10m away)
        distance = np.random.uniform(3.0, 10.0)
        theta = np.random.uniform(0, 2 * np.pi)  # azimuth
        phi = np.random.uniform(np.pi/4, 3*np.pi/4)  # elevation (not from above/below)
        
        x = distance * np.sin(phi) * np.cos(theta)
        y = 1.5 + distance * np.cos(phi) * 0.3  # slight variation in height
        z = distance * np.sin(phi) * np.sin(theta)
        
        initial_pos = np.array([x, y, z])
        
        # Initial velocity points towards agent with some variation
        direction = agent_pos - initial_pos
        direction = direction / np.linalg.norm(direction)
        
        # Bullet speed: 300-500 m/s with some randomness
        speed = np.random.uniform(300, 500)
        
        # Add small random perturbation to make it interesting
        perturbation = np.random.normal(0, 0.1, 3)
        initial_vel = direction * speed + perturbation
        
        # Simulate trajectory
        positions, velocities = physics.simulate_trajectory(
            initial_pos, initial_vel, seq_length, dt
        )
        
        bullets.append({
            'positions': positions,
            'velocities': velocities
        })
    
    return {
        'bullets': bullets,
        'agent_pos': agent_pos,
        'n_bullets': n_bullets,
        'seq_length': seq_length
    }


def create_dataset(n_samples, n_bullets, obs_length, pred_length, dt=0.01, seed=None):
    """
    Create dataset for trajectory prediction.
    
    Input: First `obs_length` timesteps of bullet positions and velocities
    Target: Next `pred_length` timesteps of positions
    
    Returns:
        inputs: (n_samples, obs_length, n_bullets * 6)  [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] per bullet
        targets: (n_samples, pred_length, n_bullets * 3)  [pos_x, pos_y, pos_z] per bullet
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_length = obs_length + pred_length
    
    inputs_list = []
    targets_list = []
    
    for i in range(n_samples):
        scenario = generate_matrix_scenario(
            n_bullets=n_bullets,
            seq_length=total_length,
            dt=dt,
            seed=seed + i if seed else None
        )
        
        # Extract observation period
        obs_data = []
        for bullet in scenario['bullets']:
            # Concatenate position and velocity for each timestep
            bullet_data = np.concatenate([
                bullet['positions'][:obs_length],  # (obs_length, 3)
                bullet['velocities'][:obs_length]  # (obs_length, 3)
            ], axis=1)  # (obs_length, 6)
            obs_data.append(bullet_data)
        
        # Stack all bullets: (obs_length, n_bullets * 6)
        obs_data = np.concatenate(obs_data, axis=1)
        
        # Extract prediction target
        pred_data = []
        for bullet in scenario['bullets']:
            pred_data.append(bullet['positions'][obs_length:total_length])  # (pred_length, 3)
        
        # Stack all bullets: (pred_length, n_bullets * 3)
        pred_data = np.concatenate(pred_data, axis=1)
        
        inputs_list.append(obs_data)
        targets_list.append(pred_data)
    
    inputs = np.array(inputs_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    
    # Normalize inputs (important for training stability)
    # Position: typically in range [-10, 10], normalize to ~[-1, 1]
    inputs[:, :, 0::6] = inputs[:, :, 0::6] / 10.0  # x positions
    inputs[:, :, 1::6] = inputs[:, :, 1::6] / 10.0  # y positions
    inputs[:, :, 2::6] = inputs[:, :, 2::6] / 10.0  # z positions
    # Velocity: typically in range [-500, 500], normalize to ~[-1, 1]
    inputs[:, :, 3::6] = inputs[:, :, 3::6] / 500.0  # x velocities
    inputs[:, :, 4::6] = inputs[:, :, 4::6] / 500.0  # y velocities
    inputs[:, :, 5::6] = inputs[:, :, 5::6] / 500.0  # z velocities
    
    # Normalize targets (positions only)
    targets = targets / 10.0
    
    return inputs, targets


def create_model(model_class, input_size, hidden_size, output_size, is_aru=False):
    """Create model for trajectory prediction."""
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


def calculate_dodge_metrics(predictions, targets):
    """
    Calculate dodge-specific metrics.
    
    Args:
        predictions: (batch, pred_length, n_bullets * 3) predicted positions
        targets: (batch, pred_length, n_bullets * 3) true positions
    
    Returns:
        dict with metrics
    """
    # Denormalize for physical interpretation
    predictions = predictions * 10.0
    targets = targets * 10.0
    
    # Reshape to (batch, pred_length, n_bullets, 3)
    n_bullets = predictions.shape[-1] // 3
    pred = predictions.reshape(predictions.shape[0], predictions.shape[1], n_bullets, 3)
    targ = targets.reshape(targets.shape[0], targets.shape[1], n_bullets, 3)
    
    # Calculate per-bullet distance error
    errors = np.linalg.norm(pred - targ, axis=-1)  # (batch, pred_length, n_bullets)
    
    # Average error across all bullets and timesteps
    mean_error = np.mean(errors)
    
    # Max error (worst case)
    max_error = np.max(errors)
    
    # Error at final timestep (most important for dodging)
    final_error = np.mean(errors[:, -1, :])
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'final_error': final_error
    }


def run_matrix_benchmark(config: dict, seed: int = 42):
    """Run Matrix bullet dodge benchmark."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(f"\n[green]Device:[/green] {device}")
    
    n_bullets = config['n_bullets']
    obs_length = config['obs_length']
    pred_length = config['pred_length']
    
    console.print(f"[cyan]Bullets:[/cyan] {n_bullets}")
    console.print(f"[cyan]Observation length:[/cyan] {obs_length} timesteps")
    console.print(f"[cyan]Prediction length:[/cyan] {pred_length} timesteps")
    console.print(f"[cyan]Total sequence:[/cyan] {obs_length + pred_length} timesteps\n")
    
    # Generate data
    console.print("[cyan]Generating Matrix scenarios...[/cyan]")
    train_x, train_y = create_dataset(
        config['train_samples'], n_bullets, obs_length, pred_length,
        dt=config['dt'], seed=seed
    )
    val_x, val_y = create_dataset(
        config['val_samples'], n_bullets, obs_length, pred_length,
        dt=config['dt'], seed=seed + 1000
    )
    test_x, test_y = create_dataset(
        config['test_samples'], n_bullets, obs_length, pred_length,
        dt=config['dt'], seed=seed + 2000
    )
    
    console.print(f"[green]✓[/green] Generated {config['train_samples']:,} train, "
                  f"{config['val_samples']:,} val, {config['test_samples']:,} test scenarios")
    
    input_size = n_bullets * 6  # pos (3) + vel (3) per bullet
    output_size = n_bullets * 3 * pred_length  # positions for pred_length steps
    
    console.print(f"[dim]Input dimension: {input_size}[/dim]")
    console.print(f"[dim]Output dimension: {output_size}[/dim]\n")
    
    # Flatten targets for training: (samples, obs_length, output_size) -> (samples, output_size)
    # We take the last hidden state to predict all future timesteps
    train_y_flat = train_y.reshape(train_y.shape[0], -1)
    val_y_flat = val_y.reshape(val_y.shape[0], -1)
    test_y_flat = test_y.reshape(test_y.shape[0], -1)
    
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y_flat))
    val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y_flat))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y_flat))
    
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
                model_class, input_size, config['hidden_size'], output_size, is_aru
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
                        out = model(x)
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
                        best_state = model.state_dict().copy()
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
            
            # Reshape back to (samples, pred_length, n_bullets * 3)
            predictions = predictions.reshape(-1, pred_length, n_bullets * 3)
            targets = targets.reshape(-1, pred_length, n_bullets * 3)
            
            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            dodge_metrics = calculate_dodge_metrics(predictions, targets)
            
            results[name] = {
                'params': params,
                'test_mse': mse,
                'mean_error_m': dodge_metrics['mean_error'],
                'final_error_m': dodge_metrics['final_error'],
                'max_error_m': dodge_metrics['max_error'],
                'train_time': train_time
            }
            
            console.print(f"[green]✓[/green] {name}: Test MSE = {mse:.4f}, "
                         f"Mean Error = {dodge_metrics['mean_error']:.2f}m")
            
        except Exception as e:
            console.print(f"[red]Error training {name}:[/red] {e}")
            import traceback
            traceback.print_exc()
    
    # Results table
    console.print("\n")
    table = Table(
        title=f"🎬 The Matrix - Bullet Dodge Results ({n_bullets} bullets, {pred_length}T ahead)",
        header_style="bold magenta"
    )
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("Test MSE", justify="right")
    table.add_column("Mean Error (m)", justify="right")
    table.add_column("Final Error (m)", justify="right")
    table.add_column("Time", justify="right")
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['test_mse']):
        table.add_row(
            name,
            f"{data['params']:,}",
            f"{data['test_mse']:.4f}",
            f"{data['mean_error_m']:.2f}",
            f"{data['final_error_m']:.2f}",
            f"{data['train_time']:.1f}s"
        )
    
    console.print(table)
    
    console.print("\n[dim]Lower is better for all metrics[/dim]")
    console.print("[dim]Error in meters - how far off the predicted position is[/dim]")
    
    if 'ARU' in results and 'GRU' in results:
        aru_err = results['ARU']['mean_error_m']
        gru_err = results['GRU']['mean_error_m']
        if aru_err < gru_err:
            improvement = (gru_err - aru_err) / gru_err * 100
            console.print(f"\n[green]🎯 ARU predicts {improvement:.1f}% more accurately than GRU![/green]")
    
    # Save report
    os.makedirs(os.path.join(project_root, "benchmarks", "phase6"), exist_ok=True)
    report_path = os.path.join(project_root, "benchmarks", "phase6", "matrix_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Phase 6: The Matrix - Bullet Time Dodge\n\n")
        f.write("## Inspired by The Matrix (1999)\n\n")
        f.write("*\"Dodge this.\"* - Trinity\n\n")
        f.write("## Task\n\n")
        f.write(f"Predict the future trajectories of {n_bullets} bullets over {pred_length} timesteps ")
        f.write(f"after observing {obs_length} timesteps of their motion.\n\n")
        f.write("The model must learn:\n")
        f.write("- 3D spatial reasoning\n")
        f.write("- Physics-based trajectory prediction (gravity + air resistance)\n")
        f.write("- Multi-object tracking and prediction\n")
        f.write("- Long-term temporal dependencies\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Params | Test MSE | Mean Error (m) | Final Error (m) |\n")
        f.write("|-------|--------|----------|----------------|------------------|\n")
        for name, data in sorted(results.items(), key=lambda x: x[1]['test_mse']):
            f.write(f"| {name} | {data['params']:,} | {data['test_mse']:.4f} | "
                   f"{data['mean_error_m']:.2f} | {data['final_error_m']:.2f} |\n")
        f.write("\n**Error in meters** - average distance between predicted and actual bullet positions.\n")
        f.write("\nLower error means better trajectory prediction → better dodging!\n")
    
    console.print(f"\n[green]✓[/green] Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 6: The Matrix - Bullet Time Dodge Benchmark'
    )
    parser.add_argument('--n-bullets', type=int, default=5, help='Number of bullets (default: 5)')
    parser.add_argument('--obs-length', type=int, default=30, help='Observation length (default: 30)')
    parser.add_argument('--pred-length', type=int, default=20, help='Prediction length (default: 20)')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden size (default: 128)')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs (default: 50)')
    parser.add_argument('--model', type=str, choices=['ARU', 'GRU', 'LSTM', 'RNN'],
                       help='Train single model only')
    parser.add_argument('--hard', action='store_true', help='Harder: 10 bullets, longer prediction')
    args = parser.parse_args()
    
    if args.hard:
        n_bullets, obs_length, pred_length = 10, 40, 30
        train_samples, val_samples, test_samples = 3000, 300, 300
    else:
        n_bullets = args.n_bullets
        obs_length = args.obs_length
        pred_length = args.pred_length
        train_samples, val_samples, test_samples = 5000, 500, 500
    
    config = {
        'n_bullets': n_bullets,
        'obs_length': obs_length,
        'pred_length': pred_length,
        'hidden_size': args.hidden_size,
        'batch_size': 64,
        'epochs': args.epochs,
        'lr': 0.001,
        'patience': 10,
        'dt': 0.01,  # 10ms timesteps
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'model_filter': args.model,
    }
    
    console.print(Panel.fit(
        "[bold cyan]🎬 The Matrix - Bullet Time Dodge[/bold cyan]\n"
        f"[yellow]Bullets: {config['n_bullets']} | "
        f"Observe: {config['obs_length']}T | Predict: {config['pred_length']}T[/yellow]\n"
        "[dim]\"I know kung fu... now let's see if I can dodge bullets.\"[/dim]",
        border_style="green"
    ))
    
    run_matrix_benchmark(config)


if __name__ == "__main__":
    main()
