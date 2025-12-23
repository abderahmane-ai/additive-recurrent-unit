"""
Training utilities for SPMN experiments.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = None
) -> Tuple[float, float]:
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        
        if max_grad_norm is not None:
             nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item() * batch_y.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)
    
    return total_loss / total, 100. * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item() * batch_y.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    
    return total_loss / total, 100. * correct / total


def test_and_print_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
    class_names: list = None
) -> None:
    """
    Test model on new data and print sample predictions.
    
    Args:
        model: The trained model
        loader: DataLoader with test data
        device: Device to run on
        num_samples: Number of sample predictions to print
        class_names: Optional list of class names for better readability
    """
    model.eval()
    
    samples_shown = 0
    correct_count = 0
    total_count = 0
    
    print("\n" + "="*60)
    print("Testing on New Data - Sample Predictions:")
    print("="*60)
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            _, predicted = output.max(1)
            
            # Show samples
            for i in range(min(len(batch_x), num_samples - samples_shown)):
                true_label = batch_y[i].item()
                pred_label = predicted[i].item()
                confidence = torch.softmax(output[i], dim=0).max().item()
                
                if class_names:
                    true_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
                    pred_name = class_names[pred_label] if pred_label < len(class_names) else f"Class {pred_label}"
                else:
                    true_name = f"Class {true_label}"
                    pred_name = f"Class {pred_label}"
                
                status = "✓" if true_label == pred_label else "✗"
                print(f"  {status} Sample {samples_shown + 1}: True={true_name}, Pred={pred_name}, Confidence={confidence:.2%}")
                
                samples_shown += 1
                if samples_shown >= num_samples:
                    break
            
            # Count accuracy
            correct_count += predicted.eq(batch_y).sum().item()
            total_count += batch_y.size(0)
            
            if samples_shown >= num_samples:
                break
    
    # Print overall test accuracy
    test_accuracy = 100. * correct_count / total_count if total_count > 0 else 0
    print(f"\nOverall Test Accuracy: {test_accuracy:.2f}% ({correct_count}/{total_count})")
    print("="*60 + "\n")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: torch.device = None,
    verbose: bool = True
) -> Dict:
    """
    Train a model and return results.
    
    Returns:
        Dict with: best_acc, train_accs, test_accs, train_time, params
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0
    train_accs, test_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        best_acc = max(best_acc, test_acc)
        
        if verbose:
            print(f"Epoch {epoch}: Train {train_acc:.2f}%, Test {test_acc:.2f}%")
    
    train_time = time.time() - start_time
    
    return {
        'best_acc': best_acc,
        'final_train_acc': train_accs[-1],
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_time': train_time,
        'params': count_parameters(model)
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
