"""
Diagnostic script to test continual learning with detailed output analysis
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.vision_transformer_nested_learning import ViTNestedLearning, ViTNestedConfig
from data.datasets import SplitCIFAR10
from torch.utils.data import DataLoader

def test_model_outputs():
    """Test what the model outputs for different tasks"""
    print("="*70)
    print("Diagnostic Test: Model Output Analysis")
    print("="*70)
    
    # Create tiny model for faster testing
    config = ViTNestedConfig(
        img_size=32, patch_size=4, num_classes=10,
        dim=192, depth=3, num_heads=3, mlp_ratio=4.0,
    )
    model = ViTNestedLearning(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load dataset
    dataset = SplitCIFAR10(root='./data', num_tasks=2)
    
    # Get Task 0 (classes 0-1) and Task 1 (classes 2-3)
    task0_train = dataset.get_task(0, train=True)
    task0_test = dataset.get_task(0, train=False)
    task1_train = dataset.get_task(1, train=True)
    task1_test = dataset.get_task(1, train=False)
    
    task0_train_loader = DataLoader(task0_train, batch_size=128, shuffle=True)
    task0_test_loader = DataLoader(task0_test, batch_size=128, shuffle=False)
    task1_train_loader = DataLoader(task1_train, batch_size=128, shuffle=True)
    task1_test_loader = DataLoader(task1_test, batch_size=128, shuffle=False)
    
    # Simple training function
    def train_one_epoch(model, loader, optimizer):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    # Evaluation function with detailed output
    def evaluate_detailed(model, loader, task_name):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                
                correct += (preds == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        accuracy = 100.0 * correct / total
        
        # Analyze predictions
        unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
        unique_labels, label_counts = np.unique(all_labels, return_counts=True)
        
        print(f"\n{task_name}:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  True labels: {unique_labels} (counts: {label_counts})")
        print(f"  Predictions: {unique_preds} (counts: {pred_counts})")
        print(f"  Mean logits per class: {all_logits.mean(dim=0)[:10].numpy()}")
        print(f"  Max logits: {all_logits.max(dim=1)[0].mean():.3f}, Min: {all_logits.min(dim=1)[0].mean():.3f}")
        
        return accuracy
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Initial state
    print("\n" + "="*70)
    print("INITIAL STATE (Random weights)")
    print("="*70)
    evaluate_detailed(model, task0_test_loader, "Task 0 (classes 0-1)")
    evaluate_detailed(model, task1_test_loader, "Task 1 (classes 2-3)")
    
    # Train on Task 0
    print("\n" + "="*70)
    print("TRAINING ON TASK 0")
    print("="*70)
    for epoch in range(3):
        loss = train_one_epoch(model, task0_train_loader, optimizer)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    print("\n" + "="*70)
    print("AFTER TRAINING ON TASK 0")
    print("="*70)
    acc0 = evaluate_detailed(model, task0_test_loader, "Task 0 (classes 0-1)")
    acc1 = evaluate_detailed(model, task1_test_loader, "Task 1 (classes 2-3)")
    
    # Train on Task 1
    print("\n" + "="*70)
    print("TRAINING ON TASK 1")
    print("="*70)
    for epoch in range(3):
        loss = train_one_epoch(model, task1_train_loader, optimizer)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}")
    
    print("\n" + "="*70)
    print("AFTER TRAINING ON TASK 1")
    print("="*70)
    acc0_after = evaluate_detailed(model, task0_test_loader, "Task 0 (classes 0-1)")
    acc1_after = evaluate_detailed(model, task1_test_loader, "Task 1 (classes 2-3)")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Task 0 accuracy: {acc0:.2f}% -> {acc0_after:.2f}% (change: {acc0_after - acc0:+.2f}%)")
    print(f"Task 1 accuracy: {acc1:.2f}% -> {acc1_after:.2f}% (change: {acc1_after - acc1:+.2f}%)")
    print(f"Forgetting: {acc0 - acc0_after:.2f}%")
    print("="*70)
    
    # Check model memory stats
    print("\nMemory Stats (Block 0):")
    stats = model.get_memory_stats()
    for key, val in stats['block_0'].items():
        print(f"  {key}: {val}")

if __name__ == '__main__':
    test_model_outputs()
