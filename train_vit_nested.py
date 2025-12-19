"""
Training script for Vision Transformer with Nested Learning
Demonstrates how to train the model on image classification tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vision_transformer_nested_learning import (
    create_vit_nested_tiny,
    create_vit_nested_small,
    create_vit_nested_base,
    ViTNestedConfig
)
import argparse
from tqdm import tqdm
import json
from pathlib import Path


def get_dataloaders(data_path: str, batch_size: int = 32, img_size: int = 224):
    """Create training and validation dataloaders"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device, teach_scale=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def test_memorization(model, dataloader, device, num_steps=5):
    """Test memorization capability at inference time"""
    model.enable_memorization(True)
    
    correct_baseline = 0
    correct_memorized = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Memorization test"):
        images, labels = images.to(device), labels.to(device)
        
        # Baseline prediction
        logits_baseline = model(images)
        _, pred_baseline = logits_baseline.max(1)
        
        # Memorization steps
        for _ in range(num_steps):
            logits = model(images)
            # Simulate teaching signal from correct answer
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
        
        # Final prediction after memorization
        logits_memorized = model(images)
        _, pred_memorized = logits_memorized.max(1)
        
        correct_baseline += pred_baseline.eq(labels).sum().item()
        correct_memorized += pred_memorized.eq(labels).sum().item()
        total += labels.size(0)
    
    model.enable_memorization(False)
    
    return 100. * correct_baseline / total, 100. * correct_memorized / total


def main():
    parser = argparse.ArgumentParser(description='Train ViT with Nested Learning')
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'])
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--inner-lr', type=float, default=0.01)
    parser.add_argument('--teach-scale', type=float, default=0.1)
    parser.add_argument('--surprise-threshold', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--test-memorization', action='store_true')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print(f"Creating {args.model} ViT-Nested model...")
    if args.model == 'tiny':
        model = create_vit_nested_tiny(num_classes=10)
    elif args.model == 'small':
        model = create_vit_nested_small(num_classes=10)
    elif args.model == 'base':
        model = create_vit_nested_base(num_classes=10)
    else:
        model = create_vit_nested_large(num_classes=10)
    
    # Override config parameters
    for block in model.blocks:
        block.teach_scale = args.teach_scale
        block.surprise_threshold = args.surprise_threshold
        block.titan_optimizer.lr = args.inner_lr
        block.cms_fast_optimizer.lr = args.inner_lr * 0.5
        block.cms_slow_optimizer.lr = args.inner_lr * 0.25
    
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(args.data_path, args.batch_size)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    history = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, args.teach_scale
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step()
        
        # Memory statistics
        memory_stats = model.get_memory_stats()
        
        # Log results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Memory stats (block 0): {memory_stats['block_0']}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'memory_stats': memory_stats['block_0']
        })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': model.config,
            }, save_dir / f'best_model_{args.model}.pt')
            print(f"✓ Saved new best model with val_acc={val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': model.config,
            }, save_dir / f'checkpoint_epoch_{epoch+1}_{args.model}.pt')
    
    # Save training history
    with open(save_dir / f'history_{args.model}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Test memorization if requested
    if args.test_memorization:
        print("\nTesting memorization capability...")
        baseline_acc, memorized_acc = test_memorization(model, val_loader, args.device)
        print(f"Baseline accuracy: {baseline_acc:.2f}%")
        print(f"Memorized accuracy: {memorized_acc:.2f}%")
        print(f"Improvement: {memorized_acc - baseline_acc:.2f}%")


if __name__ == '__main__':
    main()
