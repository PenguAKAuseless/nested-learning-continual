"""
Task-as-Class Dataset Loaders for Continual Learning

Each task is a binary classification: one class vs all others.
Task 1: class 0 vs others
Task 2: class 1 vs others
...
Task N: class N-1 vs others
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Optional


class TaskAsClassDataset(Dataset):
    """
    Converts a multi-class dataset into a task-as-class binary dataset.
    
    Args:
        base_dataset: The underlying dataset (e.g., CIFAR10)
        target_class: The class to distinguish (label=1), all others are label=0
        balance: Whether to balance positive and negative samples
    """
    def __init__(self, base_dataset, target_class: int, balance: bool = True):
        self.base_dataset = base_dataset
        self.target_class = target_class
        self.balance = balance
        
        # Get all indices - optimize by checking targets directly if available
        if hasattr(base_dataset, 'targets'):
            # Fast path for datasets with targets attribute
            targets = base_dataset.targets
            if isinstance(targets, list):
                targets = np.array(targets)
            
            self.positive_indices = np.where(targets == target_class)[0].tolist()
            self.negative_indices = np.where(targets != target_class)[0].tolist()
        else:
            # Fallback to slower method
            all_indices = list(range(len(base_dataset)))
            self.positive_indices = []
            self.negative_indices = []
            
            for idx in all_indices:
                _, label = base_dataset[idx]
                if label == target_class:
                    self.positive_indices.append(idx)
                else:
                    self.negative_indices.append(idx)
        
        # Balance if requested
        if balance and len(self.positive_indices) > 0:
            # Sample negative to match positive count
            n_positive = len(self.positive_indices)
            n_negative = len(self.negative_indices)
            
            if n_negative > n_positive:
                # Randomly sample negative examples
                np.random.seed(42)
                self.negative_indices = list(np.random.choice(
                    self.negative_indices, 
                    size=n_positive, 
                    replace=False
                ))
        
        # Combine indices
        self.indices = self.positive_indices + self.negative_indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get original sample
        original_idx = self.indices[idx]
        image, original_label = self.base_dataset[original_idx]
        
        # Convert to binary label
        binary_label = 1 if original_label == self.target_class else 0
        
        return image, binary_label


def get_cifar10_task_loaders(
    data_root: str = './data',
    num_tasks: int = 10,
    batch_size: int = 64,
    num_workers: int = 2,
    balance: bool = True,
    image_size: int = 224
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create task-as-class dataloaders for CIFAR-10.
    
    Args:
        data_root: Root directory for data
        num_tasks: Number of tasks (classes) to use
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        balance: Balance positive/negative samples
        image_size: Resize images to this size (224 for ViT)
        
    Returns:
        Tuple of (train_loaders, test_loaders), each a list of dataloaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load base CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create task-specific loaders
    train_loaders = []
    test_loaders = []
    
    for task_id in range(num_tasks):
        # Create task datasets
        task_train = TaskAsClassDataset(train_dataset, target_class=task_id, balance=balance)
        task_test = TaskAsClassDataset(test_dataset, target_class=task_id, balance=balance)
        
        # Create dataloaders
        train_loader = DataLoader(
            task_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            task_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
        print(f"Task {task_id} (class {task_id}): "
              f"Train={len(task_train)}, Test={len(task_test)}")
    
    return train_loaders, test_loaders


def get_dataset_info(dataset_name: str = 'cifar10') -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    if dataset_name == 'cifar10':
        return {
            'name': 'CIFAR-10',
            'num_classes': 10,
            'image_size': 32,
            'input_channels': 3,
            'classes': [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ],
            'task_setup': 'Each task is binary classification: one class vs all others',
            'num_train': 50000,
            'num_test': 10000
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
