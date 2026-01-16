"""
Task-based Dataset Loaders for Continual Learning

Each task learns a subset of classes with shared output space.
For CIFAR-10 with 5 tasks:
Task 0: classes 0-1
Task 1: classes 2-3
Task 2: classes 4-5
Task 3: classes 6-7
Task 4: classes 8-9

All tasks share the same output space (0-9), allowing multi-task evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List, Optional


class TaskDataset(Dataset):
    """
    Creates a task-specific dataset from a subset of classes.
    
    Args:
        base_dataset: The underlying dataset (e.g., CIFAR10)
        task_classes: List of class indices for this task (e.g., [0, 1])
    """
    def __init__(self, base_dataset, task_classes: List[int]):
        self.base_dataset = base_dataset
        self.task_classes = task_classes
        
        # Get indices for samples belonging to task classes
        if hasattr(base_dataset, 'targets'):
            # Fast path for datasets with targets attribute
            targets = base_dataset.targets
            if isinstance(targets, list):
                targets = np.array(targets)
            
            # Find all samples belonging to any of the task classes
            mask = np.isin(targets, task_classes)
            self.indices = np.where(mask)[0].tolist()
        else:
            # Fallback to slower method
            self.indices = []
            for idx in range(len(base_dataset)):
                _, label = base_dataset[idx]
                if label in task_classes:
                    self.indices.append(idx)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get original sample - labels remain unchanged (0-9 for CIFAR-10)
        original_idx = self.indices[idx]
        image, label = self.base_dataset[original_idx]
        return image, label


def get_cifar10_task_loaders(
    data_root: str = './data',
    num_tasks: int = 5,
    batch_size: int = 64,
    num_workers: int = 2,
    balance: bool = True,
    image_size: int = 224
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Create task-based dataloaders for CIFAR-10.
    
    Splits 10 classes into num_tasks tasks evenly.
    For 5 tasks: Task 0=[0,1], Task 1=[2,3], Task 2=[4,5], Task 3=[6,7], Task 4=[8,9]
    All tasks share the same output space (0-9), enabling multi-task evaluation.
    
    Args:
        data_root: Root directory for data
        num_tasks: Number of tasks to split classes into (default: 5 for CIFAR-10)
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
        balance: Not used (kept for compatibility)
        image_size: Resize images to this size (224 for ViT)
        
    Returns:
        Tuple of (train_loaders, test_loaders), each a list of dataloaders
    """
    # CIFAR-10 has 10 classes
    total_classes = 10
    classes_per_task = total_classes // num_tasks
    
    if total_classes % num_tasks != 0:
        raise ValueError(f"Cannot evenly split {total_classes} classes into {num_tasks} tasks")
    
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
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for task_id in range(num_tasks):
        # Determine which classes belong to this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        task_classes = list(range(start_class, end_class))
        
        # Create task datasets
        task_train = TaskDataset(train_dataset, task_classes)
        task_test = TaskDataset(test_dataset, task_classes)
        
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
        
        class_names_str = ', '.join([class_names[c] for c in task_classes])
        print(f"Task {task_id} (classes {task_classes} = {class_names_str}): "
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
            'task_setup': 'Multi-class continual learning: classes split across tasks with shared output space (0-9)',
            'num_train': 50000,
            'num_test': 10000
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
