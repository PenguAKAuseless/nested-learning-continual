"""
Dataset utilities for continual learning experiments
"""

from .task_as_class import (
    TaskAsClassDataset,
    get_cifar10_task_loaders,
    get_dataset_info
)

__all__ = [
    'TaskAsClassDataset',
    'get_cifar10_task_loaders',
    'get_dataset_info'
]
