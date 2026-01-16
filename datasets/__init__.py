"""
Dataset utilities for continual learning experiments
"""

from .task_as_class import (
    TaskDataset,
    get_cifar10_task_loaders,
    get_dataset_info
)

__all__ = [
    'TaskDataset',
    'get_cifar10_task_loaders',
    'get_dataset_info'
]
