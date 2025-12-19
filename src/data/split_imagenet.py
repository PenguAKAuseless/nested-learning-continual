import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR100
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
from pathlib import Path
import pickle
from PIL import Image


class DiskBasedDataset(Dataset):
    """
    Dataset that loads data from disk incrementally to avoid memory issues.
    """
    def __init__(self, data_dir, transform=None):
        """
        Parameters:
        - data_dir: Path to directory containing saved samples
        - transform: Torchvision transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load index file
        index_file = self.data_dir / 'index.pkl'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        with open(index_file, 'rb') as f:
            self.samples = pickle.load(f)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # Load image from disk
        img_path = self.data_dir / sample_info['filename']
        image = Image.open(img_path).convert('RGB')
        label = sample_info['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def _save_cifar_task_to_disk(dataset, indices, save_dir, task_id, split_name):
    """
    Save CIFAR100 task data to disk.
    
    Parameters:
    - dataset: CIFAR100 dataset object
    - indices: List of indices for this task
    - save_dir: Directory to save the task data
    - task_id: Task identifier
    - split_name: 'train' or 'test'
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # Save each sample
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Convert tensor back to PIL Image if needed
        if isinstance(image, torch.Tensor):
            # Denormalize if needed
            image = transforms.ToPILImage()(image)
        
        # Save image
        filename = f'{split_name}_{i:06d}.png'
        img_path = save_dir / filename
        image.save(img_path)
        
        # Store metadata
        samples.append({
            'filename': filename,
            'label': label,
            'original_idx': idx
        })
    
    # Save index file
    index_file = save_dir / 'index.pkl'
    with open(index_file, 'wb') as f:
        pickle.dump(samples, f)


def _save_imagenet_task_to_disk(dataset, indices, train_dir, test_dir, task_id, train_split=0.8):
    """
    Save ImageNet task data to disk, split into train/test.
    
    Parameters:
    - dataset: ImageNet dataset object
    - indices: List of indices for this task
    - train_dir: Directory to save training data
    - test_dir: Directory to save test data
    - task_id: Task identifier
    - train_split: Ratio of data to use for training
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split indices
    torch.manual_seed(42 + task_id)
    perm = torch.randperm(len(indices)).tolist()
    shuffled_indices = [indices[i] for i in perm]
    
    split_point = int(len(shuffled_indices) * train_split)
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]
    
    # Save train data
    train_samples = []
    for i, idx in enumerate(train_indices):
        img_path, label = dataset.samples[idx]
        
        # Copy image to train directory
        image = Image.open(img_path).convert('RGB')
        filename = f'train_{i:06d}.png'
        new_path = train_dir / filename
        image.save(new_path)
        
        train_samples.append({
            'filename': filename,
            'label': label,
            'original_idx': idx
        })
    
    # Save test data
    test_samples = []
    for i, idx in enumerate(test_indices):
        img_path, label = dataset.samples[idx]
        
        # Copy image to test directory
        image = Image.open(img_path).convert('RGB')
        filename = f'test_{i:06d}.png'
        new_path = test_dir / filename
        image.save(new_path)
        
        test_samples.append({
            'filename': filename,
            'label': label,
            'original_idx': idx
        })
    
    # Save index files
    with open(train_dir / 'index.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    
    with open(test_dir / 'index.pkl', 'wb') as f:
        pickle.dump(test_samples, f)


def _load_task_datasets_from_disk(data_dir, dataset_name, num_tasks, use_cifar100=True):
    """
    Load task datasets from disk.
    
    Parameters:
    - data_dir: Base data directory
    - dataset_name: Name of the dataset
    - num_tasks: Number of tasks
    - use_cifar100: Whether using CIFAR100
    
    Returns:
    - tuple: (train_task_datasets, test_task_datasets)
    """
    splits_dir = Path(data_dir) / dataset_name / 'splits'
    
    # Define transformations
    if use_cifar100:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    train_task_datasets = []
    test_task_datasets = []
    
    for task_id in range(num_tasks):
        train_task_dir = splits_dir / f'task_{task_id}' / 'train'
        test_task_dir = splits_dir / f'task_{task_id}' / 'test'
        
        # Create disk-based datasets
        train_dataset = DiskBasedDataset(train_task_dir, transform=transform)
        test_dataset = DiskBasedDataset(test_task_dir, transform=transform)
        
        train_task_datasets.append(train_dataset)
        test_task_datasets.append(test_dataset)
    
    return train_task_datasets, test_task_datasets


def download_and_prepare_data(data_dir, dataset_name='CIFAR100'):
    """
    Download and prepare dataset for continual learning.
    For demo purposes, we use CIFAR100 as a substitute for ImageNet.
    
    Parameters:
    - data_dir: str, path to store the dataset
    - dataset_name: str, name of dataset ('CIFAR100' or 'ImageNet')
    
    Returns:
    - dataset: The downloaded dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == 'CIFAR100':
        # Check if CIFAR100 already exists
        cifar_check_path = os.path.join(data_dir, 'cifar-100-python')
        already_exists = os.path.exists(cifar_check_path)
        
        if already_exists:
            print(f"✓ CIFAR100 dataset already exists at {data_dir}")
        else:
            print(f"Downloading CIFAR100 dataset to {data_dir}...")
        
        # Use CIFAR100 as a substitute for ImageNet (100 classes)
        # download=True will only download if not already present
        train_dataset = CIFAR100(
            root=data_dir, 
            train=True, 
            download=True,
            transform=None  # We'll add transforms later
        )
        test_dataset = CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=None
        )
        
        if not already_exists:
            print(f"✓ CIFAR100 downloaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
        else:
            print(f"✓ CIFAR100 loaded: {len(train_dataset)} train samples, {len(test_dataset)} test samples")
        
        return train_dataset, test_dataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")


def create_split_imagenet_tasks(data_dir, num_tasks=10, classes_per_task=10, use_cifar100=True):
    """
    Create task splits for continual learning on ImageNet or CIFAR100.
    Each task corresponds to a different set of classes (label types).
    Saves splits to disk to avoid loading all data into memory at once.
    
    Parameters:
    - data_dir: str, path to the dataset directory
    - num_tasks: int, number of tasks to split the dataset into
    - classes_per_task: int, number of classes per task
    - use_cifar100: bool, whether to use CIFAR100 instead of ImageNet
    
    Returns:
    - tuple: (train_task_datasets, test_task_datasets) - lists of datasets for each task
    """
    # Create splits directory
    dataset_name = 'CIFAR100' if use_cifar100 else 'ImageNet'
    splits_dir = Path(data_dir) / dataset_name / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits already exist
    split_info_file = splits_dir / 'split_info.pkl'
    if split_info_file.exists():
        print(f"✓ Found existing task splits at {splits_dir}")
        with open(split_info_file, 'rb') as f:
            split_info = pickle.load(f)
        
        # Verify split configuration matches
        if (split_info['num_tasks'] == num_tasks and 
            split_info['classes_per_task'] == classes_per_task):
            print(f"  Loading {num_tasks} tasks with {classes_per_task} classes each")
            return _load_task_datasets_from_disk(data_dir, dataset_name, num_tasks, use_cifar100)
        else:
            print(f"  Configuration mismatch, regenerating splits...")
    
    # Define transformations
    if use_cifar100:
        # CIFAR100 uses 32x32 images
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                               std=[0.2675, 0.2565, 0.2761]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Load or download dataset
    train_dataset = None
    test_dataset = None
    full_dataset = None
    total_classes = 0
    
    if use_cifar100:
        train_dataset, test_dataset = download_and_prepare_data(data_dir, 'CIFAR100')
        total_classes = 100
    else:
        # Load ImageNet from directory
        train_dir = os.path.join(data_dir, 'train') # FIX: Auto point to subfolder `train` in `/data/imagenet-256`
        full_dataset = ImageFolder(root=train_dir, transform=transform)
        total_classes = len(full_dataset.classes)
    
    # Validate and adjust task configuration
    if total_classes < num_tasks * classes_per_task:
        print(f"Warning: Dataset has only {total_classes} classes, adjusting tasks...")
        classes_per_task = total_classes // num_tasks
    
    print(f"\n{'='*70}")
    print(f"CREATING TASK SPLITS - Saving to disk")
    print(f"{'='*70}")
    
    # Split classes into tasks and save to disk
    train_task_datasets = []
    test_task_datasets = []
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        
        # Create task directories
        train_task_dir = splits_dir / f'task_{task_id}' / 'train'
        test_task_dir = splits_dir / f'task_{task_id}' / 'test'
        train_task_dir.mkdir(parents=True, exist_ok=True)
        test_task_dir.mkdir(parents=True, exist_ok=True)
        
        if use_cifar100 and train_dataset is not None and test_dataset is not None:
            # Get indices for this task's classes
            train_indices = [i for i in range(len(train_dataset)) 
                           if start_class <= train_dataset[i][1] < end_class]
            test_indices = [i for i in range(len(test_dataset)) 
                          if start_class <= test_dataset[i][1] < end_class]
            
            # Save train data to disk
            _save_cifar_task_to_disk(train_dataset, train_indices, train_task_dir, task_id, 'train')
            
            # Save test data to disk
            _save_cifar_task_to_disk(test_dataset, test_indices, test_task_dir, task_id, 'test')
            
            print(f"Task {task_id}: Classes {start_class}-{end_class-1}, "
                  f"Train: {len(train_indices)} samples, Test: {len(test_indices)} samples")
        elif full_dataset is not None:
            # ImageNet case
            task_indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                           if start_class <= label < end_class]
            
            # Save ImageNet task to disk
            _save_imagenet_task_to_disk(full_dataset, task_indices, train_task_dir, test_task_dir, 
                                       task_id, train_split=0.8)
            
            train_count = int(len(task_indices) * 0.8)
            test_count = len(task_indices) - train_count
            print(f"Task {task_id}: Classes {start_class}-{end_class-1}, "
                  f"Train: {train_count} samples, Test: {test_count} samples")
        else:
            raise ValueError("Dataset not properly loaded")
    
    # Save split configuration
    split_info = {
        'num_tasks': num_tasks,
        'classes_per_task': classes_per_task,
        'dataset_name': dataset_name,
        'total_classes': total_classes
    }
    with open(split_info_file, 'wb') as f:
        pickle.dump(split_info, f)
    
    print(f"{'='*70}")
    print(f"✓ Task splits saved to {splits_dir}")
    print(f"{'='*70}\n")
    
    # Return datasets that load from disk
    return _load_task_datasets_from_disk(data_dir, dataset_name, num_tasks, use_cifar100)


def load_split_imagenet(data_dir, batch_size=32, num_workers=4, shuffle=True, task_id=None, num_tasks=10):
    """
    Load the Split ImageNet dataset.

    Parameters:
    - data_dir: str, path to the dataset directory
    - batch_size: int, number of samples per batch
    - num_workers: int, number of subprocesses to use for data loading
    - shuffle: bool, whether to shuffle the dataset
    - task_id: int or None, specific task to load (None loads all data)
    - num_tasks: int, number of tasks for splitting

    Returns:
    - DataLoader: DataLoader for the Split ImageNet dataset
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    dataset = ImageFolder(root=data_dir, transform=transform)

    # Create a DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle,
        pin_memory=True
    )

    return data_loader


def get_classes(data_dir):
    """
    Get the class names from the dataset.

    Parameters:
    - data_dir: str, path to the dataset directory.

    Returns:
    - list: List of class names.
    """
    return os.listdir(data_dir)