"""
ImageNet 256x256 Dataset Loader for Continual Learning
Downloads from Kaggle using kagglehub with secure credential management
Saves train/test splits to disk to avoid memory issues
"""
import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, Dataset
from pathlib import Path
import kagglehub
import pickle
from PIL import Image
import shutil


def load_kaggle_credentials(credentials_path='kaggle_credentials.json'):
    """
    Load Kaggle API credentials from a separate JSON file.
    
    Parameters:
    - credentials_path: str, path to the credentials JSON file
    
    Returns:
    - dict: Kaggle credentials {'username': '...', 'key': '...'}
    
    The credentials file should have the format:
    {
        "username": "your_kaggle_username",
        "key": "your_kaggle_api_key"
    }
    
    To get your Kaggle API credentials:
    1. Go to https://www.kaggle.com/
    2. Click on your profile picture → Account
    3. Scroll to API section
    4. Click "Create New API Token"
    5. Save the downloaded kaggle.json content to kaggle_credentials.json
    """
    # Look for credentials file in project root
    project_root = Path(__file__).parent.parent.parent
    cred_file = project_root / credentials_path
    
    if not cred_file.exists():
        raise FileNotFoundError(
            f"Kaggle credentials file not found at {cred_file}\n\n"
            f"Please create '{credentials_path}' in the project root with your Kaggle API credentials:\n"
            f"{{\n"
            f'  "username": "your_kaggle_username",\n'
            f'  "key": "your_kaggle_api_key"\n'
            f"}}\n\n"
            f"To get your credentials:\n"
            f"1. Go to https://www.kaggle.com/\n"
            f"2. Click on your profile picture → Account\n"
            f"3. Scroll to API section\n"
            f"4. Click 'Create New API Token'\n"
            f"5. Copy the content from the downloaded kaggle.json file\n"
        )
    
    with open(cred_file, 'r') as f:
        credentials = json.load(f)
    
    if 'username' not in credentials or 'key' not in credentials:
        raise ValueError(
            f"Invalid credentials file format. Expected:\n"
            f"{{\n"
            f'  "username": "your_kaggle_username",\n'
            f'  "key": "your_kaggle_api_key"\n'
            f"}}"
        )
    
    return credentials


def setup_kaggle_credentials():
    """
    Set up Kaggle credentials as environment variables for kagglehub.
    This avoids storing credentials in code.
    """
    credentials = load_kaggle_credentials()
    
    # Set environment variables for kagglehub
    os.environ['KAGGLE_USERNAME'] = credentials['username']
    os.environ['KAGGLE_KEY'] = credentials['key']
    
    print(f"✓ Kaggle credentials loaded for user: {credentials['username']}")


def download_imagenet_256(data_dir='./data/imagenet-256'):
    """
    Download ImageNet 256x256 dataset from Kaggle.
    Checks if dataset already exists before downloading.
    
    Parameters:
    - data_dir: str, directory to store the downloaded dataset
    
    Returns:
    - str: Path to the downloaded dataset
    """
    print("\n" + "="*70)
    print("IMAGENET 256x256 DATASET LOADER")
    print("="*70)
    
    # Check if dataset already exists at specified location
    data_path = Path(data_dir)
    if data_path.exists() and any(data_path.iterdir()):
        print(f"\n✓ ImageNet dataset already exists at: {data_dir}")
        # Verify it's a valid ImageNet directory
        try:
            test_dataset = ImageFolder(root=data_dir)
            print(f"✓ Verified: {len(test_dataset.classes)} classes found")
            return str(data_dir)
        except Exception as e:
            print(f"⚠ Warning: Directory exists but may not be valid ImageNet format: {e}")
            print(f"Proceeding with download...")
    
    # Setup Kaggle credentials
    setup_kaggle_credentials()
    
    # Download dataset using kagglehub
    print("\nDownloading dataset from Kaggle (this may take a while for large datasets)...")
    try:
        path = kagglehub.dataset_download("dimensi0n/imagenet-256")
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"Path to dataset files: {path}")
        
        # Optionally copy/move to desired location
        if data_dir and data_dir != path:
            print(f"\nNote: Dataset is stored at: {path}")
            print(f"You can specify this path in your config or symlink to: {data_dir}")
        
        return path
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your Kaggle credentials in kaggle_credentials.json")
        print("2. Ensure you have accepted the dataset's terms on Kaggle website:")
        print("   https://www.kaggle.com/datasets/dimensi0n/imagenet-256")
        print("3. Check your internet connection")
        raise


class DiskBasedImageNetDataset(Dataset):
    """
    Dataset that loads ImageNet data from disk incrementally to avoid memory issues.
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


def _save_imagenet_task_split(source_dataset, indices, save_dir, split_name, task_id):
    """
    Save ImageNet task split to disk.
    
    Parameters:
    - source_dataset: ImageFolder dataset
    - indices: List of indices for this split
    - save_dir: Directory to save the data
    - split_name: 'train' or 'test'
    - task_id: Task identifier
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    print(f"  Saving {split_name} split ({len(indices)} samples)...", end='', flush=True)
    
    # Save each sample
    for i, idx in enumerate(indices):
        img_path, label = source_dataset.samples[idx]
        
        # Load and save image
        try:
            image = Image.open(img_path).convert('RGB')
            filename = f'{split_name}_{i:06d}.jpg'
            new_path = save_dir / filename
            image.save(new_path, quality=95)
            
            # Store metadata
            samples.append({
                'filename': filename,
                'label': label,
                'original_idx': idx,
                'original_path': img_path
            })
        except Exception as e:
            print(f"\n  Warning: Failed to save image {img_path}: {e}")
            continue
    
    # Save index file
    index_file = save_dir / 'index.pkl'
    with open(index_file, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f" Done ({len(samples)} samples saved)")


def create_imagenet_256_tasks(data_path, num_tasks=10, classes_per_task=100, 
                               train_split=0.8, random_seed=42):
    """
    Create task splits for continual learning on ImageNet 256x256.
    Each task corresponds to a different set of classes.
    Saves splits to disk to avoid loading all data into memory at once.
    
    Parameters:
    - data_path: str, path to the ImageNet dataset directory
    - num_tasks: int, number of tasks to split the dataset into
    - classes_per_task: int, number of classes per task
    - train_split: float, ratio of data to use for training (rest for testing)
    - random_seed: int, random seed for reproducibility
    
    Returns:
    - tuple: (train_task_datasets, test_task_datasets) - lists of datasets for each task
    """
    # Create splits directory
    data_path = Path(data_path)
    splits_dir = data_path.parent / 'ImageNet' / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits already exist
    split_info_file = splits_dir / 'split_info.pkl'
    if split_info_file.exists():
        print(f"\n✓ Found existing task splits at {splits_dir}")
        with open(split_info_file, 'rb') as f:
            split_info = pickle.load(f)
        
        # Verify split configuration matches
        if (split_info['num_tasks'] == num_tasks and 
            split_info['classes_per_task'] == classes_per_task and
            split_info['train_split'] == train_split):
            print(f"  Loading {num_tasks} tasks with {classes_per_task} classes each")
            return _load_imagenet_task_datasets_from_disk(splits_dir, num_tasks)
        else:
            print(f"  Configuration mismatch, regenerating splits...")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Define transformations (will be applied when loading from disk)
    print("\n" + "="*70)
    print("CREATING IMAGENET TASK SPLITS - Saving to disk")
    print("="*70)
    
    # Load full dataset WITHOUT transform (we'll save raw images)
    print(f"\nLoading ImageNet dataset from: {data_path}")
    full_dataset = ImageFolder(root=str(data_path), transform=None)
    total_classes = len(full_dataset.classes)
    total_samples = len(full_dataset)
    
    print(f"✓ Dataset loaded: {total_samples:,} images, {total_classes} classes")
    
    # Validate configuration
    total_required_classes = num_tasks * classes_per_task
    if total_classes < total_required_classes:
        print(f"\n⚠ Warning: Dataset has only {total_classes} classes")
        print(f"   Requested: {num_tasks} tasks × {classes_per_task} classes = {total_required_classes}")
        classes_per_task = total_classes // num_tasks
        print(f"   Adjusted to: {classes_per_task} classes per task")
    
    # Split classes into tasks
    print(f"\nSplitting into {num_tasks} tasks with {classes_per_task} classes each:")
    print("-" * 70)
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = min(start_class + classes_per_task, total_classes)
        
        print(f"\nTask {task_id}: Classes {start_class}-{end_class-1}")
        
        # Get indices for this task's classes
        task_indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                       if start_class <= label < end_class]
        
        # Shuffle and split into train/test
        torch.manual_seed(random_seed + task_id)
        perm = torch.randperm(len(task_indices)).tolist()
        shuffled_indices = [task_indices[i] for i in perm]
        
        split_point = int(len(shuffled_indices) * train_split)
        train_indices = shuffled_indices[:split_point]
        test_indices = shuffled_indices[split_point:]
        
        # Create task directories
        train_task_dir = splits_dir / f'task_{task_id}' / 'train'
        test_task_dir = splits_dir / f'task_{task_id}' / 'test'
        
        # Save train and test splits to disk
        _save_imagenet_task_split(full_dataset, train_indices, train_task_dir, 'train', task_id)
        _save_imagenet_task_split(full_dataset, test_indices, test_task_dir, 'test', task_id)
        
        # Get class names for this task
        class_names = [full_dataset.classes[i] for i in range(start_class, end_class)]
        print(f"  Classes: {class_names[0][:20]}... to {class_names[-1][:20]}...")
    
    print("-" * 70)
    
    # Save split configuration
    split_info = {
        'num_tasks': num_tasks,
        'classes_per_task': classes_per_task,
        'train_split': train_split,
        'random_seed': random_seed,
        'total_classes': total_classes,
        'dataset_path': str(data_path)
    }
    with open(split_info_file, 'wb') as f:
        pickle.dump(split_info, f)
    
    print(f"\n✓ Task splits saved to {splits_dir}")
    print("=" * 70)
    
    # Return datasets that load from disk
    return _load_imagenet_task_datasets_from_disk(splits_dir, num_tasks)


def _load_imagenet_task_datasets_from_disk(splits_dir, num_tasks):
    """
    Load ImageNet task datasets from disk.
    
    Parameters:
    - splits_dir: Directory containing saved task splits
    - num_tasks: Number of tasks
    
    Returns:
    - tuple: (train_task_datasets, test_task_datasets)
    """
    splits_dir = Path(splits_dir)
    
    # Define transformations for train and test
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_task_datasets = []
    test_task_datasets = []
    
    print(f"\nLoading {num_tasks} tasks from disk:")
    for task_id in range(num_tasks):
        train_task_dir = splits_dir / f'task_{task_id}' / 'train'
        test_task_dir = splits_dir / f'task_{task_id}' / 'test'
        
        # Create disk-based datasets
        train_dataset = DiskBasedImageNetDataset(train_task_dir, transform=train_transform)
        test_dataset = DiskBasedImageNetDataset(test_task_dir, transform=test_transform)
        
        train_task_datasets.append(train_dataset)
        test_task_datasets.append(test_dataset)
        
        print(f"  Task {task_id}: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_task_datasets, test_task_datasets


def get_imagenet_info(data_path):
    """
    Get information about the ImageNet dataset.
    
    Parameters:
    - data_path: str, path to the ImageNet dataset directory
    
    Returns:
    - dict: Dataset information including number of classes, samples, etc.
    """
    dataset = ImageFolder(root=data_path)
    
    info = {
        'num_classes': len(dataset.classes),
        'num_samples': len(dataset),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    print("ImageNet 256x256 Dataset Loader")
    print("=" * 70)
    
    # Download dataset
    try:
        dataset_path = download_imagenet_256()
        
        # Get dataset info
        info = get_imagenet_info(dataset_path)
        print(f"\nDataset Info:")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Samples: {info['num_samples']:,}")
        
        # Create task splits (example)
        print("\nCreating example task splits...")
        train_tasks, test_tasks = create_imagenet_256_tasks(
            data_path=dataset_path,
            num_tasks=10,
            classes_per_task=100
        )
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
