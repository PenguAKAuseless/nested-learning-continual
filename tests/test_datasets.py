"""
Dataset Support Verification Script

Tests all supported datasets:
- CIFAR-10 (split into tasks)
- CIFAR-100 (split into tasks)
- MNIST (split into tasks)
- Permuted MNIST
- Rotated MNIST
- ImageNet-256 (if available)
- Custom Split Image Folders
"""

import torch
from data.datasets import (
    SplitCIFAR10, SplitCIFAR100, SplitMNIST,
    PermutedMNIST, RotatedMNIST, ImageNet256, SplitImageFolder
)
from pathlib import Path
from tqdm import tqdm


def test_dataset(dataset_name: str, dataset_class, **kwargs):
    """Test a dataset class"""
    print(f"\n{'='*70}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        # Initialize dataset
        print(f"Initializing {dataset_name}...")
        dataset = dataset_class(**kwargs)
        
        # Test getting tasks
        print(f"Number of tasks: {dataset.num_tasks}")
        
        # Test first task
        task_id = 0
        train_data, test_data = dataset.get_task(task_id), dataset.get_task(task_id, train=False)
        
        print(f"Task 0 - Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Test loading a few samples
        print("Testing data loading...")
        loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        
        for i, (x, y) in enumerate(tqdm(loader, desc="Loading batches", total=min(10, len(loader)))):
            if i >= 9:  # Test first 10 batches
                break
            print(f"  Batch {i+1}: x.shape={x.shape}, y.shape={y.shape}")
        
        print(f"✅ {dataset_name} - PASSED")
        return True
        
    except FileNotFoundError as e:
        print(f"⚠️ {dataset_name} - SKIPPED (Data not available: {e})")
        return None
    except Exception as e:
        print(f"❌ {dataset_name} - FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("DATASET SUPPORT VERIFICATION")
    print("="*70)
    
    results = {}
    
    # Test CIFAR-10
    results['Split-CIFAR10'] = test_dataset(
        'Split-CIFAR10',
        SplitCIFAR10,
        root='./data',
        num_tasks=5
    )
    
    # Test CIFAR-100
    results['Split-CIFAR100'] = test_dataset(
        'Split-CIFAR100',
        SplitCIFAR100,
        root='./data',
        num_tasks=10
    )
    
    # Test MNIST
    results['Split-MNIST'] = test_dataset(
        'Split-MNIST',
        SplitMNIST,
        root='./data',
        num_tasks=5
    )
    
    # Test Permuted MNIST
    results['Permuted-MNIST'] = test_dataset(
        'Permuted-MNIST',
        PermutedMNIST,
        root='./data',
        num_tasks=10
    )
    
    # Test Rotated MNIST
    results['Rotated-MNIST'] = test_dataset(
        'Rotated-MNIST',
        RotatedMNIST,
        root='./data',
        num_tasks=20
    )
    
    # Test ImageNet-256 (if available)
    imagenet_path = Path('./data/imagenet')
    if imagenet_path.exists():
        results['ImageNet-256'] = test_dataset(
            'ImageNet-256',
            ImageNet256,
            root='./data/imagenet',
            num_tasks=10,
            image_size=256
        )
    else:
        print(f"\n{'='*70}")
        print("Testing: ImageNet-256")
        print(f"{'='*70}")
        print(f"⚠️ ImageNet-256 - SKIPPED (Data not found at {imagenet_path})")
        print("To test ImageNet-256:")
        print("  1. Download ImageNet dataset")
        print("  2. Place in ./data/imagenet/ with 'train' and 'val' subdirectories")
        results['ImageNet-256'] = None
    
    # Test SplitImageFolder (if custom data available)
    custom_path = Path('./data/custom_images')
    if custom_path.exists():
        results['SplitImageFolder'] = test_dataset(
            'SplitImageFolder',
            SplitImageFolder,
            root='./data/custom_images',
            num_tasks=5,
            image_size=224
        )
    else:
        print(f"\n{'='*70}")
        print("Testing: SplitImageFolder")
        print(f"{'='*70}")
        print(f"⚠️ SplitImageFolder - SKIPPED (No custom data at {custom_path})")
        print("To test SplitImageFolder:")
        print("  1. Create ./data/custom_images/ directory")
        print("  2. Add class subdirectories with images")
        print("  3. Example: ./data/custom_images/class1/, ./data/custom_images/class2/, ...")
        results['SplitImageFolder'] = None
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for dataset_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️ SKIPPED"
        print(f"  {dataset_name:.<40} {status}")
    
    print(f"\n{'='*70}")
    print(f"Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"{'='*70}")
    
    # Verify required datasets
    required_datasets = ['Split-CIFAR10', 'Split-MNIST']
    all_required_passed = all(results.get(d) is True for d in required_datasets)
    
    if all_required_passed:
        print("\n✅ All required datasets (CIFAR-10, MNIST) are working!")
        print("   Additional datasets (CIFAR-100, Permuted/Rotated MNIST) available.")
        print("   ImageNet-256 and SplitImageFolder require manual data setup.")
        return 0
    else:
        print("\n❌ Some required datasets failed!")
        print("   Please install: pip install torch torchvision")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
