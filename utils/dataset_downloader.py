"""
Dataset downloader utility for continual learning framework.
Provides automatic download and integrity verification for supported datasets.
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import tarfile
import zipfile
import shutil
from typing import Optional, Dict, Tuple
import torch
from torchvision import datasets


class DatasetDownloader:
    """Utility class to download and verify datasets."""
    
    # Dataset configurations with download URLs and checksums
    DATASET_CONFIGS = {
        'cifar10': {
            'name': 'CIFAR-10',
            'auto_download': True,  # torchvision handles this
            'torchvision': True,
            'size': '170 MB',
        },
        'cifar100': {
            'name': 'CIFAR-100',
            'auto_download': True,  # torchvision handles this
            'torchvision': True,
            'size': '169 MB',
        },
        'mnist': {
            'name': 'MNIST',
            'auto_download': True,  # torchvision handles this
            'torchvision': True,
            'size': '11 MB',
        },
        'imagenet256': {
            'name': 'ImageNet-256',
            'auto_download': True,  # via Kaggle API
            'torchvision': False,
            'kaggle': True,
            'kaggle_dataset': 'dimensi0n/imagenet-256',
            'size': '~37 GB (compressed)',
            'extract_to': 'imagenet'
        },
        'imagenet': {
            'name': 'ImageNet (Full)',
            'auto_download': False,  # requires manual download due to licensing
            'torchvision': False,
            'size': '~150 GB',
            'instructions': [
                '1. Register at https://image-net.org/download.php',
                '2. Download ILSVRC2012 training and validation sets',
                '3. Extract to data/imagenet/ with train/ and val/ subdirectories',
                '4. Run this script with --verify to check integrity'
            ]
        }
    }
    
    def __init__(self, data_root: str = './data'):
        """
        Initialize dataset downloader.
        
        Args:
            data_root: Root directory for storing datasets
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._check_kaggle_setup()
        
    def download(self, dataset_name: str, force: bool = False) -> bool:
        """
        Download a dataset.
        
        Args:
            dataset_name: Name of dataset to download
            force: Force re-download even if exists
            
        Returns:
            True if successful, False otherwise
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.DATASET_CONFIGS:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"   Available: {', '.join(self.DATASET_CONFIGS.keys())}")
            return False
        
        config = self.DATASET_CONFIGS[dataset_name]
        print(f"\n{'='*70}")
        print(f"Downloading: {config['name']}")
        print(f"Size: {config['size']}")
        print(f"{'='*70}\n")
        
        if not config['auto_download']:
            print(f"⚠️  {config['name']} requires manual download:")
            for instruction in config['instructions']:
                print(f"   {instruction}")
            return False
        
        # Download using torchvision
        if config.get('torchvision'):
            return self._download_torchvision_dataset(dataset_name, force)
        
        # Download using Kaggle API
        if config.get('kaggle'):
            return self._download_kaggle_dataset(dataset_name, force)
        
        return False
    
    def _download_torchvision_dataset(self, dataset_name: str, force: bool) -> bool:
        """Download dataset using torchvision."""
        try:
            dataset_path = self.data_root / dataset_name
            
            # Check if already exists
            if dataset_path.exists() and not force:
                print(f"✅ {dataset_name.upper()} already exists at {dataset_path}")
                print("   Use --force to re-download")
                return True
            
            print(f"📥 Downloading {dataset_name.upper()}...")
            
            if dataset_name == 'cifar10':
                datasets.CIFAR10(root=str(self.data_root), train=True, download=True)
                datasets.CIFAR10(root=str(self.data_root), train=False, download=True)
            elif dataset_name == 'cifar100':
                datasets.CIFAR100(root=str(self.data_root), train=True, download=True)
                datasets.CIFAR100(root=str(self.data_root), train=False, download=True)
            elif dataset_name == 'mnist':
                datasets.MNIST(root=str(self.data_root), train=True, download=True)
                datasets.MNIST(root=str(self.data_root), train=False, download=True)
            
            print(f"✅ Successfully downloaded {dataset_name.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def _check_kaggle_setup(self) -> bool:
        """Check if Kaggle API is properly configured."""
        try:
            import kaggle
            return True
        except OSError as e:
            # Kaggle API not configured, but that's okay
            return False
        except ImportError:
            # Kaggle package not installed
            return False
    
    def _setup_kaggle_instructions(self):
        """Print Kaggle setup instructions."""
        print("\n" + "="*70)
        print("KAGGLE API SETUP REQUIRED")
        print("="*70)
        print("\n1. Install Kaggle package:")
        print("   pip install kaggle")
        print("\n2. Create Kaggle API credentials:")
        print("   a. Go to https://www.kaggle.com/settings/account")
        print("   b. Scroll to 'API' section")
        print("   c. Click 'Create New Token'")
        print("   d. Download kaggle.json")
        print("\n3. Place kaggle.json in the correct location:")
        if os.name == 'nt':  # Windows
            kaggle_dir = Path.home() / '.kaggle'
            print(f"   {kaggle_dir}\\kaggle.json")
        else:  # Linux/Mac
            print("   ~/.kaggle/kaggle.json")
            print("   chmod 600 ~/.kaggle/kaggle.json")
        print("\n" + "="*70 + "\n")
    
    def _download_kaggle_dataset(self, dataset_name: str, force: bool) -> bool:
        """Download dataset using Kaggle API."""
        try:
            # Import kaggle here to provide better error messages
            try:
                import kaggle
                from kaggle.api.kaggle_api_extended import KaggleApi
            except ImportError:
                print("❌ Kaggle package not installed")
                self._setup_kaggle_instructions()
                print("Install with: pip install kaggle\n")
                return False
            except OSError as e:
                print("❌ Kaggle API credentials not configured")
                self._setup_kaggle_instructions()
                return False
            
            config = self.DATASET_CONFIGS[dataset_name]
            kaggle_dataset = config['kaggle_dataset']
            extract_to = config.get('extract_to', dataset_name)
            
            dataset_path = self.data_root / extract_to
            
            # Check if already exists
            if dataset_path.exists() and not force:
                print(f"✅ {config['name']} already exists at {dataset_path}")
                print("   Use --force to re-download")
                return True
            
            print(f"📥 Downloading {config['name']} from Kaggle...")
            print(f"   Dataset: {kaggle_dataset}")
            print(f"   This may take a while ({config['size']})...\n")
            
            # Authenticate
            api = KaggleApi()
            api.authenticate()
            
            # Download to temporary location
            temp_dir = self.data_root / f"_temp_{dataset_name}"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Download dataset
                print("📦 Downloading archive...")
                api.dataset_download_files(
                    kaggle_dataset,
                    path=str(temp_dir),
                    unzip=False,
                    quiet=False
                )
                
                # Find the downloaded zip file
                zip_files = list(temp_dir.glob("*.zip"))
                if not zip_files:
                    print("❌ No zip file found after download")
                    return False
                
                zip_file = zip_files[0]
                print(f"\n📂 Extracting {zip_file.name}...")
                print(f"   Target: {dataset_path}")
                
                # Extract
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    # Get total size for progress bar
                    total_size = sum(info.file_size for info in zip_ref.infolist())
                    
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                        for member in zip_ref.infolist():
                            zip_ref.extract(member, dataset_path)
                            pbar.update(member.file_size)
                
                print(f"\n✅ Successfully downloaded and extracted {config['name']}")
                print(f"   Location: {dataset_path}")
                
                return True
                
            finally:
                # Cleanup temp directory
                if temp_dir.exists():
                    print("\n🧹 Cleaning up temporary files...")
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify(self, dataset_name: str) -> Tuple[bool, str]:
        """
        Verify dataset integrity.
        
        Args:
            dataset_name: Name of dataset to verify
            
        Returns:
            Tuple of (is_valid, message)
        """
        dataset_name = dataset_name.lower()
        
        if dataset_name not in self.DATASET_CONFIGS:
            return False, f"Unknown dataset: {dataset_name}"
        
        config = self.DATASET_CONFIGS[dataset_name]
        
        # Get the correct directory name for torchvision datasets
        if config.get('torchvision'):
            if dataset_name == 'cifar10':
                dataset_path = self.data_root / 'cifar-10-batches-py'
            elif dataset_name == 'cifar100':
                dataset_path = self.data_root / 'cifar-100-python'
            elif dataset_name == 'mnist':
                dataset_path = self.data_root / 'MNIST'
            else:
                dataset_path = self.data_root / dataset_name
        else:
            dataset_path = self.data_root / dataset_name
        
        if not dataset_path.exists():
            return False, f"Dataset directory not found: {dataset_path}"
        
        print(f"\n{'='*70}")
        print(f"Verifying: {config['name']}")
        print(f"{'='*70}\n")
        
        # Verify torchvision datasets
        if config.get('torchvision'):
            return self._verify_torchvision_dataset(dataset_name)
        
        # Verify ImageNet (both full and 256)
        if dataset_name in ['imagenet', 'imagenet256']:
            return self._verify_imagenet()
        
        return True, "Verification not implemented for this dataset"
    
    def _verify_torchvision_dataset(self, dataset_name: str) -> Tuple[bool, str]:
        """Verify torchvision dataset by loading it."""
        try:
            print(f"🔍 Checking {dataset_name.upper()} integrity...")
            
            if dataset_name == 'cifar10':
                train_data = datasets.CIFAR10(root=str(self.data_root), train=True, download=False)
                test_data = datasets.CIFAR10(root=str(self.data_root), train=False, download=False)
                expected_train = 50000
                expected_test = 10000
            elif dataset_name == 'cifar100':
                train_data = datasets.CIFAR100(root=str(self.data_root), train=True, download=False)
                test_data = datasets.CIFAR100(root=str(self.data_root), train=False, download=False)
                expected_train = 50000
                expected_test = 10000
            elif dataset_name == 'mnist':
                train_data = datasets.MNIST(root=str(self.data_root), train=True, download=False)
                test_data = datasets.MNIST(root=str(self.data_root), train=False, download=False)
                expected_train = 60000
                expected_test = 10000
            else:
                return False, "Unknown dataset"
            
            # Check sample counts
            if len(train_data) != expected_train:
                return False, f"Train set size mismatch: {len(train_data)} vs {expected_train}"
            
            if len(test_data) != expected_test:
                return False, f"Test set size mismatch: {len(test_data)} vs {expected_test}"
            
            # Try loading a sample
            sample, label = train_data[0]
            
            print(f"✅ Dataset verified successfully")
            print(f"   Train samples: {len(train_data):,}")
            print(f"   Test samples: {len(test_data):,}")
            print(f"   Sample shape: {sample.size if hasattr(sample, 'size') else 'N/A'}")
            
            return True, "Dataset verified successfully"
            
        except Exception as e:
            return False, f"Verification failed: {e}"
    
    def _verify_imagenet(self) -> Tuple[bool, str]:
        """Verify ImageNet dataset structure."""
        imagenet_path = self.data_root / 'imagenet'
        
        if not imagenet_path.exists():
            return False, "ImageNet directory not found"
        
        train_path = imagenet_path / 'train'
        val_path = imagenet_path / 'val'
        
        issues = []
        
        if not train_path.exists():
            issues.append("Missing train/ directory")
        else:
            train_classes = list(train_path.iterdir())
            print(f"   Train classes found: {len(train_classes)}")
            if len(train_classes) != 1000:
                issues.append(f"Expected 1000 train classes, found {len(train_classes)}")
        
        if not val_path.exists():
            issues.append("Missing val/ directory")
        else:
            val_classes = list(val_path.iterdir())
            print(f"   Val classes found: {len(val_classes)}")
            if len(val_classes) != 1000:
                issues.append(f"Expected 1000 val classes, found {len(val_classes)}")
        
        if issues:
            return False, "\n".join(issues)
        
        print(f"✅ ImageNet structure verified")
        return True, "ImageNet verified successfully"
    
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Download all auto-downloadable datasets.
        
        Args:
            force: Force re-download even if exists
            
        Returns:
            Dictionary of dataset_name -> success status
        """
        results = {}
        
        print("\n" + "="*70)
        print("DOWNLOADING ALL DATASETS")
        print("="*70)
        
        for dataset_name, config in self.DATASET_CONFIGS.items():
            if config['auto_download']:
                results[dataset_name] = self.download(dataset_name, force)
            else:
                print(f"\n⏭️  Skipping {config['name']} (requires manual download)")
                results[dataset_name] = False
        
        return results
    
    def verify_all(self) -> Dict[str, bool]:
        """
        Verify all downloaded datasets.
        
        Returns:
            Dictionary of dataset_name -> is_valid
        """
        results = {}
        
        print("\n" + "="*70)
        print("VERIFYING ALL DATASETS")
        print("="*70)
        
        for dataset_name in self.DATASET_CONFIGS.keys():
            is_valid, message = self.verify(dataset_name)
            results[dataset_name] = is_valid
            
            if not is_valid:
                print(f"\n⚠️  {dataset_name}: {message}")
        
        return results
    
    def list_datasets(self):
        """List all available datasets and their status."""
        print("\n" + "="*70)
        print("AVAILABLE DATASETS")
        print("="*70 + "\n")
        
        for dataset_name, config in self.DATASET_CONFIGS.items():
            # Check for dataset existence based on type
            if config.get('torchvision'):
                # For torchvision datasets, check for actual data folders
                if dataset_name == 'cifar10':
                    exists = (self.data_root / 'cifar-10-batches-py').exists()
                elif dataset_name == 'cifar100':
                    exists = (self.data_root / 'cifar-100-python').exists()
                elif dataset_name == 'mnist':
                    exists = (self.data_root / 'MNIST').exists()
                else:
                    exists = (self.data_root / dataset_name).exists()
            elif config.get('kaggle'):
                # For Kaggle datasets, check extract_to location
                extract_to = config.get('extract_to', dataset_name)
                exists = (self.data_root / extract_to).exists()
            else:
                # Default check
                dataset_path = self.data_root / dataset_name
                exists = dataset_path.exists()
            
            status = "✅ Downloaded" if exists else "❌ Not downloaded"
            auto = "🤖 Auto" if config.get('auto_download') else "📋 Manual"
            
            print(f"{config['name']:<20} {status:<20} {auto:<15} {config['size']}")
        
        print(f"\n{'='*70}\n")


def main():
    """Command-line interface for dataset downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and verify datasets')
    parser.add_argument('--download', type=str, help='Download specific dataset')
    parser.add_argument('--download-all', action='store_true', help='Download all datasets')
    parser.add_argument('--verify', type=str, help='Verify specific dataset')
    parser.add_argument('--verify-all', action='store_true', help='Verify all datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_datasets()
    elif args.download:
        downloader.download(args.download, args.force)
    elif args.download_all:
        downloader.download_all(args.force)
    elif args.verify:
        is_valid, message = downloader.verify(args.verify)
        print(f"\n{message}\n")
    elif args.verify_all:
        downloader.verify_all()
    else:
        downloader.list_datasets()
        print("Use --help for more options\n")


if __name__ == '__main__':
    main()
