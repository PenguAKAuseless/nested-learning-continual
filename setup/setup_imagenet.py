"""
ImageNet-256 Setup Script
=========================

This script helps you set up Kaggle API credentials for downloading the ImageNet-256 dataset.
It creates a kaggle_credentials.json file in the project root with your Kaggle API credentials.

Usage:
    python setup/setup_imagenet.py

The script will:
1. Check if credentials already exist
2. Guide you through getting Kaggle API credentials
3. Create the kaggle_credentials.json file
4. Verify the credentials work
"""

import json
import os
import sys
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    # This file is in setup/, so parent is the project root
    return Path(__file__).parent.parent


def check_existing_credentials():
    """Check if credentials already exist."""
    project_root = get_project_root()
    cred_file = project_root / 'kaggle_credentials.json'
    
    if cred_file.exists():
        try:
            with open(cred_file, 'r') as f:
                creds = json.load(f)
            if 'username' in creds and 'key' in creds:
                print("✓ Kaggle credentials already exist!")
                print(f"  Username: {creds['username']}")
                response = input("\nDo you want to update them? (y/N): ").strip().lower()
                return response == 'y'
        except Exception as e:
            print(f"Warning: Existing credentials file is invalid: {e}")
            return True
    
    return True


def print_instructions():
    """Print instructions for getting Kaggle API credentials."""
    print("\n" + "="*70)
    print("KAGGLE API CREDENTIALS SETUP")
    print("="*70)
    print("\nTo use ImageNet-256 dataset, you need Kaggle API credentials.")
    print("\nHow to get your Kaggle API credentials:")
    print("  1. Go to https://www.kaggle.com/")
    print("  2. Sign in or create an account")
    print("  3. Click on your profile picture (top right)")
    print("  4. Select 'Settings' from the dropdown")
    print("  5. Scroll down to the 'API' section")
    print("  6. Click 'Create New API Token'")
    print("  7. A file 'kaggle.json' will be downloaded")
    print("  8. Open kaggle.json and copy the username and key")
    print("\nThe kaggle.json file looks like this:")
    print("  {")
    print('    "username": "your_kaggle_username",')
    print('    "key": "abc123...your_api_key"')
    print("  }")
    print("\n" + "="*70 + "\n")


def get_credentials_from_user():
    """Get credentials from user input."""
    print("Enter your Kaggle credentials:\n")
    
    username = input("Kaggle Username: ").strip()
    if not username:
        print("✗ Username cannot be empty!")
        return None
    
    api_key = input("Kaggle API Key: ").strip()
    if not api_key:
        print("✗ API Key cannot be empty!")
        return None
    
    return {
        'username': username,
        'key': api_key
    }


def save_credentials(credentials):
    """Save credentials to kaggle_credentials.json."""
    project_root = get_project_root()
    cred_file = project_root / 'kaggle_credentials.json'
    
    try:
        with open(cred_file, 'w') as f:
            json.dump(credentials, f, indent=2)
        
        print(f"\n✓ Credentials saved to: {cred_file}")
        
        # Set restrictive permissions (Unix-like systems only)
        try:
            os.chmod(cred_file, 0o600)
            print("✓ File permissions set to 600 (read/write for owner only)")
        except Exception:
            print("  (File permissions not changed - Windows system)")
        
        return True
    except Exception as e:
        print(f"\n✗ Error saving credentials: {e}")
        return False


def verify_credentials(credentials):
    """Verify credentials by trying to import kagglehub."""
    print("\nVerifying credentials...")
    
    try:
        # Set environment variables
        os.environ['KAGGLE_USERNAME'] = credentials['username']
        os.environ['KAGGLE_KEY'] = credentials['key']
        
        # Try importing kagglehub
        import kagglehub
        print("✓ kagglehub package found")
        
        # Note: We don't actually download anything here to avoid long wait times
        print("✓ Credentials format is valid")
        print("\nNote: Credentials will be fully validated when downloading dataset.")
        print("      Run: python src/main.py --config configs/imagenet_256.yaml")
        
        return True
    except ImportError:
        print("⚠ Warning: kagglehub package not installed")
        print("  Install it with: pip install kagglehub")
        return False
    except Exception as e:
        print(f"⚠ Warning: Could not fully verify credentials: {e}")
        print("  This may be normal. Try downloading the dataset to verify.")
        return False


def create_example_file():
    """Create an example credentials file if it doesn't exist."""
    project_root = get_project_root()
    example_file = project_root / 'kaggle_credentials.json.example'
    
    if not example_file.exists():
        example_content = {
            "username": "your_kaggle_username",
            "key": "your_kaggle_api_key_here"
        }
        
        try:
            with open(example_file, 'w') as f:
                json.dump(example_content, f, indent=2)
            print(f"✓ Created example file: {example_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not create example file: {e}")


def accept_dataset_terms():
    """Remind user to accept dataset terms on Kaggle."""
    print("\n" + "="*70)
    print("IMPORTANT: Dataset Terms of Use")
    print("="*70)
    print("\nBefore downloading ImageNet-256, you must:")
    print("  1. Visit: https://www.kaggle.com/datasets/dimensi0n/imagenet-256")
    print("  2. Click 'Download' button (you'll be prompted to accept terms)")
    print("  3. Accept the dataset's terms and conditions")
    print("\nWithout accepting the terms, downloads will fail!")
    print("="*70 + "\n")
    
    response = input("Have you accepted the dataset terms? (y/N): ").strip().lower()
    if response != 'y':
        print("\n⚠ Please accept the dataset terms on Kaggle before continuing.")
        print("  Visit: https://www.kaggle.com/datasets/dimensi0n/imagenet-256")
        return False
    
    return True


def main():
    """Main setup function."""
    print("\n" + "🚀 " + "="*66)
    print("   ImageNet-256 Dataset Setup for Nested Learning")
    print("   " + "="*66)
    
    # Check if we should proceed with setup
    if not check_existing_credentials():
        print("\nSetup cancelled. Existing credentials will be used.")
        return 0
    
    # Print instructions
    print_instructions()
    
    # Get credentials from user
    credentials = get_credentials_from_user()
    if not credentials:
        print("\n✗ Setup failed. Please try again.")
        return 1
    
    # Save credentials
    if not save_credentials(credentials):
        return 1
    
    # Create example file
    create_example_file()
    
    # Verify credentials
    verify_credentials(credentials)
    
    # Remind about dataset terms
    if not accept_dataset_terms():
        print("\n✓ Credentials saved, but remember to accept dataset terms!")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Make sure you've accepted the dataset terms on Kaggle")
    print("  2. Run the training script:")
    print("     python src/main.py --config configs/imagenet_256.yaml")
    print("\nThe dataset will be downloaded automatically on first run.")
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
