import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

from utils.config import load_config
# Import both old and new model architectures for compatibility
from models.nested_network import NestedNetwork as OldNestedNetwork
from models.nested_learning_network import NestedLearningNetwork, NestedNetwork
from training.nested_optimizer import create_nested_optimizer
from data.split_imagenet import create_split_imagenet_tasks
from data.imagenet_loader import download_imagenet_256, create_imagenet_256_tasks
from data.ood_generator import create_ood_dataset
from training.continual_learner import ContinualLearner

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Continual Learning with Nested Networks')
    parser.add_argument('--config', type=str, default='configs/split_imagenet.yaml',
                        help='Path to configuration file (default: configs/split_imagenet.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        # If relative path, make it relative to project root
        project_root = os.path.join(os.path.dirname(__file__), '..')
        config_path = os.path.join(project_root, config_path)
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("="*70)
    print("REALISTIC CONTINUAL LEARNING WITH NESTED LEARNING")
    print("="*70)
    print(f"Dataset: {config['data'].get('dataset_name', 'CIFAR100')}")
    print(f"Tasks: {config['data']['num_tasks']} tasks × {config['data']['classes_per_task']} classes/task")
    print("\nNested Learning Features:")
    print("  ✓ Multi-frequency parameter updates (Continuum Memory System)")
    print("  ✓ Fast layers (freq=1): Quick adaptation to new tasks")
    print("  ✓ Medium layers (freq=10): Balance adaptation and stability")
    print("  ✓ Slow layers (freq=100): Long-term knowledge preservation")
    print("\nRealistic Streaming Features:")
    print("  ✓ Single-pass data streaming (no epoch repeating)")
    print("  ✓ Blurry boundaries (future task data appears early)")
    print("  ✓ Other task interference (past task data mixed in)")
    print("  ✓ OOD noise injection (synthetic noise)")
    print("  ✓ Old tasks evaluated on test set only")
    print("="*70)

    # Download and prepare dataset with task splits
    print("\n[Step 1] Checking and preparing dataset...")
    
    dataset_name = config['data'].get('dataset_name', 'CIFAR100')
    data_dir = config['data']['data_dir']
    train_tasks = None
    test_tasks = None
    
    if dataset_name == 'ImageNet256':
        # Use ImageNet 256x256 from Kaggle
        try:
            # Check if dataset already exists
            if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
                print(f"✓ ImageNet dataset found at {data_dir}")
                data_path = data_dir
            else:
                print(f"ImageNet dataset not found. Downloading...")
                data_path = download_imagenet_256(data_dir)
            
            train_tasks, test_tasks = create_imagenet_256_tasks(
                data_path=data_path,
                num_tasks=config['data']['num_tasks'],
                classes_per_task=config['data']['classes_per_task'],
                train_split=config['data'].get('train_split', 0.8)
            )
        except FileNotFoundError as e:
            print(f"\n{e}")
            print("\nPlease set up Kaggle credentials before continuing.")
            return
        except Exception as e:
            print(f"\n✗ Error loading ImageNet: {e}")
            print("\nFalling back to CIFAR100...")
            dataset_name = 'CIFAR100'
    
    if dataset_name == 'CIFAR100' or train_tasks is None:
        # Use CIFAR100 as demo dataset
        # Check if CIFAR100 is already downloaded
        cifar_path = os.path.join(data_dir, 'cifar-100-python')
        if os.path.exists(cifar_path):
            print(f"✓ CIFAR100 dataset found at {data_dir}")
        else:
            print(f"CIFAR100 dataset not found. Downloading...")
        
        train_tasks, test_tasks = create_split_imagenet_tasks(
            data_dir=data_dir,
            num_tasks=config['data']['num_tasks'],
            classes_per_task=config['data']['classes_per_task'],
            use_cifar100=True
        )
    
    # Ensure tasks were created
    if train_tasks is None or test_tasks is None:
        raise ValueError("Failed to create task datasets")
    
    print(f"\nCreated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
    
    # Create OOD dataset for noise injection
    print("\n[Step 2] Generating OOD noise dataset...")
    image_size = tuple(config['data'].get('image_size', [32, 32]))
    ood_dataset = create_ood_dataset(
        num_samples=2000,  # Pool of OOD samples
        image_size=image_size,
        num_channels=3,
        noise_type='mixed',  # Mix of different noise types
        num_classes=config['model']['num_classes']
    )
    print(f"Generated {len(ood_dataset)} OOD noise samples")

    # Initialize model
    print("\n[Step 3] Initializing Nested Learning model...")
    
    # Determine whether to use new NL architecture
    use_nested_learning = config['model'].get('use_nested_learning', True)
    num_cms_levels = config['model'].get('num_cms_levels', 3)  # 3 levels: fast/medium/slow
    
    if use_nested_learning:
        print(f"Using Nested Learning architecture with {num_cms_levels} CMS levels")
        model = NestedLearningNetwork(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes'],
            base_channels=config['model'].get('base_channels', 64),
            num_cms_levels=num_cms_levels
        ).to(device)
        
        # Initialize optimizer with multi-frequency updates
        print("\n[Step 4] Creating Nested Optimizer...")
        optimizer = create_nested_optimizer(
            model,
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
        
        # Print optimizer configuration
        print("\nNested Optimizer Configuration:")
        param_groups = model.get_nested_param_groups()
        level_info = {}
        for group in param_groups:
            freq = group['frequency']
            if freq not in level_info:
                level_info[freq] = []
            level_info[freq].append(group['layer_name'])
        
        for freq in sorted(level_info.keys()):
            layers = level_info[freq]
            print(f"  Frequency {freq:3d}: {', '.join(layers)}")
        
    else:
        print("Using legacy CNN architecture")
        model = OldNestedNetwork(
            input_channels=config['model']['input_channels'],
            num_classes=config['model']['num_classes']
        ).to(device)
        
        # Standard Adam optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.0001)
        )
    
    # Note: OOD samples have label -1, handled in continual_learner by filtering
    criterion = nn.CrossEntropyLoss()

    # Initialize continual learner
    print("\n[Step 5] Initializing continual learner...")
    learner = ContinualLearner(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=str(device),
        memory_size=config['continual_learning'].get('memory_size', 2000),
        log_optimizer_stats_freq=config['training'].get('log_optimizer_stats_freq', 100)
    )

    # Train on tasks sequentially
    print("\n[Step 6] Starting continual learning...")
    num_tasks = config['data']['num_tasks']
    blur_ratio = config['continual_learning'].get('blur_ratio', 0.1)
    other_task_ratio = config['continual_learning'].get('other_task_ratio', 0.05)
    ood_ratio = config['continual_learning'].get('ood_ratio', 0.05)
    
    for task_id in range(num_tasks):
        print(f"\n{'='*70}")
        print(f"TASK {task_id}/{num_tasks-1}")
        print(f"{'='*70}")
        
        # Train on current task with realistic streaming
        learner.train_on_task(
            task_dataset=train_tasks[task_id],
            test_dataset=test_tasks[task_id],
            all_task_datasets=train_tasks,  # All tasks for future blurring
            ood_dataset=ood_dataset,
            batch_size=config['data']['batch_size'],
            task_name="Task",
            evaluate_old_tasks=True,
            blur_ratio=blur_ratio,
            other_task_ratio=other_task_ratio,
            ood_ratio=ood_ratio
        )
        
        # Move to next task
        if task_id < num_tasks - 1:
            learner.update_task(task_name="Task")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    metrics = learner.get_metrics()
    metrics.print_summary()
    
    # Save model if configured
    if config.get('evaluation', {}).get('save_best_model', False):
        model_path = config['evaluation'].get('model_save_path', './models/best_model.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 Model saved to: {model_path}")
    
    print("\n✅ Continual learning completed!")

if __name__ == "__main__":
    main()