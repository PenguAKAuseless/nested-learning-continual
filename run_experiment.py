"""
Quick experiment launcher for continual learning experiments.

Usage:
    python run_experiment.py --method vit_nested --strategy naive --dataset split_cifar10
    python run_experiment.py --method vit_nested --strategy ewc --dataset split_cifar10 --epochs 10
    python run_experiment.py --method vit_nested --strategy lwf --dataset split_mnist --num_tasks 5
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys

from model.vision_transformer_nested_learning import ViTNestedLearning, ViTNestedConfig
from continual_learning.rivalry_strategies import (
    NaiveStrategy, EWCStrategy, LwFStrategy, GEMStrategy, 
    PackNetStrategy, SynapticIntelligence
)
from data.datasets import (
    SplitCIFAR10, SplitCIFAR100, SplitMNIST, PermutedMNIST, RotatedMNIST,
    ImageNet256, SplitImageFolder
)
from data.stream_loaders import OfflineStreamLoader
from utils.runner import ExperimentRunner, RunConfig
from utils.helpers import set_seed


def get_model(model_size: str, num_classes: int, img_size: int = 32) -> nn.Module:
    """Create model based on configuration"""
    
    # ViT-Nested configurations
    configs = {
        'tiny': ViTNestedConfig(
            img_size=img_size, patch_size=4, num_classes=num_classes,
            dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
        ),
        'small': ViTNestedConfig(
            img_size=img_size, patch_size=4, num_classes=num_classes,
            dim=384, depth=8, num_heads=6, mlp_ratio=4.0,
        ),
        'base': ViTNestedConfig(
            img_size=img_size, patch_size=4, num_classes=num_classes,
            dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        ),
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size]
    return ViTNestedLearning(config)


def get_strategy(strategy_type: str, model: nn.Module, device: str, **kwargs):
    """Create continual learning strategy"""
    
    strategies = {
        'naive': lambda: NaiveStrategy(model, device),
        'ewc': lambda: EWCStrategy(
            model, device,
            lambda_ewc=kwargs.get('lambda_ewc', 5000.0),
            fisher_sample_size=kwargs.get('fisher_sample_size', 200)
        ),
        'lwf': lambda: LwFStrategy(
            model, device,
            lambda_lwf=kwargs.get('lambda_lwf', 1.0),
            temperature=kwargs.get('temperature', 2.0)
        ),
        'gem': lambda: GEMStrategy(
            model, device,
            memory_size=kwargs.get('memory_size', 256)
        ),
        'packnet': lambda: PackNetStrategy(
            model, device,
            prune_ratio=kwargs.get('prune_ratio', 0.5)
        ),
        'si': lambda: SynapticIntelligence(
            model, device,
            si_lambda=kwargs.get('si_lambda', 1.0),
            xi=kwargs.get('xi', 1.0)
        ),
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_type}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy_type]()


def get_dataset(dataset_name: str, num_tasks: int, data_dir: str = './data', **kwargs):
    """Load dataset and create task loaders"""
    
    datasets = {
        'split_cifar10': SplitCIFAR10,
        'split_cifar100': SplitCIFAR100,
        'split_mnist': SplitMNIST,
        'permuted_mnist': PermutedMNIST,
        'rotated_mnist': RotatedMNIST,
        'imagenet256': ImageNet256,
        'split_image_folder': SplitImageFolder,
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(datasets.keys())}")
    
    dataset_cls = datasets[dataset_name]
    dataset = dataset_cls(root=data_dir, num_tasks=num_tasks)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Run continual learning experiment')
    
    # Model arguments
    parser.add_argument('--method', type=str, default='vit_nested',
                        help='Method name (currently only vit_nested)')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'],
                        help='Model size')
    
    # Strategy arguments
    parser.add_argument('--strategy', type=str, default='naive',
                        choices=['naive', 'ewc', 'lwf', 'gem', 'packnet', 'si'],
                        help='Continual learning strategy')
    parser.add_argument('--lambda_ewc', type=float, default=50000.0,
                        help='EWC lambda parameter (default: 50000, higher = stronger regularization)')
    parser.add_argument('--lambda_lwf', type=float, default=1.0,
                        help='LwF lambda parameter')
    parser.add_argument('--memory_size', type=int, default=256,
                        help='GEM memory size')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='split_cifar10',
                        choices=['split_cifar10', 'split_cifar100', 'split_mnist', 
                                'permuted_mnist', 'rotated_mnist', 'imagenet256', 
                                'split_image_folder'],
                        help='Dataset to use')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda:0/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (default: auto - 0 on Windows, 4 on Linux/Mac)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision training (faster on modern GPUs)')
    
    # Experiment arguments
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Log interval')
    parser.add_argument('--no_checkpoints', action='store_true',
                        help='Disable checkpoint saving')
    
    args = parser.parse_args()
    
    # Auto-detect num_workers for Windows compatibility
    if args.num_workers is None:
        import platform
        if platform.system() == 'Windows':
            args.num_workers = 0  # Windows has issues with multiprocessing DataLoader
            print("Note: Setting num_workers=0 for Windows compatibility")
        else:
            args.num_workers = 4  # Linux/Mac can use multiple workers
    
    # Set seed
    set_seed(args.seed)
    
    # Auto-generate experiment name
    if args.experiment_name is None:
        # Will be updated after dataset loading to include single_class/group_class
        args.experiment_name = None
    
    print(f"\n{'='*70}")
    print(f"Continual Learning Experiment")
    print(f"{'='*70}")
    print(f"Method: {args.method} ({args.model_size})")
    print(f"Strategy: {args.strategy}")
    print(f"Dataset: {args.dataset} ({args.num_tasks} tasks)")
    print(f"Epochs per task: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Experiment: {args.experiment_name}")
    print(f"{'='*70}\n")
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = get_dataset(args.dataset, args.num_tasks, args.data_dir)
        
        # Determine image size and classes per task
        if 'cifar' in args.dataset:
            img_size = 32
        elif 'mnist' in args.dataset:
            img_size = 28
        else:
            img_size = 224  # ImageNet default
        
        # Get classes_per_task from dataset (default 1)
        classes_per_task = getattr(dataset, 'classes_per_task', 1)
        # CRITICAL: Model needs outputs for ALL classes in the dataset, not just per-task
        # Otherwise with classes_per_task=1, all labels get remapped to 0 (trivial problem)
        if 'cifar10' in args.dataset.lower():
            num_classes = 10  # CIFAR-10 has 10 total classes
        elif 'cifar100' in args.dataset.lower():
            num_classes = 100  # CIFAR-100 has 100 total classes
        elif 'mnist' in args.dataset.lower():
            num_classes = 10  # MNIST has 10 total classes
        else:
            # For custom datasets, use total classes if available
            num_classes = getattr(dataset, 'num_classes', classes_per_task * args.num_tasks)
        
        # Determine class mode for experiment name
        class_mode = 'single_class' if classes_per_task == 1 else f'group_class_{classes_per_task}'
        
        # Generate experiment name if not provided
        if args.experiment_name is None:
            args.experiment_name = f"{class_mode}_{args.model_size}_{args.strategy}_{args.dataset}_tasks{args.num_tasks}_seed{args.seed}"
        
        print(f"Dataset: {args.dataset}")
        print(f"Number of tasks: {args.num_tasks}")
        print(f"Classes per task: {classes_per_task} ({class_mode})")
        print(f"Model output size: {num_classes}")
        print(f"Experiment name: {args.experiment_name}")
        
        # Create task loaders
        print("Creating task loaders...")
        task_loaders = []
        for task_id in range(args.num_tasks):
            train_dataset = dataset.get_task(task_id, train=True)
            test_dataset = dataset.get_task(task_id, train=False)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True if 'cuda' in args.device else False,
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True if 'cuda' in args.device else False,
            )
            
            task_loaders.append((train_loader, test_loader))
        
        print(f"Created {len(task_loaders)} task loaders")
        
        # Create model
        print(f"Creating {args.model_size} model...")
        model = get_model(args.model_size, num_classes, img_size)
        
        # Move model to device immediately
        device = torch.device(args.device)
        model = model.to(device)
        
        # Enable GPU optimizations if using CUDA
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
            torch.backends.cudnn.deterministic = False  # Allows non-deterministic ops for speed
            print(f"GPU optimizations enabled on {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {trainable_params:,} (total: {total_params:,})")
        
        # Create strategy
        print(f"Creating {args.strategy} strategy...")
        strategy = get_strategy(
            args.strategy, model, args.device,
            lambda_ewc=args.lambda_ewc,
            lambda_lwf=args.lambda_lwf,
            memory_size=args.memory_size,
        )
        
        # Create run configuration
        config = RunConfig(
            method_name=args.method,
            model_size=args.model_size,
            num_classes=num_classes,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            strategy_type=args.strategy,
            dataset_name=args.dataset,
            num_tasks=args.num_tasks,
            device=args.device,
            seed=args.seed,
            num_workers=args.num_workers,
            save_dir=args.save_dir,
            experiment_name=args.experiment_name,
            log_interval=args.log_interval,
            save_checkpoints=not args.no_checkpoints,
            use_amp=args.amp,
        )
        
        # Create runner and run experiment
        print("\nInitializing experiment runner...")
        runner = ExperimentRunner(config)
        
        print("\nStarting training...\n")
        results = runner.run(model, strategy, task_loaders)
        
        # Print final summary
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        if 'metrics' in results:
            metrics = results['metrics']
            print("\nFinal Metrics:")
            print(f"  Average Accuracy: {metrics.get('average_accuracy', 0):.2f}%")
            print(f"  Forgetting: {metrics.get('forgetting', 0):.2f}%")
            print(f"  Forward Transfer: {metrics.get('forward_transfer', 0):.2f}%")
            print(f"  Backward Transfer: {metrics.get('backward_transfer', 0):.2f}%")
        
        print(f"\nResults saved to: {runner.save_dir}")
        print(f"  - Plots: {runner.plot_dir}")
        print(f"  - Checkpoints: {runner.checkpoint_dir}")
        print(f"  - Results JSON: {runner.save_dir / 'results.json'}")
        
        if results.get('errors'):
            print(f"\nWarning: {len(results['errors'])} errors occurred during execution")
            print("Check results.json for details")
        
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
