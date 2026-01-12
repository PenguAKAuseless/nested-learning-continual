"""
Continual Learning Experiment Runner

Usage:
    python run_experiment.py --model vit_cms --dataset cifar10 --num_tasks 5 --epochs 10
"""

import torch
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from models import ViT_CMS, ViT_Simple, CNN_Replay
from datasets import get_cifar10_task_loaders, get_dataset_info
from training import Trainer, Evaluator


def get_model(model_name, num_classes=2, device='cuda', **kwargs):
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model ('vit_cms', 'vit_simple', 'cnn_replay')
        num_classes: Number of output classes (2 for binary task-as-class)
        device: Device to use
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    print(f"\n{'='*60}")
    print(f"Initializing Model: {model_name}")
    print(f"{'='*60}")
    
    if model_name == 'vit_cms':
        model = ViT_CMS(
            model_name='vit_base_patch16_224',
            pretrained=kwargs.get('pretrained', True),
            num_classes=num_classes,
            cms_levels=kwargs.get('cms_levels', 3),
            k=kwargs.get('k', 2),
            freeze_backbone=kwargs.get('freeze_backbone', False)
        )
    elif model_name == 'vit_simple':
        model = ViT_Simple(
            model_name='vit_base_patch16_224',
            pretrained=kwargs.get('pretrained', True),
            num_classes=num_classes,
            head_layers=kwargs.get('head_layers', 2),
            hidden_dim=kwargs.get('hidden_dim', 512),
            freeze_backbone=kwargs.get('freeze_backbone', False)
        )
    elif model_name == 'cnn_replay':
        model = CNN_Replay(
            num_classes=num_classes,
            buffer_size=kwargs.get('buffer_size', 1000),
            hidden_dim=kwargs.get('cnn_hidden_dim', 64)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model initialized successfully!")
    return model


def run_experiment(args):
    """
    Run a continual learning experiment.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    print(f"\nDataset: {dataset_info['name']}")
    print(f"Task setup: {dataset_info['task_setup']}")
    print(f"Number of tasks: {args.num_tasks}")
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loaders, test_loaders = get_cifar10_task_loaders(
        data_root=args.data_root,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance=True,
        image_size=224  # ViT input size
    )
    
    # Initialize model
    model_kwargs = {
        'pretrained': args.pretrained,
        'cms_levels': args.cms_levels,
        'k': args.k,
        'freeze_backbone': args.freeze_backbone,
        'head_layers': args.head_layers,
        'hidden_dim': args.hidden_dim,
        'buffer_size': args.buffer_size,
        'cnn_hidden_dim': args.cnn_hidden_dim
    }
    model = get_model(args.model, num_classes=2, device=device, **model_kwargs)
    
    if device == 'cuda':
        print(f"\nGPU Memory after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    # Initialize trainer and evaluator
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_replay=(args.model == 'cnn_replay'),
        replay_batch_size=args.replay_batch_size
    )
    
    evaluator = Evaluator(model=model, device=device)
    
    # Track results
    results = {
        'config': vars(args),
        'dataset_info': dataset_info,
        'task_results': {},
        'baseline_accuracies': {},
        'final_evaluation': {}
    }
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting Continual Learning Training")
    print(f"{'='*60}\n")
    
    for task_id in range(args.num_tasks):
        print(f"\n{'='*60}")
        print(f"Training on Task {task_id}")
        print(f"{'='*60}\n")
        
        # Train on current task
        train_metrics = trainer.train_task(
            train_loaders[task_id],
            task_id=task_id,
            epochs=args.epochs,
            verbose=True
        )
        
        # Evaluate on current task
        print(f"\nEvaluating Task {task_id} after training...")
        eval_metrics = evaluator.evaluate_task(
            test_loaders[task_id],
            task_id=task_id,
            verbose=True
        )
        
        # Store baseline accuracy
        results['baseline_accuracies'][task_id] = eval_metrics['accuracy']
        
        # Evaluate on all seen tasks so far
        if task_id > 0:
            print(f"\nEvaluating all tasks up to Task {task_id}...")
            all_task_metrics = evaluator.evaluate_all_tasks(
                test_loaders[:task_id+1],
                verbose=True
            )
        else:
            all_task_metrics = {0: eval_metrics, 'average': eval_metrics}
        
        # Store results
        results['task_results'][task_id] = {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
            'all_tasks_eval': all_task_metrics
        }
        
        print(f"\nTask {task_id} Summary:")
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Acc: {eval_metrics['accuracy']:.2f}%")
        print(f"  Test F1: {eval_metrics['f1']:.2f}%")
        if task_id > 0:
            print(f"  Avg Acc (all tasks): {all_task_metrics['average']['avg_accuracy']:.2f}%")
    
    # Final evaluation on all tasks
    print(f"\n{'='*60}")
    print("Final Evaluation on All Tasks")
    print(f"{'='*60}\n")
    
    final_results = evaluator.evaluate_all_tasks(test_loaders, verbose=True)
    results['final_evaluation'] = final_results
    
    # Calculate forgetting
    forgetting_metrics = evaluator.calculate_forgetting(
        test_loaders,
        results['baseline_accuracies']
    )
    results['forgetting_metrics'] = forgetting_metrics
    
    print(f"\nForgetting Analysis:")
    print(f"  Average Forgetting: {forgetting_metrics['average_forgetting']:.2f}%")
    for task_id, forget in forgetting_metrics['per_task_forgetting'].items():
        print(f"  Task {task_id}: {forget:.2f}%")
    
    # Save results
    save_results(results, args)
    
    print(f"\n{'='*60}")
    print("Experiment Completed!")
    print(f"{'='*60}\n")
    
    return results


def save_results(results, args):
    """Save experiment results to disk."""
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.model}_{args.dataset}_tasks{args.num_tasks}_{timestamp}"
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results as JSON
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Save config
    config_file = results_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return results_dir


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Experiments')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='vit_cms',
                       choices=['vit_cms', 'vit_simple', 'cnn_replay'],
                       help='Model to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--cms_levels', type=int, default=3,
                       help='Number of CMS levels (for vit_cms)')
    parser.add_argument('--k', type=int, default=2,
                       help='Speed multiplier: level i updates every k^i steps (for vit_cms)')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights')
    parser.add_argument('--head_layers', type=int, default=2,
                       help='Number of head layers (for vit_simple)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension for head')
    parser.add_argument('--buffer_size', type=int, default=1000,
                       help='Replay buffer size (for cnn_replay)')
    parser.add_argument('--cnn_hidden_dim', type=int, default=64,
                       help='CNN hidden dimension (for cnn_replay)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for data')
    parser.add_argument('--num_tasks', type=int, default=5,
                       help='Number of tasks to train on')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--replay_batch_size', type=int, default=16,
                       help='Replay batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loader workers')
    
    # System arguments
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args)


if __name__ == '__main__':
    main()
