"""
Continual Learning Experiment Runner

Usage:
    python run_experiment.py --model vit_cms --dataset cifar10 --num_tasks 5 --epochs 10 --backbone vit_base_patch16_224
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import random
import torch
import torch.nn as nn
import argparse
import json
import hashlib
from datetime import datetime
from pathlib import Path

from models import ViT_CMS, ViT_Simple, ViT_Replay, CNN_Replay
from my_datasets.task_as_class import get_cifar10_task_loaders, get_dataset_info
from training import Trainer, Evaluator
from training.cms_optim import CMSOptimizerWrapper

def get_config_hash(config_dict):
    """Generate hash from configuration for checkpoint identification."""
    relevant_params = {
        'model': config_dict.get('model'),
        'dataset': config_dict.get('dataset'),
        'num_tasks': config_dict.get('num_tasks'),
        'epochs': config_dict.get('epochs'),
        'learning_rate': config_dict.get('learning_rate'),
        'batch_size': config_dict.get('batch_size'),
        'pretrained': config_dict.get('pretrained'),
        'cms_levels': config_dict.get('cms_levels'),
        'k': config_dict.get('k'),
        'head_layers': config_dict.get('head_layers'),
        'hidden_dim': config_dict.get('hidden_dim'),
        'buffer_size': config_dict.get('buffer_size'),
        'freeze_backbone': config_dict.get('freeze_backbone'),
        'backbone': config_dict.get('backbone')
    }
    config_str = json.dumps(relevant_params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_checkpoint_dir(args):
    """Get checkpoint directory based on config hash."""
    ckpt_dir = Path(args.checkpoint_dir) / f"task_{args.num_tasks}_{args.model}_{args.dataset}"
    return ckpt_dir


def save_checkpoint(model, optimizer, task_id, results, ckpt_dir, args):
    """Save model checkpoint after completing a task."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'task_id': task_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'config': vars(args)
    }
    
    ckpt_path = ckpt_dir / f'checkpoint_task_{task_id}.pt'
    torch.save(checkpoint, ckpt_path)
    
    config_path = ckpt_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n✓ Checkpoint saved: {ckpt_path}")
    return ckpt_path


def load_checkpoint(ckpt_dir, task_id=None):
    """Load checkpoint for a specific task or latest checkpoint."""
    if not ckpt_dir.exists():
        return None
    
    if task_id is not None:
        ckpt_path = ckpt_dir / f'checkpoint_task_{task_id}.pt'
        if not ckpt_path.exists():
            return None
    else:
        checkpoints = sorted(ckpt_dir.glob('checkpoint_task_*.pt'))
        if not checkpoints:
            return None
        ckpt_path = checkpoints[-1]
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)
    return checkpoint


def check_existing_checkpoints(ckpt_dir, num_tasks):
    """Check which tasks have been completed."""
    if not ckpt_dir.exists():
        return -1
    
    completed_tasks = []
    for task_id in range(num_tasks):
        ckpt_path = ckpt_dir / f'checkpoint_task_{task_id}.pt'
        if ckpt_path.exists():
            completed_tasks.append(task_id)
    
    return max(completed_tasks) if completed_tasks else -1


def get_model(model_name, num_classes=10, device='cuda', **kwargs):
    """Create a model instance."""
    # Lấy tên backbone từ kwargs
    backbone = kwargs.get('backbone_name', 'vit_base_patch16_224')
    
    print(f"\n{'='*60}")
    print(f"Initializing Model: {model_name}")
    print(f"Backbone: {backbone}")
    print(f"{'='*60}")
    
    if model_name == 'vit_cms':
        model = ViT_CMS(
            model_name=backbone,
            pretrained=kwargs.get('pretrained', True),
            num_classes=num_classes,
            cms_levels=kwargs.get('cms_levels', 3),
            k=kwargs.get('k', 5)
        )
    elif model_name == 'vit_simple':
        model = ViT_Simple(
            model_name=backbone,
            pretrained=kwargs.get('pretrained', True),
            num_classes=num_classes,
            head_layers=kwargs.get('head_layers', 3),
            hidden_dim=kwargs.get('hidden_dim', 512)
        )
    elif model_name == 'vit_replay':
        model = ViT_Replay(
            model_name=backbone,
            pretrained=kwargs.get('pretrained', True),
            num_classes=num_classes,
            head_layers=kwargs.get('head_layers', 3),
            hidden_dim=kwargs.get('hidden_dim', 512),
            buffer_size=kwargs.get('buffer_size', 1000)
        )
    elif model_name == 'cnn_replay':
        input_size = kwargs.get('input_size', 32)
        model = CNN_Replay(
            num_classes=num_classes,
            buffer_size=kwargs.get('buffer_size', 1000),
            hidden_dim=kwargs.get('cnn_hidden_dim', 64),
            input_size=input_size
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Model initialized successfully!")
    return model


def run_experiment(args):
    """Run a continual learning experiment."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*60}")
    print(f"Random seed set to: {args.seed}")
    print(f"{'='*60}\n")
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    dataset_info = get_dataset_info(args.dataset)
    print(f"\nDataset: {dataset_info['name']}")
    print(f"Task setup: {dataset_info['task_setup']}")
    print(f"Number of tasks: {args.num_tasks}")
    
    print(f"\nCreating dataloaders...")
    image_size = 32 if args.model == 'cnn_replay' else 224
    train_loaders, test_loaders = get_cifar10_task_loaders(
        data_root=args.data_root,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance=True,
        image_size=image_size
    )
    
    model_kwargs = {
        'pretrained': args.pretrained,
        'cms_levels': args.cms_levels,
        'k': args.k,
        'head_layers': args.head_layers,
        'hidden_dim': args.hidden_dim,
        'buffer_size': args.buffer_size,
        'cnn_hidden_dim': args.cnn_hidden_dim,
        'input_size': image_size,
        'backbone_name': args.backbone 
    }
    model = get_model(args.model, num_classes=10, device=device, **model_kwargs)
    
    # Init Head = 0 để tránh nhiễu (Lỗi Acc < 10%)
    print("Initializing Head weights to 0 to prevent random noise dominance...")
    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        nn.init.constant_(model.head.weight, 0)
        nn.init.constant_(model.head.bias, 0)
    elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        nn.init.constant_(model.fc.weight, 0)
        nn.init.constant_(model.fc.bias, 0)
    elif hasattr(model, 'heads'):
        for h in model.heads:
             if isinstance(h, nn.Linear):
                nn.init.constant_(h.weight, 0)
                nn.init.constant_(h.bias, 0)

    model.to(device)
    
    if device == 'cuda':
        print(f"\nGPU Memory after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        use_replay=(args.model in ['cnn_replay', 'vit_replay']),
        replay_batch_size=args.replay_batch_size
    )
    
    evaluator = Evaluator(model=model, device=device)
    
    ckpt_dir = get_checkpoint_dir(args)
    last_completed_task = check_existing_checkpoints(ckpt_dir, args.num_tasks)
    
    results = {
        'config': vars(args),
        'dataset_info': dataset_info,
        'task_results': {},
        'baseline_accuracies': {},
        'per_task_history': {},
        'final_evaluation': {}
    }
    
    start_task = 0
    if last_completed_task >= 0 and not args.no_checkpoint:
        print(f"\n{'='*60}")
        print(f"Found existing checkpoints up to task {last_completed_task}")
        print(f"{'='*60}")
        
        if args.resume:
            checkpoint = load_checkpoint(ckpt_dir, last_completed_task)
            if checkpoint is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                results = checkpoint['results']
                start_task = last_completed_task + 1
                print(f"✓ Resumed from task {last_completed_task}")
                print(f"  Continuing from task {start_task}")
            else:
                print("⚠ Could not load checkpoint, starting from scratch")
        else:
            print("Use --resume to continue from checkpoint")
            print("Starting fresh training...")
    
    print(f"\n{'='*60}")
    if start_task == 0:
        print("Starting Continual Learning Training")
    else:
        print(f"Resuming Continual Learning Training from Task {start_task}")
    print(f"{'='*60}\n")
    
    for task_id in range(start_task, args.num_tasks):
        print(f"\n{'='*60}")
        print(f"Training on Task {task_id}")
        print(f"{'='*60}\n")
        
        train_metrics = trainer.train_task(
            train_loaders[task_id],
            task_id=task_id,
            epochs=args.epochs,
            verbose=True
        )
        
        if task_id == 0:
            print(f"\nEvaluating all tasks (Task 0)...")
        else:
            print(f"\nEvaluating all tasks (0 to {task_id}) to measure forgetting...")
        all_task_metrics = evaluator.evaluate_all_tasks(
            test_loaders[:task_id+1],
            verbose=True
        )
        
        results['baseline_accuracies'][task_id] = all_task_metrics[task_id]['accuracy']
        
        for tid in range(task_id + 1):
            if tid not in results['per_task_history']:
                results['per_task_history'][tid] = []
            results['per_task_history'][tid].append({
                'after_training_task': task_id,
                'accuracy': all_task_metrics[tid]['accuracy'],
                'f1': all_task_metrics[tid]['f1']
            })
        
        results['task_results'][task_id] = {
            'train_metrics': train_metrics,
            'eval_metrics': all_task_metrics[task_id],
            'all_tasks_eval': all_task_metrics
        }
        
        print(f"\nTask {task_id} Summary:")
        print(f"  Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Test Acc: {all_task_metrics[task_id]['accuracy']:.2f}%")
        print(f"  Test F1 (current task): {all_task_metrics[task_id]['f1']:.2f}%")
        print(f"  Avg Acc (all tasks 0-{task_id}): {all_task_metrics['average']['avg_accuracy']:.2f}%")
        if task_id > 0:
            print(f"  Previous tasks performance:")
            for tid in range(task_id):
                print(f"    Task {tid}: {all_task_metrics[tid]['accuracy']:.2f}% (baseline: {results['baseline_accuracies'][tid]:.2f}%)")
        
        if not args.no_checkpoint:
            save_checkpoint(model, trainer.optimizer, task_id, results, ckpt_dir, args)
        
        model.train()
    
    print(f"\n{'='*60}")
    print("Final Evaluation on All Tasks")
    print(f"{'='*60}\n")
    
    final_results = evaluator.evaluate_all_tasks(test_loaders, verbose=True)
    results['final_evaluation'] = final_results
    
    forgetting_metrics = evaluator.calculate_forgetting(
        test_loaders,
        results['baseline_accuracies']
    )
    results['forgetting_metrics'] = forgetting_metrics
    
    print(f"\nForgetting Analysis:")
    print(f"  Average Forgetting: {forgetting_metrics['average_forgetting']:.2f}%")
    for task_id, forget in forgetting_metrics['per_task_forgetting'].items():
        print(f"  Task {task_id}: {forget:.2f}%")
    
    save_results(results, args)
    
    print(f"\n{'='*60}")
    print("Experiment Completed!")
    print(f"{'='*60}\n")
    
    return results


def save_results(results, args):
    """Save experiment results to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"task_{args.num_tasks}_{args.model}_{args.dataset}_{timestamp}"
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    config_file = results_dir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return results_dir


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Experiments')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='vit_cms',
                        choices=['vit_cms', 'vit_simple', 'vit_replay', 'cnn_replay'],
                        help='Model to use')
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224',
                        help='Backbone model name from timm')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--cms_levels', type=int, default=3,
                        help='Number of CMS levels (for vit_cms)')
    parser.add_argument('--k', type=int, default=5,
                        help='Speed multiplier: level i updates every k^i steps (for vit_cms)')
    parser.add_argument('--head_layers', type=int, default=3,
                        help='Number of head layers (for vit_simple and vit_replay)')
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving/loading checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint if available')
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='Disable checkpoint saving/loading')
    
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()