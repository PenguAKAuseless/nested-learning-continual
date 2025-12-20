"""
Method Comparator - Framework for comparing continual learning algorithms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import time

from continual_learning.rivalry_strategies import RivalryStrategy
from continual_learning.metrics import (
    compute_average_accuracy,
    compute_forgetting,
    compute_forward_transfer,
    compute_backward_transfer,
    print_metrics_summary,
    create_accuracy_matrix,
)


class MethodComparator:
    """
    Compare multiple continual learning methods on same benchmark.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def evaluate_method(
        self,
        method_name: str,
        model_fn: Callable,
        strategy_fn: Callable,
        task_loaders: List[Tuple[DataLoader, DataLoader]],
        num_epochs: int = 10,
        lr: float = 0.001,
        log_interval: int = 100,
    ) -> Dict:
        """
        Evaluate a single method on continual learning benchmark.
        
        Args:
            method_name: Name of the method
            model_fn: Function that returns a fresh model instance
            strategy_fn: Function that takes model and returns strategy instance
            task_loaders: List of (train_loader, test_loader) tuples for each task
            num_epochs: Number of epochs per task
            lr: Learning rate
            log_interval: Logging frequency
            
        Returns:
            Dictionary with accuracy matrix and metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*70}\n")
        
        num_tasks = len(task_loaders)
        
        # Initialize model and strategy
        model = model_fn().to(self.device)
        strategy = strategy_fn(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Track accuracies
        task_accuracies = []
        training_times = []
        
        # Train on each task sequentially
        for task_id in range(num_tasks):
            print(f"\n--- Task {task_id + 1}/{num_tasks} ---")
            
            train_loader, _ = task_loaders[task_id]
            strategy.before_task(task_id)
            
            # Training
            start_time = time.time()
            
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0.0
                num_batches = 0
                
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
                for batch_idx, (x, y) in enumerate(pbar):
                    x, y = x.to(self.device), y.to(self.device)
                    
                    loss = strategy.train_step(x, y, optimizer)
                    total_loss += loss
                    num_batches += 1
                    
                    if batch_idx % log_interval == 0:
                        pbar.set_postfix({'loss': f'{loss:.4f}'})
                
                avg_loss = total_loss / num_batches
                print(f"  Avg Loss: {avg_loss:.4f}")
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            print(f"  Training Time: {training_time:.2f}s")
            
            # Post-task processing
            strategy.after_task(train_loader)
            
            # Evaluate on all tasks seen so far
            accuracies = {}
            for eval_task_id in range(task_id + 1):
                _, test_loader = task_loaders[eval_task_id]
                acc = self._evaluate(model, test_loader)
                accuracies[eval_task_id] = acc
                print(f"  Task {eval_task_id} Acc: {acc:.2%}")
            
            task_accuracies.append(accuracies)
        
        # Create accuracy matrix
        accuracy_matrix = create_accuracy_matrix(task_accuracies)
        
        # Compute metrics
        metrics = {
            'accuracy_matrix': accuracy_matrix.tolist(),
            'average_accuracy': compute_average_accuracy(accuracy_matrix),
            'forgetting': compute_forgetting(accuracy_matrix),
            'forward_transfer': compute_forward_transfer(accuracy_matrix),
            'backward_transfer': compute_backward_transfer(accuracy_matrix),
            'training_times': training_times,
            'total_time': sum(training_times),
        }
        
        # Print summary
        print_metrics_summary(accuracy_matrix, method_name)
        
        # Store results
        self.results[method_name] = metrics
        
        return metrics
    
    def _evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        """Evaluate model on test set with progress bar"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
                x, y = x.to(self.device), y.to(self.device)
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                output = model(x)
                _, predicted = output.max(1)
                
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        return correct / total
    
    def save_results(self, save_path: str):
        """Save all results to JSON file"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {save_path}")
    
    def get_summary_table(self) -> str:
        """Generate summary comparison table"""
        if not self.results:
            return "No results available"
        
        table = "\n" + "="*80 + "\n"
        table += f"{'Method':<30} {'Avg Acc':<12} {'Forgetting':<12} {'FWT':<12} {'BWT':<12}\n"
        table += "="*80 + "\n"
        
        for method_name, metrics in self.results.items():
            table += f"{method_name:<30} "
            table += f"{metrics['average_accuracy']:<12.2%} "
            table += f"{metrics['forgetting']:<12.2%} "
            table += f"{metrics['forward_transfer']:<12.2%} "
            table += f"{metrics['backward_transfer']:<12.2%}\n"
        
        table += "="*80 + "\n"
        
        return table


class BenchmarkSuite:
    """
    Run comprehensive benchmark suite comparing multiple methods.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparator = MethodComparator()
        
    def run_benchmark(
        self,
        benchmark_name: str,
        methods: Dict[str, Dict],  # method_name -> {model_fn, strategy_fn, config}
        task_loaders: List[Tuple[DataLoader, DataLoader]],
        default_config: Optional[Dict] = None,
    ):
        """
        Run benchmark comparing multiple methods.
        
        Args:
            benchmark_name: Name of the benchmark
            methods: Dictionary mapping method names to their configurations
            task_loaders: Task data loaders
            default_config: Default training configuration
        """
        print(f"\n{'#'*80}")
        print(f"# Benchmark: {benchmark_name}")
        print(f"# Number of tasks: {len(task_loaders)}")
        print(f"# Methods to compare: {len(methods)}")
        print(f"{'#'*80}\n")
        
        if default_config is None:
            default_config = {
                'num_epochs': 10,
                'lr': 0.001,
                'log_interval': 100,
            }
        
        # Evaluate each method
        for method_name, method_config in methods.items():
            config = {**default_config, **method_config.get('config', {})}
            
            try:
                self.comparator.evaluate_method(
                    method_name=method_name,
                    model_fn=method_config['model_fn'],
                    strategy_fn=method_config['strategy_fn'],
                    task_loaders=task_loaders,
                    **config
                )
            except Exception as e:
                print(f"\nError evaluating {method_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results
        results_path = self.output_dir / f"{benchmark_name}_results.json"
        self.comparator.save_results(str(results_path))
        
        # Print summary
        print("\n" + self.comparator.get_summary_table())
        
        return self.comparator.results
