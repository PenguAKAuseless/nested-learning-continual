"""
Experiment Runner for Continual Learning

Provides a unified interface to run offline continual learning experiments
with proper error handling, checkpointing, and result tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import time
import traceback
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

from continual_learning.rivalry_strategies import RivalryStrategy
from continual_learning.metrics import (
    compute_average_accuracy,
    compute_forgetting,
    compute_forward_transfer,
    compute_backward_transfer,
    create_accuracy_matrix,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for experiment runs"""
    # Model settings
    method_name: str = "ViT-Nested"
    model_size: str = "tiny"  # tiny, small, base, large
    num_classes: int = 10
    
    # Training settings
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Strategy settings
    strategy_type: str = "naive"  # naive, ewc, lwf, gem, packnet, si
    strategy_params: Dict[str, Any] = None
    
    # Data settings
    dataset_name: str = "split_cifar10"
    num_tasks: int = 5
    
    # System settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    
    # Experiment settings
    save_dir: str = "results"
    experiment_name: str = "experiment_1"
    log_interval: int = 50
    save_checkpoints: bool = True
    evaluate_every_epoch: bool = False  # If True, evaluate on all tasks every epoch
    
    def __post_init__(self):
        if self.strategy_params is None:
            self.strategy_params = {}


class ExperimentRunner:
    """
    Main runner for continual learning experiments.
    Handles training, evaluation, error recovery, and result saving.
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup directories
        self.save_dir = Path(config.save_dir) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.save_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.plot_dir = self.save_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.results = {
            'config': asdict(config),
            'accuracy_matrix': [],  # [num_tasks, num_tasks] matrix
            'task_accuracies': [],  # List of accuracies after each task
            'losses': [],  # Training losses
            'training_times': [],  # Time per task
            'metrics': {},  # Final metrics
            'errors': [],  # Any errors encountered
        }
        
        # Save config
        with open(self.save_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)
            
        logger.info(f"Experiment initialized: {config.experiment_name}")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Device: {self.device}")
    
    def run(
        self,
        model: nn.Module,
        strategy: RivalryStrategy,
        task_loaders: List[Tuple[DataLoader, DataLoader]],
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict:
        """
        Run the complete continual learning experiment.
        
        Args:
            model: Neural network model
            strategy: Continual learning strategy
            task_loaders: List of (train_loader, test_loader) for each task
            optimizer: Optional optimizer (created if not provided)
            
        Returns:
            Dictionary with all results
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Starting experiment: {self.config.method_name}")
            logger.info(f"Strategy: {self.config.strategy_type}")
            logger.info(f"Number of tasks: {len(task_loaders)}")
            logger.info(f"Epochs per task: {self.config.num_epochs}")
            logger.info(f"{'='*70}\n")
            
            # Move model to device
            model = model.to(self.device)
            
            # Create optimizer if not provided
            if optimizer is None:
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            
            num_tasks = len(task_loaders)
            accuracy_matrix = np.zeros((num_tasks, num_tasks))
            
            # Train on each task sequentially
            for task_id in range(num_tasks):
                try:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Task {task_id + 1}/{num_tasks}")
                    logger.info(f"{'='*50}")
                    
                    train_loader, test_loader = task_loaders[task_id]
                    
                    # Notify strategy of new task
                    strategy.before_task(task_id)
                    
                    # Train on current task
                    task_start_time = time.time()
                    losses = self._train_task(
                        model, strategy, train_loader, optimizer, task_id
                    )
                    task_time = time.time() - task_start_time
                    
                    self.results['losses'].append(losses)
                    self.results['training_times'].append(task_time)
                    
                    logger.info(f"Task {task_id + 1} training completed in {task_time:.2f}s")
                    
                    # Post-task processing (e.g., compute Fisher, store exemplars)
                    try:
                        strategy.after_task(train_loader)
                    except Exception as e:
                        logger.warning(f"Error in after_task: {e}")
                        self.results['errors'].append({
                            'task': task_id,
                            'phase': 'after_task',
                            'error': str(e)
                        })
                    
                    # Evaluate on all tasks seen so far
                    task_accs = self._evaluate_all_tasks(
                        model, task_loaders[:task_id + 1], task_id
                    )
                    
                    # Update accuracy matrix
                    for eval_task_id, acc in enumerate(task_accs):
                        accuracy_matrix[task_id, eval_task_id] = acc
                    
                    self.results['task_accuracies'].append(task_accs)
                    
                    # Log current performance
                    logger.info(f"Task {task_id + 1} accuracies: {[f'{a:.2f}' for a in task_accs]}")
                    
                    # Save checkpoint
                    if self.config.save_checkpoints:
                        self._save_checkpoint(model, optimizer, strategy, task_id)
                    
                    # Save intermediate results and plots
                    self.results['accuracy_matrix'] = accuracy_matrix.tolist()
                    self._save_results()
                    self._plot_results(accuracy_matrix[:task_id + 1, :task_id + 1], task_id)
                    
                except Exception as e:
                    logger.error(f"Error in task {task_id}: {e}")
                    logger.error(traceback.format_exc())
                    self.results['errors'].append({
                        'task': task_id,
                        'phase': 'training',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    # Continue to next task despite error
                    continue
            
            # Compute final metrics
            self._compute_metrics(accuracy_matrix)
            
            # Save final results
            self._save_results()
            
            # Generate final plots
            self._plot_results(accuracy_matrix, num_tasks - 1, final=True)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Results saved to: {self.save_dir}")
            logger.info(f"{'='*70}\n")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Fatal error in experiment: {e}")
            logger.error(traceback.format_exc())
            self.results['errors'].append({
                'phase': 'experiment',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            self._save_results()
            raise
    
    def _train_task(
        self,
        model: nn.Module,
        strategy: RivalryStrategy,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        task_id: int,
    ) -> List[float]:
        """Train on a single task"""
        model.train()
        epoch_losses = []
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(
                train_loader,
                desc=f"Task {task_id + 1} - Epoch {epoch + 1}/{self.config.num_epochs}",
                leave=False
            )
            
            for batch_idx, (x, y) in enumerate(pbar):
                try:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    
                    # Perform training step through strategy
                    loss = strategy.train_step(x, y, optimizer)
                    
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Update progress bar
                    if batch_idx % self.config.log_interval == 0:
                        pbar.set_postfix({'loss': f'{loss:.4f}'})
                        
                except Exception as e:
                    logger.warning(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            epoch_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Loss: {avg_epoch_loss:.4f}")
        
        return epoch_losses
    
    def _evaluate_all_tasks(
        self,
        model: nn.Module,
        task_loaders: List[Tuple[DataLoader, DataLoader]],
        current_task_id: int,
    ) -> List[float]:
        """Evaluate on all tasks seen so far"""
        model.eval()
        task_accuracies = []
        
        with torch.no_grad():
            for task_id, (_, test_loader) in enumerate(task_loaders):
                correct = 0
                total = 0
                
                for x, y in test_loader:
                    try:
                        x = x.to(self.device, non_blocking=True)
                        y = y.to(self.device, non_blocking=True)
                        
                        outputs = model(x)
                        _, predicted = outputs.max(1)
                        
                        total += y.size(0)
                        correct += predicted.eq(y).sum().item()
                        
                    except Exception as e:
                        logger.warning(f"Error evaluating batch: {e}")
                        continue
                
                accuracy = 100.0 * correct / max(total, 1)
                task_accuracies.append(accuracy)
                
                task_label = "Current" if task_id == current_task_id else f"Task {task_id + 1}"
                logger.info(f"  {task_label} accuracy: {accuracy:.2f}%")
        
        return task_accuracies
    
    def _compute_metrics(self, accuracy_matrix: np.ndarray):
        """Compute continual learning metrics"""
        try:
            avg_acc = compute_average_accuracy(accuracy_matrix)
            forgetting = compute_forgetting(accuracy_matrix)
            forward_transfer = compute_forward_transfer(accuracy_matrix)
            backward_transfer = compute_backward_transfer(accuracy_matrix)
            
            self.results['metrics'] = {
                'average_accuracy': float(avg_acc),
                'forgetting': float(forgetting),
                'forward_transfer': float(forward_transfer),
                'backward_transfer': float(backward_transfer),
            }
            
            logger.info("\n" + "="*50)
            logger.info("Final Metrics:")
            logger.info(f"  Average Accuracy: {avg_acc:.2f}%")
            logger.info(f"  Forgetting: {forgetting:.2f}%")
            logger.info(f"  Forward Transfer: {forward_transfer:.2f}%")
            logger.info(f"  Backward Transfer: {backward_transfer:.2f}%")
            logger.info("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            self.results['errors'].append({
                'phase': 'metrics',
                'error': str(e)
            })
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        strategy: RivalryStrategy,
        task_id: int,
    ):
        """Save model checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / f"task_{task_id}_checkpoint.pt"
            
            checkpoint = {
                'task_id': task_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': asdict(self.config),
            }
            
            # Try to save strategy state if available
            if hasattr(strategy, 'state_dict'):
                checkpoint['strategy_state_dict'] = strategy.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _save_results(self):
        """Save results to JSON"""
        try:
            results_path = self.save_dir / "results.json"
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _plot_results(self, accuracy_matrix: np.ndarray, current_task: int, final: bool = False):
        """Generate and save plots"""
        try:
            suffix = "final" if final else f"task_{current_task}"
            
            # 1. Accuracy Matrix Heatmap
            self._plot_accuracy_matrix(accuracy_matrix, suffix)
            
            # 2. Task Accuracy Evolution
            self._plot_task_evolution(suffix)
            
            # 3. Loss Curves
            self._plot_losses(suffix)
            
            logger.info(f"Plots saved to: {self.plot_dir}")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            logger.error(traceback.format_exc())
    
    def _plot_accuracy_matrix(self, accuracy_matrix: np.ndarray, suffix: str):
        """Plot accuracy matrix heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            accuracy_matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'},
            xticklabels=[f'T{i+1}' for i in range(accuracy_matrix.shape[1])],
            yticklabels=[f'T{i+1}' for i in range(accuracy_matrix.shape[0])],
        )
        plt.xlabel('Evaluated Task')
        plt.ylabel('Training Progress (after task)')
        plt.title(f'{self.config.method_name} - Accuracy Matrix\n{self.config.strategy_type.upper()} Strategy')
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"accuracy_matrix_{suffix}.png", dpi=300)
        plt.close()
    
    def _plot_task_evolution(self, suffix: str):
        """Plot how task accuracies evolve during training"""
        if not self.results['task_accuracies']:
            return
        
        task_accs = np.array(self.results['task_accuracies'])
        num_evaluations, num_tasks = task_accs.shape
        
        plt.figure(figsize=(12, 6))
        
        for task_id in range(num_tasks):
            # Get accuracies for this task across evaluations
            accs = task_accs[task_id:, task_id]  # Only plot after task is learned
            x = list(range(task_id, num_evaluations))
            plt.plot(x, accs, marker='o', label=f'Task {task_id + 1}', linewidth=2)
        
        plt.xlabel('Training Progress (tasks completed)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title(f'{self.config.method_name} - Task Accuracy Evolution\n{self.config.strategy_type.upper()} Strategy', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"task_evolution_{suffix}.png", dpi=300)
        plt.close()
    
    def _plot_losses(self, suffix: str):
        """Plot training losses"""
        if not self.results['losses']:
            return
        
        plt.figure(figsize=(12, 6))
        
        for task_id, task_losses in enumerate(self.results['losses']):
            epochs = list(range(1, len(task_losses) + 1))
            plt.plot(epochs, task_losses, marker='o', label=f'Task {task_id + 1}', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.config.method_name} - Training Loss per Task\n{self.config.strategy_type.upper()} Strategy', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"losses_{suffix}.png", dpi=300)
        plt.close()


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
    """Load a saved checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('task_id', -1)
