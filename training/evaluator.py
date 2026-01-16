"""
Evaluation Pipeline for Continual Learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm
import numpy as np


class Evaluator:
    """
    Evaluator for continual learning experiments.
    
    Args:
        model: The model to evaluate
        device: Device to use (cuda/cpu)
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    @torch.no_grad()
    def evaluate_task(
        self,
        test_loader: DataLoader,
        task_id: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a single task.
        
        Args:
            test_loader: DataLoader for the task
            task_id: Task identifier
            verbose: Print progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(test_loader, desc=f"Evaluating Task {task_id}") if verbose else test_loader
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Predictions
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / labels.size(0):.2f}%'
                })
        
        # Calculate metrics
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * total_correct / total_samples
        
        # Calculate per-class metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Precision, Recall, F1 for multi-class classification (macro-averaged)
        from sklearn.metrics import precision_recall_fscore_support
        precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        # For backwards compatibility, compute simple averages
        precision = precision_scores
        recall = recall_scores
        f1 = f1_scores
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100
        }
    
    @torch.no_grad()
    def evaluate_all_tasks(
        self,
        test_loaders: List[DataLoader],
        verbose: bool = True
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate on all tasks.
        
        Args:
            test_loaders: List of test dataloaders
            verbose: Print progress
            
        Returns:
            Dictionary mapping task_id to metrics
        """
        results = {}
        
        for task_id, test_loader in enumerate(test_loaders):
            metrics = self.evaluate_task(test_loader, task_id, verbose=verbose)
            results[task_id] = metrics
            
            if verbose:
                print(f"Task {task_id}: Acc={metrics['accuracy']:.2f}%, "
                      f"F1={metrics['f1']:.2f}%")
        
        # Calculate average metrics
        avg_metrics = {
            'avg_accuracy': np.mean([m['accuracy'] for m in results.values()]),
            'avg_f1': np.mean([m['f1'] for m in results.values()]),
            'avg_loss': np.mean([m['loss'] for m in results.values()])
        }
        
        if verbose:
            print(f"\nAverage: Acc={avg_metrics['avg_accuracy']:.2f}%, "
                  f"F1={avg_metrics['avg_f1']:.2f}%")
        
        results['average'] = avg_metrics
        
        return results
    
    @torch.no_grad()
    def calculate_forgetting(
        self,
        test_loaders: List[DataLoader],
        baseline_accuracies: Dict[int, float]
    ) -> Dict[str, float]:
        """
        Calculate forgetting metrics.
        
        Args:
            test_loaders: List of test dataloaders
            baseline_accuracies: Accuracy after training each task (before training next)
            
        Returns:
            Dictionary with forgetting metrics
        """
        current_results = self.evaluate_all_tasks(test_loaders, verbose=False)
        
        forgetting = {}
        for task_id in baseline_accuracies:
            if task_id in current_results:
                forgetting[task_id] = baseline_accuracies[task_id] - current_results[task_id]['accuracy']
        
        avg_forgetting = np.mean(list(forgetting.values())) if forgetting else 0.0
        
        return {
            'per_task_forgetting': forgetting,
            'average_forgetting': avg_forgetting
        }
