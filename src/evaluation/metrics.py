import torch
import numpy as np
from typing import Dict, List, Optional


def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total

def precision(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum().item()
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

def recall(y_true, y_pred):
    true_positive = ((y_true == 1) & (y_pred == 1)).sum().item()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum().item()
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


class ContinualLearningMetrics:
    """
    Track metrics across multiple tasks in continual learning.
    Includes current task accuracy and old task accuracy tracking.
    """
    def __init__(self):
        self.task_accuracies = {}  # {task_id: accuracy}
        self.task_history = []  # List of dicts with task performance over time
        self.current_task = 0
        
    def update_task_accuracy(self, task_id: int, accuracy: float, epoch: Optional[int] = None):
        """Update accuracy for a specific task"""
        if task_id not in self.task_accuracies:
            self.task_accuracies[task_id] = []
        
        self.task_accuracies[task_id].append({
            'epoch': epoch,
            'accuracy': accuracy,
            'evaluated_at_task': self.current_task
        })
    
    def set_current_task(self, task_id: int):
        """Set the current task being trained"""
        self.current_task = task_id
    
    def get_current_task_accuracy(self) -> float:
        """Get the latest accuracy for current task"""
        if self.current_task in self.task_accuracies and len(self.task_accuracies[self.current_task]) > 0:
            return self.task_accuracies[self.current_task][-1]['accuracy']
        return 0.0
    
    def get_old_tasks_accuracy(self) -> Dict[int, float]:
        """Get latest accuracy for all old tasks"""
        old_accuracies = {}
        for task_id in range(self.current_task):
            if task_id in self.task_accuracies and len(self.task_accuracies[task_id]) > 0:
                old_accuracies[task_id] = self.task_accuracies[task_id][-1]['accuracy']
        return old_accuracies
    
    def get_average_old_task_accuracy(self) -> float:
        """Get average accuracy across all old tasks"""
        old_accs = self.get_old_tasks_accuracy()
        if not old_accs:
            return 0.0
        return float(np.mean(list(old_accs.values())))
    
    def get_forgetting_measure(self) -> Dict[int, float]:
        """
        Calculate forgetting for each old task.
        Forgetting = max_accuracy - current_accuracy
        """
        forgetting = {}
        for task_id in range(self.current_task):
            if task_id in self.task_accuracies and len(self.task_accuracies[task_id]) > 0:
                accuracies = [record['accuracy'] for record in self.task_accuracies[task_id]]
                max_acc = max(accuracies)
                current_acc = accuracies[-1]
                forgetting[task_id] = max_acc - current_acc
        return forgetting
    
    def get_average_forgetting(self) -> float:
        """Get average forgetting across all old tasks"""
        forgetting = self.get_forgetting_measure()
        if not forgetting:
            return 0.0
        return float(np.mean(list(forgetting.values())))
    
    def get_summary(self) -> Dict:
        """Get a summary of all metrics"""
        return {
            'current_task': self.current_task,
            'current_task_accuracy': self.get_current_task_accuracy(),
            'old_tasks_accuracy': self.get_old_tasks_accuracy(),
            'average_old_task_accuracy': self.get_average_old_task_accuracy(),
            'forgetting_per_task': self.get_forgetting_measure(),
            'average_forgetting': self.get_average_forgetting(),
            'all_task_accuracies': {k: [r['accuracy'] for r in v] 
                                   for k, v in self.task_accuracies.items()}
        }
    
    def print_summary(self):
        """Print a formatted summary of metrics"""
        summary = self.get_summary()
        print("\n" + "="*60)
        print(f"CONTINUAL LEARNING METRICS SUMMARY (Current Task: {summary['current_task']})")
        print("="*60)
        print(f"Current Task Accuracy: {summary['current_task_accuracy']:.2f}%")
        
        if summary['old_tasks_accuracy']:
            print(f"\nOld Tasks Accuracy:")
            for task_id, acc in sorted(summary['old_tasks_accuracy'].items()):
                print(f"  Task {task_id}: {acc:.2f}%")
            print(f"\nAverage Old Task Accuracy: {summary['average_old_task_accuracy']:.2f}%")
        
        if summary['forgetting_per_task']:
            print(f"\nForgetting per Task:")
            for task_id, forget in sorted(summary['forgetting_per_task'].items()):
                print(f"  Task {task_id}: {forget:.2f}%")
            print(f"\nAverage Forgetting: {summary['average_forgetting']:.2f}%")
        print("="*60 + "\n")