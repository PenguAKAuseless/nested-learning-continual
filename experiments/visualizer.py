"""
Visualization utilities for continual learning experiments
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import pandas as pd


def plot_accuracy_matrix(accuracy_matrix: np.ndarray, method_name: str = "Method",
                         save_path: Optional[str] = None, figsize: tuple = (8, 6)):
    """
    Plot accuracy matrix heatmap.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] accuracy matrix
        method_name: Name of the method
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(accuracy_matrix * 100, annot=True, fmt='.1f', cmap='YlGnBu',
                cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)
    
    plt.xlabel('Task ID')
    plt.ylabel('Training Progress (After Task)')
    plt.title(f'Accuracy Matrix - {method_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_forgetting_curves(results: Dict[str, Dict], save_path: Optional[str] = None,
                           figsize: tuple = (10, 6)):
    """
    Plot forgetting curves for all methods.
    
    Args:
        results: Dictionary mapping method names to their results
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for method_name, metrics in results.items():
        accuracy_matrix = np.array(metrics['accuracy_matrix'])
        num_tasks = accuracy_matrix.shape[0]
        
        # Compute forgetting for each task over time
        forgetting_curves = []
        for task_id in range(num_tasks - 1):
            max_acc = accuracy_matrix[task_id, task_id]
            forgetting = []
            
            for step in range(task_id + 1, num_tasks):
                current_acc = accuracy_matrix[step, task_id]
                forgetting.append((max_acc - current_acc) * 100)
            
            forgetting_curves.append(forgetting)
        
        # Average forgetting over tasks
        max_len = max(len(f) for f in forgetting_curves)
        padded_curves = [f + [f[-1]] * (max_len - len(f)) for f in forgetting_curves]
        avg_forgetting = np.mean(padded_curves, axis=0)
        
        plt.plot(range(1, len(avg_forgetting) + 1), avg_forgetting, 
                marker='o', label=method_name, linewidth=2)
    
    plt.xlabel('Tasks After Initial Training')
    plt.ylabel('Average Forgetting (%)')
    plt.title('Forgetting Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_comparison(results: Dict[str, Dict], metrics: List[str] = None,
                   save_path: Optional[str] = None, figsize: tuple = (12, 5)):
    """
    Plot comparison bar charts for multiple metrics.
    
    Args:
        results: Dictionary mapping method names to their results
        metrics: List of metric names to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    if metrics is None:
        metrics = ['average_accuracy', 'forgetting', 'forward_transfer', 'backward_transfer']
    
    method_names = list(results.keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    if num_metrics == 1:
        axes = [axes]
    
    metric_labels = {
        'average_accuracy': 'Average Accuracy',
        'forgetting': 'Forgetting',
        'forward_transfer': 'Forward Transfer',
        'backward_transfer': 'Backward Transfer',
    }
    
    colors = sns.color_palette('husl', len(method_names))
    
    for idx, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) * 100 for name in method_names]
        
        axes[idx].bar(range(len(method_names)), values, color=colors, alpha=0.8)
        axes[idx].set_xticks(range(len(method_names)))
        axes[idx].set_xticklabels(method_names, rotation=45, ha='right')
        axes[idx].set_ylabel('Percentage (%)')
        axes[idx].set_title(metric_labels.get(metric, metric))
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_task_accuracy_progression(accuracy_matrix: np.ndarray, method_name: str = "Method",
                                   save_path: Optional[str] = None, figsize: tuple = (10, 6)):
    """
    Plot how accuracy on each task evolves over training.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] accuracy matrix
        method_name: Name of the method
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    num_tasks = accuracy_matrix.shape[0]
    colors = plt.cm.viridis(np.linspace(0, 1, num_tasks))
    
    for task_id in range(num_tasks):
        # Plot accuracy on task_id as training progresses
        accuracies = accuracy_matrix[task_id:, task_id] * 100
        x = range(task_id, num_tasks)
        
        plt.plot(x, accuracies, marker='o', label=f'Task {task_id}',
                color=colors[task_id], linewidth=2)
    
    plt.xlabel('Training Progress (After Task)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Task Accuracy Progression - {method_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_learning_curves(training_history: Dict[str, List[float]],
                         save_path: Optional[str] = None, figsize: tuple = (12, 4)):
    """
    Plot training curves (loss, accuracy over time).
    
    Args:
        training_history: Dictionary with 'loss', 'train_acc', 'val_acc' lists
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss curve
    if 'loss' in training_history:
        axes[0].plot(training_history['loss'], linewidth=2)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_acc' in training_history:
        axes[1].plot(training_history['train_acc'], label='Train', linewidth=2)
    if 'val_acc' in training_history:
        axes[1].plot(training_history['val_acc'], label='Validation', linewidth=2)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def create_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create pandas DataFrame summarizing all results.
    
    Args:
        results: Dictionary mapping method names to their results
        
    Returns:
        DataFrame with summary statistics
    """
    data = []
    
    for method_name, metrics in results.items():
        row = {
            'Method': method_name,
            'Avg Accuracy (%)': metrics.get('average_accuracy', 0) * 100,
            'Forgetting (%)': metrics.get('forgetting', 0) * 100,
            'Forward Transfer (%)': metrics.get('forward_transfer', 0) * 100,
            'Backward Transfer (%)': metrics.get('backward_transfer', 0) * 100,
            'Total Time (s)': metrics.get('total_time', 0),
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('Avg Accuracy (%)', ascending=False)
    
    return df
