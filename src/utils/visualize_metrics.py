"""
Visualization utilities for continual learning metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def plot_task_accuracies(metrics_summary: Dict, save_path: str = None):
    """
    Plot accuracy for each task over continual learning process.
    
    Parameters:
    - metrics_summary: Dictionary from ContinualLearningMetrics.get_summary()
    - save_path: Optional path to save the figure
    """
    all_task_accs = metrics_summary['all_task_accuracies']
    
    plt.figure(figsize=(12, 6))
    
    # Plot each task's accuracy trajectory
    for task_id in sorted(all_task_accs.keys()):
        accuracies = all_task_accs[task_id]
        x = list(range(len(accuracies)))
        plt.plot(x, accuracies, marker='o', label=f'Task {task_id}', linewidth=2)
    
    plt.xlabel('Evaluation Point', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Task Accuracies During Continual Learning', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_forgetting(metrics_summary: Dict, save_path: str = None):
    """
    Plot forgetting measure for each old task.
    
    Parameters:
    - metrics_summary: Dictionary from ContinualLearningMetrics.get_summary()
    - save_path: Optional path to save the figure
    """
    forgetting = metrics_summary.get('forgetting_per_task', {})
    
    if not forgetting:
        print("No forgetting data available (only one task completed)")
        return
    
    plt.figure(figsize=(10, 6))
    
    tasks = sorted(forgetting.keys())
    forget_values = [forgetting[t] for t in tasks]
    
    colors = ['red' if f > 5 else 'orange' if f > 2 else 'green' for f in forget_values]
    
    plt.bar(tasks, forget_values, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Forgetting (%)', fontsize=12)
    plt.title('Forgetting per Task', fontsize=14, fontweight='bold')
    plt.xticks(tasks)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_forget = metrics_summary.get('average_forgetting', 0)
    plt.axhline(y=avg_forget, color='blue', linestyle='--', 
                label=f'Average Forgetting: {avg_forget:.2f}%', linewidth=2)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_old_vs_current_accuracy(metrics_summary: Dict, save_path: str = None):
    """
    Plot comparison of current task vs average old task accuracy.
    
    Parameters:
    - metrics_summary: Dictionary from ContinualLearningMetrics.get_summary()
    - save_path: Optional path to save the figure
    """
    current_acc = metrics_summary.get('current_task_accuracy', 0)
    avg_old_acc = metrics_summary.get('average_old_task_accuracy', 0)
    
    if avg_old_acc == 0:
        print("No old task data available yet")
        return
    
    plt.figure(figsize=(8, 6))
    
    categories = ['Current Task', 'Avg Old Tasks']
    values = [current_acc, avg_old_acc]
    colors = ['#2ecc71', '#3498db']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Current Task vs Old Tasks Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_all_metrics(metrics_summary: Dict, save_dir: str = None):
    """
    Create all visualizations and optionally save them.
    
    Parameters:
    - metrics_summary: Dictionary from ContinualLearningMetrics.get_summary()
    - save_dir: Optional directory to save all figures
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot 1: Task accuracies over time
    save_path_1 = os.path.join(save_dir, 'task_accuracies.png') if save_dir else None
    plot_task_accuracies(metrics_summary, save_path_1)
    
    # Plot 2: Forgetting
    save_path_2 = os.path.join(save_dir, 'forgetting.png') if save_dir else None
    plot_forgetting(metrics_summary, save_path_2)
    
    # Plot 3: Current vs Old
    save_path_3 = os.path.join(save_dir, 'current_vs_old.png') if save_dir else None
    plot_old_vs_current_accuracy(metrics_summary, save_path_3)
    
    print("\n✅ All visualizations created!")


def create_summary_table(metrics_summary: Dict) -> str:
    """
    Create a formatted text table of all metrics.
    
    Parameters:
    - metrics_summary: Dictionary from ContinualLearningMetrics.get_summary()
    
    Returns:
    - Formatted string table
    """
    lines = []
    lines.append("="*70)
    lines.append("CONTINUAL LEARNING METRICS SUMMARY")
    lines.append("="*70)
    
    lines.append(f"\nCurrent Task: {metrics_summary['current_task']}")
    lines.append(f"Current Task Accuracy: {metrics_summary['current_task_accuracy']:.2f}%")
    
    old_accs = metrics_summary.get('old_tasks_accuracy', {})
    if old_accs:
        lines.append(f"\nOld Tasks Accuracy:")
        for task_id in sorted(old_accs.keys()):
            lines.append(f"  Task {task_id}: {old_accs[task_id]:.2f}%")
        lines.append(f"\nAverage Old Task Accuracy: {metrics_summary['average_old_task_accuracy']:.2f}%")
    
    forgetting = metrics_summary.get('forgetting_per_task', {})
    if forgetting:
        lines.append(f"\nForgetting per Task:")
        for task_id in sorted(forgetting.keys()):
            lines.append(f"  Task {task_id}: {forgetting[task_id]:.2f}%")
        lines.append(f"\nAverage Forgetting: {metrics_summary['average_forgetting']:.2f}%")
    
    lines.append("="*70)
    
    return "\n".join(lines)
