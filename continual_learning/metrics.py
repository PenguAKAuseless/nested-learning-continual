"""
Continual Learning Metrics

Standard metrics for evaluating continual learning performance:
- Average Accuracy
- Forgetting Measure
- Forward Transfer
- Backward Transfer
"""

import torch
import numpy as np
from typing import List, Dict


def compute_average_accuracy(accuracy_matrix: np.ndarray) -> float:
    """
    Compute average accuracy across all tasks after training on all tasks.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix where entry (i,j) is 
                        accuracy on task j after training on task i
    
    Returns:
        Average accuracy
    """
    num_tasks = accuracy_matrix.shape[0]
    # Last row contains final accuracies
    return accuracy_matrix[-1, :].mean()


def compute_forgetting(accuracy_matrix: np.ndarray) -> float:
    """
    Compute forgetting measure (Chaudhry et al., 2018).
    
    Forgetting = average of maximum accuracy drop for each task
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
    
    Returns:
        Average forgetting measure
    """
    num_tasks = accuracy_matrix.shape[0]
    forgetting = []
    
    for j in range(num_tasks - 1):  # Exclude last task (can't forget yet)
        # Maximum accuracy achieved on task j
        max_acc = accuracy_matrix[j:, j].max()
        # Final accuracy on task j
        final_acc = accuracy_matrix[-1, j]
        # Forgetting for task j
        forgetting.append(max_acc - final_acc)
    
    return np.mean(forgetting) if forgetting else 0.0


def compute_forward_transfer(accuracy_matrix: np.ndarray, random_baseline: np.ndarray = None) -> float:
    """
    Compute forward transfer (Lopez-Paz & Ranzato, 2017).
    
    Forward transfer measures how training on previous tasks helps with new tasks.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
        random_baseline: [num_tasks] array of random baseline accuracies
    
    Returns:
        Average forward transfer
    """
    num_tasks = accuracy_matrix.shape[0]
    
    if random_baseline is None:
        # Assume uniform random guessing
        random_baseline = np.zeros(num_tasks)
    
    forward_transfer = []
    
    for j in range(1, num_tasks):  # Start from task 1
        # Accuracy on task j before training on it (after task j-1)
        acc_before = accuracy_matrix[j - 1, j]
        # Subtract random baseline
        forward_transfer.append(acc_before - random_baseline[j])
    
    return np.mean(forward_transfer) if forward_transfer else 0.0


def compute_backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """
    Compute backward transfer (Lopez-Paz & Ranzato, 2017).
    
    Backward transfer measures how training on new tasks affects performance on old tasks.
    Positive values indicate positive transfer, negative values indicate forgetting.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
    
    Returns:
        Average backward transfer
    """
    num_tasks = accuracy_matrix.shape[0]
    backward_transfer = []
    
    for j in range(num_tasks - 1):  # Exclude last task
        # Accuracy on task j immediately after training on it
        acc_after_training = accuracy_matrix[j, j]
        # Final accuracy on task j
        final_acc = accuracy_matrix[-1, j]
        # Backward transfer for task j
        backward_transfer.append(final_acc - acc_after_training)
    
    return np.mean(backward_transfer) if backward_transfer else 0.0


def compute_plasticity_stability_tradeoff(accuracy_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute plasticity-stability tradeoff metrics.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
    
    Returns:
        Dictionary with plasticity and stability metrics
    """
    num_tasks = accuracy_matrix.shape[0]
    
    # Plasticity: ability to learn new tasks (diagonal elements)
    plasticity = np.diag(accuracy_matrix).mean()
    
    # Stability: ability to retain old tasks (lower triangular part)
    stability_scores = []
    for i in range(1, num_tasks):
        for j in range(i):
            stability_scores.append(accuracy_matrix[i, j])
    
    stability = np.mean(stability_scores) if stability_scores else 0.0
    
    return {
        'plasticity': plasticity,
        'stability': stability,
        'harmonic_mean': 2 * plasticity * stability / (plasticity + stability + 1e-8)
    }


def compute_learning_curve_area(accuracy_matrix: np.ndarray) -> float:
    """
    Compute area under learning curve.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
    
    Returns:
        Normalized area under learning curve
    """
    num_tasks = accuracy_matrix.shape[0]
    
    # Average accuracy after each task
    avg_accuracies = []
    for i in range(num_tasks):
        # Average over tasks seen so far
        avg_acc = accuracy_matrix[i, :i+1].mean()
        avg_accuracies.append(avg_acc)
    
    # Compute area (trapezoid rule)
    area = np.trapz(avg_accuracies, dx=1.0) / num_tasks
    
    return area


def print_metrics_summary(accuracy_matrix: np.ndarray, method_name: str = "Method"):
    """
    Print comprehensive metrics summary.
    
    Args:
        accuracy_matrix: [num_tasks, num_tasks] matrix
        method_name: Name of the method for display
    """
    print(f"\n{'='*60}")
    print(f"Continual Learning Metrics - {method_name}")
    print(f"{'='*60}")
    
    avg_acc = compute_average_accuracy(accuracy_matrix)
    print(f"Average Accuracy: {avg_acc:.2%}")
    
    forgetting = compute_forgetting(accuracy_matrix)
    print(f"Forgetting: {forgetting:.2%}")
    
    forward = compute_forward_transfer(accuracy_matrix)
    print(f"Forward Transfer: {forward:.2%}")
    
    backward = compute_backward_transfer(accuracy_matrix)
    print(f"Backward Transfer: {backward:.2%}")
    
    ps_tradeoff = compute_plasticity_stability_tradeoff(accuracy_matrix)
    print(f"Plasticity: {ps_tradeoff['plasticity']:.2%}")
    print(f"Stability: {ps_tradeoff['stability']:.2%}")
    print(f"Harmonic Mean: {ps_tradeoff['harmonic_mean']:.2%}")
    
    area = compute_learning_curve_area(accuracy_matrix)
    print(f"Learning Curve Area: {area:.2%}")
    
    print(f"{'='*60}\n")


def create_accuracy_matrix(task_accuracies: List[Dict[int, float]]) -> np.ndarray:
    """
    Create accuracy matrix from list of task accuracy dictionaries.
    
    Args:
        task_accuracies: List where each element is a dict mapping task_id to accuracy
                        after training on task i
    
    Returns:
        Accuracy matrix [num_tasks, num_tasks]
    """
    num_tasks = len(task_accuracies)
    matrix = np.zeros((num_tasks, num_tasks))
    
    for i, acc_dict in enumerate(task_accuracies):
        for task_id, accuracy in acc_dict.items():
            matrix[i, task_id] = accuracy
    
    return matrix
