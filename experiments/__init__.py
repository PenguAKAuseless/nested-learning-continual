"""
Experiment Utilities for Continual Learning
Comparison framework for evaluating different algorithms.
"""

from .comparator import MethodComparator, BenchmarkSuite
from .visualizer import plot_accuracy_matrix, plot_forgetting_curves, plot_comparison
from .logger import ExperimentLogger

__all__ = [
    'MethodComparator',
    'BenchmarkSuite',
    'plot_accuracy_matrix',
    'plot_forgetting_curves',
    'plot_comparison',
    'ExperimentLogger',
]
