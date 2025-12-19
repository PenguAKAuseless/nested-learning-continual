"""
Continual Learning Module
Implements various rivalry strategies and baselines for continual learning.
"""

from .rivalry_strategies import (
    RivalryStrategy,
    EWCStrategy,
    LwFStrategy,
    GEMStrategy,
    PackNetStrategy,
    SynapticIntelligence,
)
from .metrics import (
    compute_forgetting,
    compute_forward_transfer,
    compute_backward_transfer,
    compute_average_accuracy,
)

__all__ = [
    'RivalryStrategy',
    'EWCStrategy',
    'LwFStrategy',
    'GEMStrategy',
    'PackNetStrategy',
    'SynapticIntelligence',
    'compute_forgetting',
    'compute_forward_transfer',
    'compute_backward_transfer',
    'compute_average_accuracy',
]
