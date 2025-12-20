# src/models/__init__.py

from .nested_network import NestedNetwork as LegacyNestedNetwork
from .nested_learning_network import (
    ContinuumMemoryBlock,
    NestedLearningBlock,
    NestedLearningNetwork,
    NestedNetwork  # Alias for backward compatibility
)
from .base_model import BaseModel

__all__ = [
    'BaseModel',
    'LegacyNestedNetwork',
    'ContinuumMemoryBlock',
    'NestedLearningBlock', 
    'NestedLearningNetwork',
    'NestedNetwork'
]
