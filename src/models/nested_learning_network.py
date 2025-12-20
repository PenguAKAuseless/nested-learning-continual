"""
Nested Learning Network for Continual Learning (Based on Google's NL Paper)

This implementation incorporates key concepts from Nested Learning:
1. Multi-frequency update hierarchy (Continuum Memory System)
2. Deep optimizers with associative memory
3. Self-referential learning with nested optimization levels

Key differences from standard deep learning:
- Components update at different frequencies (fast → slow hierarchy)
- Optimization is nested across multiple levels
- Memory operates on a continuum (not binary short/long-term)

Citations refer to the Nested Learning paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple


class ContinuumMemoryBlock(nn.Module):
    """
    Continuum Memory System (CMS) block that updates at a specific frequency.
    
    In NL, components are organized by update frequency:
    - High frequency (fast): Updates every step, handles immediate context
    - Low frequency (slow): Updates rarely, integrates over long cycles
    
    This is inspired by brain waves coordinating activity at multiple time scales.
    [cite: 38, 39, 40, 291, 292]
    """
    def __init__(self, channels: int, frequency: int, level_id: int, 
                 use_conv: bool = True, kernel_size: int = 3):
        """
        Args:
            channels: Number of channels (features)
            frequency: Update frequency (1=every step, 10=every 10 steps, etc.)
            level_id: Level in the hierarchy (0=fastest, higher=slower)
            use_conv: Whether to use Conv layers (True) or Linear (False)
            kernel_size: Kernel size for conv layers
        """
        super().__init__()
        self.frequency = frequency
        self.level_id = level_id
        self.use_conv = use_conv
        self.step_counter = 0
        
        # Memory block structure: MLP/Conv with residual connections
        # [cite: 83, 291, 295]
        if use_conv:
            self.memory_net = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(channels),  # BatchNorm for conv layers
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
            )
        else:
            self.memory_net = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.LayerNorm(channels * 2),
                nn.GELU(),
                nn.Linear(channels * 2, channels)
            )
        
        # Residual scaling for stability [cite: 83]
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # Gradient accumulator for nested optimization
        self.register_buffer('grad_accumulator', None)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform [cite: implementation]"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        Residual connections are standard in NL memory blocks [cite: 83]
        """
        identity = x
        out = self.memory_net(x)
        # In-place operation to save memory
        return identity.add_(self.residual_scale * out)
    
    def should_update(self, global_step: int) -> bool:
        """Check if this block should update at current global step"""
        return (global_step % self.frequency) == 0
    
    def get_param_group(self) -> Dict[str, Any]:
        """
        Return parameters with frequency metadata for nested optimizer.
        [cite: 295]
        """
        return {
            'params': list(self.parameters()),
            'frequency': self.frequency,
            'level_id': self.level_id,
            'layer_name': f"CMS_Level_{self.level_id}_Freq_{self.frequency}"
        }


class NestedLearningBlock(nn.Module):
    """
    A complete nested learning block combining multiple frequency levels.
    
    Implements the multi-level hierarchy where:
    - Level 0 (Fast): Updates every step
    - Level 1 (Medium): Updates every 10 steps  
    - Level 2 (Slow): Updates every 100 steps
    
    This creates a spectrum of memory operating at different time scales.
    [cite: 16, 217, 223, 291]
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 num_levels: int = 3, base_frequency: int = 1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_levels: Number of nested levels (default 3)
            base_frequency: Base update frequency (multiplied by 10^level)
        """
        super().__init__()
        self.num_levels = num_levels
        
        # Projection to match dimensions
        self.input_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Create Continuum Memory System with decreasing frequencies
        # Level 0: freq=1 (fastest), Level 1: freq=10, Level 2: freq=100 (slowest)
        self.cms_layers = nn.ModuleList()
        for level in range(num_levels):
            freq = base_frequency * (10 ** level)  # 1, 10, 100
            cms_block = ContinuumMemoryBlock(
                channels=out_channels,
                frequency=freq,
                level_id=level,
                use_conv=True
            )
            self.cms_layers.append(cms_block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through nested levels.
        Earlier layers update quickly, later layers integrate over longer cycles.
        [cite: 38, 40]
        """
        x = self.input_proj(x)
        
        # Pass through each CMS level sequentially
        for cms_layer in self.cms_layers:
            x = cms_layer(x)
        
        return x
    
    def get_nested_param_groups(self) -> List[Dict[str, Any]]:
        """Collect parameter groups for nested optimizer"""
        groups = []
        
        # Input projection is fastest (Level 0)
        input_proj_params = list(self.input_proj.parameters()) if isinstance(self.input_proj, nn.Conv2d) else []
        if input_proj_params:
            groups.append({
                'params': input_proj_params,
                'frequency': 1,
                'level_id': -1,
                'layer_name': 'InputProjection'
            })
        
        # Add CMS layers
        for cms_layer in self.cms_layers:
            if isinstance(cms_layer, ContinuumMemoryBlock):
                cms_group = cms_layer.get_param_group()
                # Ensure params is a list, not a generator
                params_list = list(cms_group.get('params', []))
                if params_list:  # Only add non-empty param groups
                    cms_group['params'] = params_list
                    groups.append(cms_group)
        
        return groups


class NestedLearningNetwork(nn.Module):
    """
    Complete Nested Learning Network for Continual Learning.
    
    Key Features:
    1. Multi-level frequency hierarchy (CMS)
    2. Components update at different rates
    3. Designed for continual learning scenarios
    
    Architecture Philosophy:
    - What appears as "depth" is actually a nested optimization process [cite: 111]
    - Different levels process information at varying temporal granularities
    - Enables continual learning without catastrophic forgetting
    
    [cite: 16, 60, 65, 304]
    """
    def __init__(self, input_channels: int = 3, num_classes: int = 100, 
                 base_channels: int = 64, num_cms_levels: int = 3):
        """
        Args:
            input_channels: Input image channels (3 for RGB)
            num_classes: Number of output classes
            base_channels: Base number of channels
            num_cms_levels: Number of CMS levels (3 recommended: fast/medium/slow)
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_cms_levels = num_cms_levels
        
        # Initial stem - processes raw input
        # This updates at highest frequency (every step)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),  # BatchNorm for conv layers
            nn.GELU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Nested Learning Blocks with CMS
        # Each block has multiple frequency levels
        self.nl_block1 = NestedLearningBlock(
            base_channels, base_channels, num_levels=num_cms_levels, base_frequency=1
        )
        
        self.nl_block2 = NestedLearningBlock(
            base_channels, base_channels * 2, num_levels=num_cms_levels, base_frequency=1
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.nl_block3 = NestedLearningBlock(
            base_channels * 2, base_channels * 4, num_levels=num_cms_levels, base_frequency=1
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.nl_block4 = NestedLearningBlock(
            base_channels * 4, base_channels * 8, num_levels=num_cms_levels, base_frequency=1
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global pooling and classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier with CMS for continual learning
        # Fast adaptation to new tasks, slow preservation of old knowledge
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_channels * 8, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Track global training step for frequency-based updates
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following NL paper guidelines"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through nested learning hierarchy.
        
        Information flows through multiple frequency levels,
        enabling both fast adaptation and slow integration.
        [cite: 40, 304]
        """
        # Stem processing
        x = self.stem(x)
        
        # Nested learning blocks with gradient checkpointing for memory efficiency
        # Only use checkpointing during training to save memory
        if self.training and x.requires_grad:
            # Gradient checkpointing: trade compute for memory
            from torch.utils.checkpoint import checkpoint
            x = checkpoint(self.nl_block1, x, use_reentrant=False)
            x = checkpoint(self.nl_block2, x, use_reentrant=False)
            x = self.pool2(x)
            
            x = checkpoint(self.nl_block3, x, use_reentrant=False)
            x = self.pool3(x)
            
            x = checkpoint(self.nl_block4, x, use_reentrant=False)
            x = self.pool4(x)
        else:
            # Standard forward during inference
            x = self.nl_block1(x)
            x = self.nl_block2(x)
            x = self.pool2(x)
            
            x = self.nl_block3(x)
            x = self.pool3(x)
            
            x = self.nl_block4(x)
            x = self.pool4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for continual learning analysis.
        
        Returns features before classification head,
        useful for task-specific adaptation.
        """
        x = self.stem(x)
        x = self.nl_block1(x)
        x = self.nl_block2(x)
        x = self.pool2(x)
        x = self.nl_block3(x)
        x = self.pool3(x)
        x = self.nl_block4(x)
        x = self.pool4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def get_nested_param_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups organized by update frequency for nested optimizer.
        
        This is crucial for NL: different parameters update at different rates.
        [cite: 213, 217, 295]
        
        Returns:
            List of parameter groups with frequency metadata
        """
        groups = []
        
        # Stem: Highest frequency (updates every step)
        groups.append({
            'params': self.stem.parameters(),
            'frequency': 1,
            'level_id': 0,
            'layer_name': 'Stem'
        })
        
        # Nested Learning Blocks
        for i, block in enumerate([self.nl_block1, self.nl_block2, self.nl_block3, self.nl_block4]):
            block_groups = block.get_nested_param_groups()
            for group in block_groups:
                # Ensure params is a list, not a generator
                params_list = list(group.get('params', []))
                if params_list:  # Only add non-empty param groups
                    group['params'] = params_list
                    group['layer_name'] = f"Block{i+1}_{group['layer_name']}"
                    groups.append(group)
        
        # Classifier: Medium frequency (balance adaptation and stability)
        groups.append({
            'params': list(self.classifier.parameters()),
            'frequency': 10,
            'level_id': 1,
            'layer_name': 'Classifier'
        })
        
        return groups
    
    def increment_step(self):
        """Increment global training step counter"""
        self.global_step += 1
    
    def get_step(self) -> int:
        """Get current global training step"""
        return int(self.global_step)


# Alias for backward compatibility
NestedNetwork = NestedLearningNetwork
