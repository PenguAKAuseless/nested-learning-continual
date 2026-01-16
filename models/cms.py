"""
Continuum Memory System (CMS) - Nested Learning Implementation
Based on Google's Nested Learning paper.

CMS replaces standard MLP blocks with a hierarchy of nested MLPs,
enabling continual learning by utilizing different processing speeds.
"""

import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """
    Standard MLP Block with Layer Normalization.
    Architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CMS(nn.Module):
    """
    Continuum Memory System (CMS)
    
    Replaces a single MLP with a hierarchy of nested MLPs operating at different speeds.
    Each level processes information with residual connections for stable training.
    
    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension for the original MLP (will be distributed across levels)
        out_features: Output dimension (typically same as in_features for ViT)
        drop: Dropout rate
        num_levels: Number of nested MLP levels (default: 3)
        k: Speed multiplier - level i updates every k^i steps (default: 2)
        
    The hidden dimension is distributed across levels to maintain similar parameter count
    to the original MLP while enabling nested processing.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., num_levels=3, k=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.num_levels = num_levels
        self.in_features = in_features
        self.out_features = out_features
        self.k = k  # Speed multiplier for nested levels
        self.step_counter = 0
        
        # Distribute hidden dimension across levels
        # Each level gets a portion of the total capacity
        level_hidden = hidden_features // num_levels
        
        # Create nested MLP levels
        self.levels = nn.ModuleList()
        for i in range(num_levels):
            # Each level: input_dim -> level_hidden -> input_dim
            # This maintains the residual stream dimension
            self.levels.append(
                MlpBlock(in_features, level_hidden, in_features, drop)
            )
        
        # Final projection to output dimension if different from input
        if out_features != in_features:
            self.output_proj = nn.Linear(in_features, out_features)
        else:
            self.output_proj = None

    def forward(self, x):
        """
        Forward pass through nested MLP levels.
        Level i updates every k^i steps for different processing speeds.
        Each level applies: x = x + MLP(x) (residual connection)
        """
        self.step_counter += 1
        
        # Apply each nested level with speed-based gating
        for i, level in enumerate(self.levels):
            # Level i updates every k^i steps
            if self.step_counter % (self.k ** i) == 0:
                x = x + level(x)
        
        # Apply output projection if needed
        if self.output_proj is not None:
            x = self.output_proj(x)
            
        return x

    def __repr__(self):
        return (f"CMS(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"num_levels={self.num_levels}, "
                f"k={self.k})")