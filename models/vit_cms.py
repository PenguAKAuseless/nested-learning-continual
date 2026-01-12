"""
Vision Transformer with CMS (Continuum Memory System)

This module loads a pretrained ViT from timm and replaces all MLP blocks
with CMS modules for continual learning capabilities.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional
from .cms import CMS


def replace_mlp_with_cms(model, num_levels=3, k=2, verbose=False):
    """
    Recursively replaces all MLP layers in a timm ViT with CMS layers.
    
    Args:
        model: The model to modify (typically a timm ViT)
        num_levels: Number of nested levels in each CMS module
        k: Speed multiplier - level i updates every k^i steps
        verbose: Print replacement details
        
    Returns:
        Modified model with CMS layers
    """
    def _replace_in_module(module, parent_name=""):
        for name, child in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name
            
            # Check if this is an MLP module (timm uses different class names)
            if child.__class__.__name__ in ['Mlp', 'MlpBlock', 'FeedForward']:
                if verbose:
                    print(f"Replacing MLP at: {full_name}")
                
                # Extract dimensions from the existing MLP
                in_features = child.fc1.in_features
                hidden_features = child.fc1.out_features
                out_features = child.fc2.out_features
                
                # Get dropout rate
                drop_rate = 0.0
                if hasattr(child, 'drop'):
                    drop_rate = child.drop.p if hasattr(child.drop, 'p') else 0.0
                elif hasattr(child, 'drop1'):
                    drop_rate = child.drop1.p if hasattr(child.drop1, 'p') else 0.0
                
                # Create CMS replacement
                cms_layer = CMS(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    out_features=out_features,
                    drop=drop_rate,
                    num_levels=num_levels,
                    k=k
                )
                
                # Replace the module
                setattr(module, name, cms_layer)
            else:
                # Recursively process child modules
                _replace_in_module(child, full_name)
    
    _replace_in_module(model)
    return model


class ViT_CMS(nn.Module):
    """
    Vision Transformer with Continuum Memory System.
    
    Loads a pretrained ViT from timm and replaces MLP blocks with CMS modules.
    Optionally adds a task-specific head for continual learning.
    
    Args:
        model_name: Name of the timm model (e.g., 'vit_base_patch16_224')
        pretrained: Load pretrained ImageNet weights
        num_classes: Number of output classes (if different from pretrained)
        cms_levels: Number of nested levels in CMS modules
        k: Speed multiplier - level i updates every k^i steps
        freeze_backbone: Whether to freeze the backbone (except CMS modules)
    """
    def __init__(
        self, 
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=None,
        cms_levels=3,
        k=2,
        freeze_backbone=False
    ):
        super().__init__()
        
        # Load pretrained ViT
        print(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Replace MLPs with CMS
        print(f"Replacing MLP blocks with CMS (levels={cms_levels}, k={k})...")
        self.backbone = replace_mlp_with_cms(self.backbone, num_levels=cms_levels, k=k, verbose=False)
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for name, param in self.backbone.named_parameters():
                # Keep CMS parameters trainable
                if 'levels' not in name:
                    param.requires_grad = False
        
        # Create head if num_classes specified
        self.head = None
        if num_classes is not None:
            self.head = nn.Linear(self.feature_dim, num_classes)
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass through backbone and optional head."""
        features = self.backbone(x)
        
        if self.head is not None:
            return self.head(features)
        return features
    
    def set_head(self, num_classes):
        """Set or replace the classification head."""
        self.num_classes = num_classes
        self.head = nn.Linear(self.feature_dim, num_classes)
        
    def get_features(self, x):
        """Extract features without classification head."""
        return self.backbone(x)


class ViT_Simple(nn.Module):
    """
    Standard Vision Transformer with simple linear head.
    Used as baseline for comparison with ViT_CMS.
    
    Args:
        model_name: Name of the timm model
        pretrained: Load pretrained ImageNet weights
        num_classes: Number of output classes
        head_layers: Number of linear layers in head (2 or 3)
        hidden_dim: Hidden dimension for multi-layer head
        freeze_backbone: Whether to freeze the backbone
    """
    def __init__(
        self,
        model_name='vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        head_layers=2,
        hidden_dim=512,
        freeze_backbone=False
    ):
        super().__init__()
        
        # Load pretrained ViT
        print(f"Loading {model_name} (pretrained={pretrained})...")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Create head
        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            raise ValueError(f"head_layers must be 2 or 3, got {head_layers}")
        
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass through backbone and head."""
        features = self.backbone(x)
        return self.head(features)
    
    def get_features(self, x):
        """Extract features without classification head."""
        return self.backbone(x)
    
    def set_head(self, num_classes, head_layers=2, hidden_dim=512):
        """Set or replace the classification head."""
        self.num_classes = num_classes
        
        if head_layers == 2:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_layers == 3:
            self.head = nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)
            )
