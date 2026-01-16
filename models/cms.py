"""
Continuum Memory System (CMS) - Nested Learning Implementation
Based on "Nested Learning: The Illusion of Deep Learning Architecture"

This module implements:
1. Self-Modifying Layers (Titans): Layers that generate their own learning rates and forget gates.
2. CMS Hierarchy: Nested structure where deeper levels operate at lower frequencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpBlock(nn.Module):
    """
    Standard MLP Block kept for compatibility and baseline comparisons.
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

class SelfModifyingLayer(nn.Module):
    """
    Implements the 'Self-Modifying' mechanism (Titans).
    The layer dynamically generates its own update rules (learning rate, forget gate)
    based on the input context.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim // 2
        
        # Meta-Learners: Generate 'alpha' (forget) and 'eta' (learning rate)
        self.gen_alpha = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() 
        )
        self.gen_eta = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Persistent memory matrix (W_t)
        self.memory = nn.Parameter(torch.randn(dim, dim) * 0.02)

    def forward(self, x):
        b, s, d = x.shape
        outputs = []
        
        # Clone memory for the loop (simulation of online updates)
        curr_mem = self.memory.unsqueeze(0).expand(b, -1, -1)
        
        for t in range(s):
            xt = x[:, t, :]
            
            # 1. Generate dynamic parameters
            alpha = self.gen_alpha(xt).unsqueeze(-1)
            eta = self.gen_eta(xt).unsqueeze(-1)
            
            # 2. Retrieval
            yt = torch.bmm(curr_mem, xt.unsqueeze(-1)).squeeze(-1)
            outputs.append(yt)
            
            # 3. Update Memory (Delta Rule)
            update_val = torch.bmm(yt.unsqueeze(-1), xt.unsqueeze(1))
            curr_mem = alpha * curr_mem - eta * update_val
            
        out = torch.stack(outputs, dim=1)
        return out + x 

class CMS(nn.Module):
    """
    Continuum Memory System (CMS) Main Module.
    Combines Self-Modifying fast weights with a hierarchy of slow weights (MLPs).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., num_levels=3, k=2):
        super().__init__()
        self.in_features = in_features
        self.num_levels = num_levels
        self.k = k
        
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        
        # Level 0: Self-Modifying Layer (Fastest context adaptation)
        self.fast_memory = SelfModifyingLayer(in_features)
        
        # Higher Levels: Nested MLPs (Slow, persistent knowledge)
        self.levels = nn.ModuleList()
        level_dim = hidden_features // num_levels
        
        for i in range(num_levels):
            block = nn.Sequential(
                nn.Linear(in_features, level_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(level_dim, in_features),
                nn.Dropout(drop)
            )
            self.levels.append(block)
            
        self.norm = nn.LayerNorm(in_features)
        if out_features != in_features:
            self.proj = nn.Linear(in_features, out_features)
        else:
            self.proj = None

    def forward(self, x):
        # 1. Fast Adaptation
        x = self.fast_memory(x)
        
        # 2. Slow Consolidation (Nested Levels)
        for block in self.levels:
            x = x + block(x)
            
        x = self.norm(x)
        if self.proj:
            x = self.proj(x)
        return x