"""
Vision Transformer with Nested Learning (HOPE Architecture)
Combines ViT patch-based processing with hierarchical memory and adaptive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class ViTNestedConfig:
    """Configuration for Vision Transformer with Nested Learning"""
    # Image & patch settings
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    
    # Model dimensions
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Nested learning settings
    num_levels: int = 3  # TITAN, CMS_fast, CMS_slow
    titan_mem_size: int = 512
    cms_fast_size: int = 256
    cms_slow_size: int = 128
    
    # Update schedules
    titan_update_period: int = 1
    cms_fast_update_period: int = 8
    cms_slow_update_period: int = 64
    
    # Optimizer settings
    inner_lr: float = 0.01
    teach_scale: float = 0.10
    surprise_threshold: float = 0.0
    
    # Architecture flags
    use_cls_token: bool = True
    dropout: float = 0.1
    drop_path: float = 0.1
    
    # Data loading settings
    num_workers: int = 32


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        
        # [B, C, H, W] -> [B, embed_dim, H/P, W/P] -> [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TITANMemory(nn.Module):
    """TITAN memory module - fast associative storage"""
    
    def __init__(self, dim: int, mem_size: int, update_period: int = 1):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size
        self.update_period = update_period
        
        # Memory slots: [mem_size, dim]
        self.register_buffer('memory', torch.zeros(mem_size, dim))
        self.register_buffer('step_counter', torch.tensor(0))
        
        # Query/Key/Value projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, teach_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] - input tokens
            teach_signal: [B, N, dim] - gradient-based teaching signal
        Returns:
            output: [B, N, dim] - memory-augmented representation
        """
        B, N, D = x.shape
        
        # Compute attention with memory
        q = self.to_q(x)  # [B, N, dim]
        k = self.to_k(self.memory.unsqueeze(0).expand(B, -1, -1))  # [B, mem_size, dim]
        v = self.to_v(self.memory.unsqueeze(0).expand(B, -1, -1))  # [B, mem_size, dim]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)  # [B, N, mem_size]
        
        # Retrieve from memory
        retrieved = torch.matmul(attn, v)  # [B, N, dim]
        output = self.output_proj(retrieved)
        
        # Update memory if teach signal provided and period elapsed
        if teach_signal is not None and self.training:
            self._update_memory(x, teach_signal)
        
        return output
    
    def _update_memory(self, x: torch.Tensor, teach_signal: torch.Tensor):
        """Update memory slots based on teaching signal"""
        if self.step_counter % self.update_period == 0:
            # Average across batch and sequence
            update = (x.detach() + teach_signal.detach()).mean(dim=(0, 1))  # [dim]
            
            # Write to memory slot (simple round-robin)
            slot_idx = (self.step_counter // self.update_period) % self.mem_size
            self.memory[slot_idx] = 0.9 * self.memory[slot_idx] + 0.1 * update
        
        self.step_counter += 1


class CMSMemory(nn.Module):
    """Continual Memory System - slower hierarchical storage"""
    
    def __init__(self, dim: int, mem_size: int, update_period: int, hidden_multiplier: int = 2):
        super().__init__()
        self.dim = dim
        self.mem_size = mem_size
        self.update_period = update_period
        self.hidden_dim = dim * hidden_multiplier
        
        # Memory storage
        self.register_buffer('memory', torch.zeros(mem_size, dim))
        self.register_buffer('step_counter', torch.tensor(0))
        
        # Chunk accumulation buffer
        self.register_buffer('chunk_buffer', torch.zeros(update_period, dim))
        self.register_buffer('chunk_idx', torch.tensor(0))
        
        # Processing network
        self.encoder = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )
        
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        
    def forward(self, x: torch.Tensor, teach_signal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            teach_signal: [B, N, dim]
        Returns:
            output: [B, N, dim]
        """
        B, N, D = x.shape
        
        # Encode input
        encoded = self.encoder(x)
        
        # Attention with memory
        memory_expanded = self.memory.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.attention(encoded, memory_expanded, memory_expanded)
        
        # Update memory
        if teach_signal is not None and self.training:
            self._accumulate_and_update(x, teach_signal)
        
        return attn_out
    
    def _accumulate_and_update(self, x: torch.Tensor, teach_signal: torch.Tensor):
        """Accumulate chunks and update memory periodically (Eq. 31)"""
        # Add to chunk buffer
        chunk_data = (x.detach() + teach_signal.detach()).mean(dim=(0, 1))  # [dim]
        self.chunk_buffer[self.chunk_idx] = chunk_data
        self.chunk_idx += 1
        
        # When buffer is full, update memory
        if self.chunk_idx >= self.update_period:
            # Aggregate chunks
            aggregated = self.chunk_buffer.mean(dim=0)  # [dim]
            
            # Write to memory (round-robin)
            slot_idx = (self.step_counter // self.update_period) % self.mem_size
            self.memory[slot_idx] = 0.95 * self.memory[slot_idx] + 0.05 * aggregated
            
            # Reset buffer
            self.chunk_idx.zero_()
            self.step_counter += self.update_period


class DeepMomentumOptimizer(nn.Module):
    """Inner optimizer with L2-regression update (Eq. 27-29)"""
    
    def __init__(self, dim: int, lr: float = 0.01, beta: float = 0.9):
        super().__init__()
        self.lr = lr
        self.beta = beta
        
        # Momentum buffer
        self.register_buffer('momentum', torch.zeros(dim, dim))
        
        # Projector for L2-regression
        self.projector = nn.Linear(dim, dim, bias=False)
        
    def compute_update(self, x: torch.Tensor, teach_signal: torch.Tensor) -> torch.Tensor:
        """
        Compute parameter update using input-aware L2 regression
        Args:
            x: [B, N, dim] - input activations
            teach_signal: [B, N, dim] - gradient-based teaching signal
        Returns:
            update: [dim, dim] - weight update
        """
        B, N, D = x.shape
        
        # Compute rank-1 preconditioner from input covariance
        x_flat = x.reshape(-1, D)  # [B*N, dim]
        cov = torch.matmul(x_flat.T, x_flat) / (B * N)  # [dim, dim]
        
        # Regularized inverse
        precond = torch.inverse(cov + 1e-4 * torch.eye(D, device=x.device))
        
        # Compute gradient
        grad = teach_signal.mean(dim=(0, 1))  # [dim]
        
        # L2-regression update with momentum
        raw_update = torch.outer(grad, precond @ grad)  # [dim, dim]
        self.momentum = self.beta * self.momentum + (1 - self.beta) * raw_update
        
        return -self.lr * self.momentum


class HOPEBlock(nn.Module):
    """HOPE block: Attention → TITAN → CMS hierarchy"""
    
    def __init__(self, config: ViTNestedConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        
        # Pre-normalization
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.norm3 = nn.LayerNorm(config.dim)
        
        # Self-attention with SDPA
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.attn_proj = nn.Linear(config.dim, config.dim)
        self.attn_drop = nn.Dropout(config.dropout)
        
        # Hierarchical memory
        self.titan = TITANMemory(config.dim, config.titan_mem_size, config.titan_update_period)
        self.cms_fast = CMSMemory(config.dim, config.cms_fast_size, config.cms_fast_update_period)
        self.cms_slow = CMSMemory(config.dim, config.cms_slow_size, config.cms_slow_update_period)
        
        # MLP (feed-forward)
        hidden_dim = int(config.dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.dim),
            nn.Dropout(config.dropout)
        )
        
        # Inner optimizers
        self.titan_optimizer = DeepMomentumOptimizer(config.dim, config.inner_lr)
        self.cms_fast_optimizer = DeepMomentumOptimizer(config.dim, config.inner_lr * 0.5)
        self.cms_slow_optimizer = DeepMomentumOptimizer(config.dim, config.inner_lr * 0.25)
        
        self.teach_scale = config.teach_scale
        self.surprise_threshold = config.surprise_threshold
        
    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention with SDPA"""
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0.0)
        
        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.attn_proj(attn_out)
        out = self.attn_drop(out)
        
        return out
    
    def compute_teach_signal(self, x: torch.Tensor, residual: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute teaching signal from prediction error"""
        if not self.training:
            return None
        
        # Prediction error as teach signal
        error = residual - x
        teach_signal = self.teach_scale * error
        
        # Surprise gating
        if self.surprise_threshold > 0:
            surprise = teach_signal.norm(dim=-1, keepdim=True).mean()
            if surprise < self.surprise_threshold:
                return None
        
        return teach_signal
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] - input tokens
        Returns:
            output: [B, N, dim]
        """
        # Self-attention with residual
        attn_input = self.norm1(x)
        attn_out = self.attention(attn_input)
        x = x + attn_out
        
        # Compute teaching signal
        teach_signal = self.compute_teach_signal(attn_input, x)
        
        # TITAN memory layer
        titan_input = self.norm2(x)
        titan_out = self.titan(titan_input, teach_signal)
        x = x + titan_out
        
        # CMS hierarchy
        cms_fast_out = self.cms_fast(x, teach_signal)
        cms_slow_out = self.cms_slow(x, teach_signal)
        x = x + cms_fast_out + cms_slow_out
        
        # MLP with residual
        mlp_input = self.norm3(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        
        return x


class ViTNestedLearning(nn.Module):
    """Vision Transformer with Nested Learning (HOPE)"""
    
    def __init__(self, config: ViTNestedConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.img_size, config.patch_size, config.in_channels, config.dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.use_cls_token = config.use_cls_token
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Position embedding
        self.num_tokens = num_patches + (1 if config.use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(config.dropout)
        
        # HOPE blocks
        self.blocks = nn.ModuleList([
            HOPEBlock(config) for _ in range(config.depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.dim)
        
        # Classification head
        self.head = nn.Linear(config.dim, config.num_classes)
        
        # Tie embeddings to output head (faithfulness to paper)
        # Note: For ViT, we keep them separate as patch embeddings are convolutional
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] - input images
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, dim]
        
        # Add class token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_tokens, dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply HOPE blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Classification (use CLS token or global average pool)
        if self.use_cls_token:
            x = x[:, 0]  # [B, dim]
        else:
            x = x.mean(dim=1)  # [B, dim]
        
        # Classification head
        logits = self.head(x)
        
        return logits
    
    def enable_memorization(self, enable: bool = True):
        """Enable/disable test-time memorization"""
        for block in self.blocks:
            block.train(enable)
    
    def get_memory_stats(self) -> dict:
        """Get statistics from all memory modules"""
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f'block_{i}'] = {
                'titan_step': block.titan.step_counter.item(),
                'titan_mem_norm': block.titan.memory.norm().item(),
                'cms_fast_step': block.cms_fast.step_counter.item(),
                'cms_fast_mem_norm': block.cms_fast.memory.norm().item(),
                'cms_slow_step': block.cms_slow.step_counter.item(),
                'cms_slow_mem_norm': block.cms_slow.memory.norm().item(),
            }
        return stats


def create_vit_nested_tiny(num_classes: int = 1000) -> ViTNestedLearning:
    """Create a tiny ViT-Nested model for testing"""
    config = ViTNestedConfig(
        img_size=224,
        patch_size=16,
        dim=192,
        depth=12,
        num_heads=3,
        num_classes=num_classes,
        titan_mem_size=128,
        cms_fast_size=64,
        cms_slow_size=32,
    )
    return ViTNestedLearning(config)


def create_vit_nested_small(num_classes: int = 1000) -> ViTNestedLearning:
    """Create a small ViT-Nested model"""
    config = ViTNestedConfig(
        img_size=224,
        patch_size=16,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        titan_mem_size=256,
        cms_fast_size=128,
        cms_slow_size=64,
    )
    return ViTNestedLearning(config)


def create_vit_nested_base(num_classes: int = 1000) -> ViTNestedLearning:
    """Create a base ViT-Nested model (ViT-B equivalent)"""
    config = ViTNestedConfig(
        img_size=224,
        patch_size=16,
        dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        titan_mem_size=512,
        cms_fast_size=256,
        cms_slow_size=128,
    )
    return ViTNestedLearning(config)


def create_vit_nested_large(num_classes: int = 1000) -> ViTNestedLearning:
    """Create a large ViT-Nested model (ViT-L equivalent)"""
    config = ViTNestedConfig(
        img_size=224,
        patch_size=16,
        dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        titan_mem_size=1024,
        cms_fast_size=512,
        cms_slow_size=256,
    )
    return ViTNestedLearning(config)


if __name__ == "__main__":
    # Quick test
    print("Testing ViT with Nested Learning...")
    
    # Create model
    model = create_vit_nested_tiny(num_classes=10)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test memory stats
    stats = model.get_memory_stats()
    print(f"\nMemory stats (block 0): {stats['block_0']}")
    
    print("\n✓ All tests passed!")
