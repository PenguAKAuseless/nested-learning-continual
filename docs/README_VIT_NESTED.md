# Vision Transformer with Nested Learning

A PyTorch implementation combining Vision Transformer (ViT) architecture with Google's Nested Learning (HOPE) framework. This implementation features hierarchical memory systems (TITAN + CMS) and adaptive inner optimizers for continual learning on visual tasks.

## Features

- **Vision Transformer Backbone**: Patch-based image processing with self-attention
- **Hierarchical Memory System**:
  - TITAN memory (fast associative storage)
  - CMS fast/slow (continual memory with chunk accumulation)
- **Inner Optimizers**: Deep momentum with L2-regression updates
- **Test-time Memorization**: Adaptive learning during inference
- **Multiple Model Sizes**: Tiny, Small, Base, and Large configurations

## Architecture Overview

```
Input Image (224×224×3)
    ↓
Patch Embedding (16×16 patches)
    ↓
[CLS Token] + Position Embeddings
    ↓
HOPE Block × N:
  ├─ Self-Attention (Multi-head with SDPA)
  ├─ TITAN Memory (fast associative)
  ├─ CMS Fast (hierarchical storage)
  ├─ CMS Slow (long-term memory)
  └─ MLP (feed-forward)
    ↓
Layer Normalization
    ↓
Classification Head
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd nested-learning-continual

# Install dependencies
pip install torch torchvision tqdm
```

## Quick Start

### Basic Usage

```python
from vision_transformer_nested_learning import create_vit_nested_tiny
import torch

# Create model
model = create_vit_nested_tiny(num_classes=10)

# Forward pass
x = torch.randn(2, 3, 224, 224)  # [batch, channels, height, width]
logits = model(x)  # [batch, num_classes]

# Get memory statistics
stats = model.get_memory_stats()
print(stats['block_0'])
```

### Training on CIFAR-10

```bash
# Train tiny model
python train_vit_nested.py \
  --model tiny \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-3 \
  --inner-lr 0.01 \
  --teach-scale 0.1

# Train with surprise gating
python train_vit_nested.py \
  --model small \
  --surprise-threshold 0.02 \
  --teach-scale 0.15

# Test memorization capability
python train_vit_nested.py \
  --model tiny \
  --epochs 50 \
  --test-memorization
```

## Model Configurations

| Model | Dim | Depth | Heads | TITAN Memory | CMS Fast | CMS Slow | Params |
|-------|-----|-------|-------|--------------|----------|----------|--------|
| Tiny  | 192 | 12    | 3     | 128          | 64       | 32       | ~5M    |
| Small | 384 | 12    | 6     | 256          | 128      | 64       | ~22M   |
| Base  | 768 | 12    | 12    | 512          | 256      | 128      | ~86M   |
| Large | 1024| 24    | 16    | 1024         | 512      | 256      | ~307M  |

## Key Components

### 1. HOPE Block

Each HOPE block combines attention with hierarchical memory:

```python
class HOPEBlock(nn.Module):
    def forward(self, x):
        # Self-attention
        x = x + self.attention(self.norm1(x))
        
        # Compute teaching signal from prediction error
        teach_signal = self.compute_teach_signal(x)
        
        # Hierarchical memory updates
        x = x + self.titan(x, teach_signal)
        x = x + self.cms_fast(x, teach_signal)
        x = x + self.cms_slow(x, teach_signal)
        
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        return x
```

### 2. TITAN Memory

Fast associative memory with attention-based retrieval:

```python
titan = TITANMemory(
    dim=768,
    mem_size=512,
    update_period=1  # Update every step
)

# Memory retrieves and updates based on teach signal
output = titan(input_tokens, teach_signal)
```

### 3. CMS Memory

Continual Memory System with chunk accumulation (Equation 31):

```python
cms_fast = CMSMemory(
    dim=768,
    mem_size=256,
    update_period=8  # Update every 8 steps
)

# Accumulates chunks and updates periodically
output = cms_fast(input_tokens, teach_signal)
```

### 4. Inner Optimizers

L2-regression updates with input-aware preconditioning:

```python
optimizer = DeepMomentumOptimizer(dim=768, lr=0.01, beta=0.9)

# Computes weight update from input activations and teach signal
update = optimizer.compute_update(x, teach_signal)
```

## Advanced Features

### Test-Time Memorization

Enable adaptive learning during inference:

```python
# Enable memorization mode
model.enable_memorization(True)

# Model will adapt to test examples
for images, labels in test_loader:
    logits = model(images)
    # Inner optimizers update based on prediction error
    loss = criterion(logits, labels)
    loss.backward()

# Disable memorization
model.enable_memorization(False)
```

### Surprise Gating

Gate updates based on prediction error magnitude:

```python
# Set surprise threshold (0.0 = no gating)
for block in model.blocks:
    block.surprise_threshold = 0.02

# Updates only occur when teach signal exceeds threshold
```

### Memory Statistics

Monitor memory state during training:

```python
stats = model.get_memory_stats()

for block_id, block_stats in stats.items():
    print(f"{block_id}:")
    print(f"  TITAN steps: {block_stats['titan_step']}")
    print(f"  TITAN memory norm: {block_stats['titan_mem_norm']:.4f}")
    print(f"  CMS fast steps: {block_stats['cms_fast_step']}")
    # ... more stats
```

## Training Tips

1. **Start Small**: Begin with the tiny model to validate pipeline
2. **Adjust Inner Learning Rates**: 
   - TITAN: `inner_lr` (default 0.01)
   - CMS fast: `inner_lr * 0.5`
   - CMS slow: `inner_lr * 0.25`
3. **Teach Scale**: Controls strength of teaching signal (0.05-0.15)
4. **Surprise Threshold**: Use for sparse updates (0.0-0.05)
5. **Gradient Clipping**: Essential for stability with inner updates

## Evaluation

The model can be evaluated in two modes:

1. **Standard Evaluation**: Normal forward pass without adaptation
2. **Memorization Evaluation**: Enable test-time adaptation

```bash
# Compare baseline vs memorization
python train_vit_nested.py \
  --model tiny \
  --test-memorization \
  --data-path ./data
```

## Hyperparameter Recommendations

### For CIFAR-10 (32×32 → 224×224 upsampled):
- Model: `tiny` or `small`
- Batch size: 32-64
- Learning rate: 1e-3 with cosine schedule
- Inner LR: 0.01
- Teach scale: 0.10
- Epochs: 100-200

### For ImageNet (224×224):
- Model: `base` or `large`
- Batch size: 128-256 (distributed)
- Learning rate: 5e-4 with warmup
- Inner LR: 0.005
- Teach scale: 0.05
- Epochs: 300

## Implementation Details

### Key Design Choices:

1. **SDPA Attention**: Uses `F.scaled_dot_product_attention` for FlashAttention compatibility
2. **Chunk Accumulation**: CMS follows Equation 31 from the paper
3. **L2 Regression**: Inner optimizers use input covariance for preconditioning
4. **Separate Embeddings**: Patch embeddings remain convolutional (not tied to head)

### Memory Update Schedule:

| Level | Update Period | Purpose |
|-------|---------------|---------|
| TITAN | 1 step | Fast adaptation to immediate patterns |
| CMS Fast | 8 steps | Medium-term pattern recognition |
| CMS Slow | 64 steps | Long-term knowledge consolidation |

## References

- **Nested Learning (HOPE)**: Google's hierarchical online learning framework
- **TITAN**: Fast associative memory for continual learning
- **Vision Transformer**: An Image is Worth 16×16 Words (Dosovitskiy et al., 2021)

## Citation

If you use this code, please cite:

```bibtex
@article{nested_learning_2024,
  title={Nested Learning: Hierarchical Memory for Continual Learning},
  author={Google Research},
  year={2024}
}

@article{vit_2021,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  journal={ICLR},
  year={2021}
}
```

## License

Apache 2.0 (matching the parent repository)

## TODO

- [ ] Add distributed training support (DDP/FSDP)
- [ ] Implement additional memory architectures (e.g., retrievalaugmented)
- [ ] Add more evaluation benchmarks (ImageNet, etc.)
- [ ] Optimize memory footprint for large models
- [ ] Add mixed precision training (bf16/fp16)
- [ ] Implement continual learning benchmarks
- [ ] Add visualization tools for memory contents

## Contributing

Contributions welcome! Please ensure:
1. Code follows existing style
2. Add tests for new features
3. Update documentation
4. Verify training stability

## Contact

For questions or issues, please open a GitHub issue.
