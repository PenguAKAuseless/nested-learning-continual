# Comprehensive User Guide

## Getting Started

### Installation Steps

1. **Clone and setup**:
```bash
git clone <repo-url>
cd nested-learning-continual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Verify installation**:
```bash
python test_vit_nested.py
```

## Usage Examples

### Example 1: Basic Training

```python
from vision_transformer_nested_learning import create_vit_nested_tiny
from torchvision import datasets, transforms
import torch.optim as optim

# Create model
model = create_vit_nested_tiny(num_classes=10).cuda()

# Load data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
```

### Example 2: Continual Learning with EWC

```python
from continual_learning import EWCStrategy
from data import SplitCIFAR10

# Setup
model = create_vit_nested_tiny(num_classes=10).cuda()
strategy = EWCStrategy(model, device='cuda', lambda_ewc=1000)
split_cifar = SplitCIFAR10(root='./data', num_tasks=5)

# Train on each task
for task_id in range(5):
    train_dataset = split_cifar.get_task(task_id, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    strategy.before_task(task_id)
    
    for epoch in range(10):
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = strategy.train_step(x, y, optimizer)
    
    strategy.after_task(train_loader)
```

### Example 3: Method Comparison

```python
from experiments import BenchmarkSuite
from continual_learning import EWCStrategy, LwFStrategy, NaiveStrategy

# Define methods
methods = {
    'Naive': {
        'model_fn': lambda: create_vit_nested_tiny(num_classes=10),
        'strategy_fn': lambda m: NaiveStrategy(m, 'cuda'),
    },
    'EWC': {
        'model_fn': lambda: create_vit_nested_tiny(num_classes=10),
        'strategy_fn': lambda m: EWCStrategy(m, 'cuda'),
    },
}

# Run benchmark
suite = BenchmarkSuite(output_dir='./results')
results = suite.run_benchmark('split_cifar10', methods, task_loaders)
```

### Example 4: Custom Model Configuration

```python
from vision_transformer_nested_learning import ViTNestedConfig, ViTNestedLearning

config = ViTNestedConfig(
    img_size=224,
    patch_size=16,
    dim=384,
    depth=12,
    num_heads=6,
    titan_mem_size=256,
    cms_fast_size=128,
    cms_slow_size=64,
    titan_update_period=1,
    cms_fast_update_period=8,
    cms_slow_update_period=64,
    inner_lr=0.01,
    teach_scale=0.1,
    surprise_threshold=0.02,
    num_classes=100,
)

model = ViTNestedLearning(config)
```

## Hyperparameter Tuning Guide

### ViT-Nested Specific Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `teach_scale` | 0.05-0.15 | 0.10 | Teaching signal strength |
| `inner_lr` | 0.001-0.05 | 0.01 | Inner optimizer learning rate |
| `surprise_threshold` | 0.0-0.05 | 0.0 | Surprise gating threshold |
| `titan_mem_size` | 64-1024 | 512 | TITAN memory slots |
| `cms_fast_size` | 32-512 | 256 | CMS fast memory slots |
| `cms_slow_size` | 16-256 | 128 | CMS slow memory slots |

### Training Parameters

| Parameter | Tiny | Small | Base | Large |
|-----------|------|-------|------|-------|
| Batch Size | 32-64 | 32-64 | 16-32 | 8-16 |
| Learning Rate | 1e-3 | 5e-4 | 3e-4 | 1e-4 |
| Warmup Epochs | 5 | 10 | 15 | 20 |
| Weight Decay | 0.05 | 0.05 | 0.1 | 0.1 |

### Strategy-Specific Parameters

**EWC**:
- `lambda_ewc`: 100-10000 (higher = more conservative)
- `fisher_sample_size`: 200-1000

**LwF**:
- `lambda_lwf`: 0.5-2.0
- `temperature`: 1.0-4.0

**GEM**:
- `memory_size`: 128-1024 (samples per task)

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller model variant

**2. NaN Loss**
- Lower learning rate
- Reduce teach_scale
- Add gradient clipping

**3. Poor Continual Learning Performance**
- Increase memory sizes
- Adjust teach_scale
- Try different update periods

**4. Slow Training**
- Reduce num_workers if CPU-bound
- Use GPU if available
- Consider mixed precision training

## Advanced Topics

### Custom Memory Architectures

```python
class CustomMemory(nn.Module):
    def __init__(self, dim, mem_size):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, dim))
    
    def forward(self, x, teach_signal=None):
        # Custom memory logic
        return output
```

### Custom Continual Learning Strategy

```python
from continual_learning.rivalry_strategies import RivalryStrategy

class MyStrategy(RivalryStrategy):
    def train_step(self, x, y, optimizer):
        # Custom training logic
        pass
    
    def after_task(self, dataloader):
        # Post-task processing
        pass
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)

# Train as usual
```

## Performance Optimization

### Tips for Faster Training

1. **Use CUDA**: Always train on GPU when available
2. **Data Loading**: Set `num_workers=4` or higher
3. **Pin Memory**: Use `pin_memory=True` in DataLoader
4. **Batch Size**: Maximize batch size within memory limits
5. **Gradient Accumulation**: For larger effective batch sizes

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision**: Use torch.cuda.amp
3. **Memory-Efficient Attention**: Already implemented with SDPA

## FAQ

**Q: Can I use this for other vision tasks?**
A: Yes, adapt the classification head for detection, segmentation, etc.

**Q: How do I add a new benchmark dataset?**
A: Implement similar to `SplitCIFAR10` in `data/datasets.py`

**Q: Can I combine multiple strategies?**
A: Yes, create a composite strategy that applies multiple regularizations

**Q: How do I cite this work?**
A: See Citation section in README.md

## Best Practices

1. **Always set seeds** for reproducibility
2. **Log extensively** during experiments
3. **Use validation sets** for hyperparameter tuning
4. **Save checkpoints** regularly
5. **Compare against baselines** to validate improvements
6. **Document experiments** thoroughly

## References

- [Nested Learning Paper](google_papers/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [EWC Paper](https://arxiv.org/abs/1612.00796)
- [LwF Paper](https://arxiv.org/abs/1606.09282)

## Support

For issues, questions, or contributions:
- GitHub Issues
- Pull Requests welcome
- Email: your.email@example.com
