# Experiment Runner Usage Guide

## Overview

This guide explains how to use the experiment runner to conduct continual learning experiments with proper error handling, result tracking, and visualization.

## Quick Start

### Basic Usage

```bash
# Run ViT-Nested with naive strategy on Split-CIFAR10
python run_experiment.py --method vit_nested --strategy naive --dataset split_cifar10

# Run with EWC strategy for 10 epochs per task
python run_experiment.py --strategy ewc --epochs 10 --lambda_ewc 5000

# Run on Split-MNIST with 10 tasks
python run_experiment.py --dataset split_mnist --num_tasks 10
```

### Installation

First, ensure all dependencies are installed:

```bash
# Install basic requirements
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt

# For GPU support, install PyTorch with CUDA (REQUIRED for GPU usage)
# Check your CUDA version first: nvidia-smi
# For CUDA 12.1 (most compatible):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Note: Installing torch without specifying the index-url will install CPU-only version
```

**Important**: If you get `"Torch not compiled with CUDA enabled"` error:
```bash
# Uninstall CPU version
pip uninstall torch torchvision -y

# Reinstall with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Command-Line Arguments

### Model Configuration
- `--method`: Method name (currently: `vit_nested`)
- `--model_size`: Model size (`tiny`, `small`, `base`)
  - **tiny**: 192-dim, 6 layers, ~5M params
  - **small**: 384-dim, 8 layers, ~22M params
  - **base**: 768-dim, 12 layers, ~86M params

### Strategy Configuration
- `--strategy`: Continual learning strategy
  - `naive`: No regularization (baseline)
  - `ewc`: Elastic Weight Consolidation
  - `lwf`: Learning without Forgetting
  - `gem`: Gradient Episodic Memory
  - `packnet`: PackNet pruning
  - `si`: Synaptic Intelligence

#### Strategy-Specific Parameters
- `--lambda_ewc`: EWC regularization strength (default: 5000.0)
- `--lambda_lwf`: LwF distillation weight (default: 1.0)
- `--memory_size`: GEM memory buffer size (default: 256)

### Dataset Configuration
- `--dataset`: Dataset to use
  - `split_cifar10`: CIFAR-10 split into tasks by classes
  - `split_cifar100`: CIFAR-100 split into tasks
  - `split_mnist`: MNIST split into tasks
  - `permuted_mnist`: MNIST with random pixel permutations
  - `rotated_mnist`: MNIST with rotations
- `--num_tasks`: Number of tasks (default: 5)
- `--data_dir`: Data directory (default: `./data`)

### Training Configuration
- `--epochs`: Epochs per task (default: 5)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0001)

### System Configuration
- `--device`: Device (`cuda:0` or `cpu`)
- `--seed`: Random seed (default: 42)
- `--num_workers`: Data loading workers (default: 4)
- `--amp`: Enable automatic mixed precision training (reserved for future implementation)

### Experiment Configuration
- `--save_dir`: Results directory (default: `results`)
- `--experiment_name`: Custom experiment name
- `--log_interval`: Logging frequency (default: 50)
- `--no_checkpoints`: Disable checkpoint saving

## Example Workflows

### 1. Quick Test Run (Small Model, Few Epochs)

```bash
python run_experiment.py \
    --model_size tiny \
    --strategy naive \
    --dataset split_cifar10 \
    --num_tasks 5 \
    --epochs 3 \
    --batch_size 64 \
    --experiment_name quick_test
```

### 2. Full Experiment with EWC

```bash
python run_experiment.py \
    --model_size small \
    --strategy ewc \
    --lambda_ewc 5000 \
    --dataset split_cifar10 \
    --num_tasks 5 \
    --epochs 10 \
    --lr 0.0005 \
    --seed 42 \
    --experiment_name ewc_full
```

### 3. Comparison Run (Multiple Strategies)

```bash
# Naive baseline
python run_experiment.py --strategy naive --experiment_name comp_naive

# EWC
python run_experiment.py --strategy ewc --experiment_name comp_ewc

# LwF
python run_experiment.py --strategy lwf --experiment_name comp_lwf

# GEM
python run_experiment.py --strategy gem --experiment_name comp_gem
```

### 4. Different Datasets

```bash
# CIFAR-100 with 10 tasks (10 classes per task)
python run_experiment.py --dataset split_cifar100 --num_tasks 10 --epochs 15

# Permuted MNIST (domain-incremental)
python run_experiment.py --dataset permuted_mnist --num_tasks 10 --epochs 5

# Split MNIST (class-incremental)
python run_experiment.py --dataset split_mnist --num_tasks 5 --epochs 10
```

## Output Structure

After running an experiment, results are saved in the following structure:

```
results/
└── experiment_name/
    ├── config.json                          # Experiment configuration
    ├── results.json                         # Full results including metrics
    ├── checkpoints/                         # Model checkpoints
    │   ├── task_0_checkpoint.pt
    │   ├── task_1_checkpoint.pt
    │   └── ...
    └── plots/                               # Visualizations
        ├── accuracy_matrix_final.png        # Final accuracy matrix heatmap
        ├── task_evolution_final.png         # Task accuracy over time
        ├── losses_final.png                 # Training losses
        ├── accuracy_matrix_task_0.png       # Intermediate results
        └── ...
```

## Understanding Results

### Metrics

The runner computes four key continual learning metrics:

1. **Average Accuracy**: Mean accuracy across all tasks after training
   - Higher is better
   - Formula: Mean of final row in accuracy matrix

2. **Forgetting**: Average accuracy drop on previous tasks
   - Lower is better (negative values indicate improvement)
   - Formula: Mean difference between peak and final accuracy

3. **Forward Transfer**: Knowledge transfer to future tasks
   - Positive values indicate positive transfer
   - Formula: Initial accuracy on new tasks vs random baseline

4. **Backward Transfer**: Effect of learning new tasks on old tasks
   - Negative values indicate catastrophic forgetting
   - Formula: Change in old task accuracy after new task

### Visualizations

#### 1. Accuracy Matrix (`accuracy_matrix_final.png`)
- **Rows**: Training progress (after completing each task)
- **Columns**: Evaluation task
- **Diagonal**: Current task accuracy (usually highest)
- **Below diagonal**: Old task accuracy (shows forgetting)

#### 2. Task Evolution (`task_evolution_final.png`)
- Shows how each task's accuracy changes over training
- Flat lines = no forgetting
- Declining lines = catastrophic forgetting
- Rising lines = positive backward transfer

#### 3. Training Losses (`losses_final.png`)
- Loss curves for each task
- Log scale for better visualization
- Separate curves for each task

## Common Error Scenarios & Solutions

### 1. Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
python run_experiment.py --batch_size 16

# Use smaller model
python run_experiment.py --model_size tiny

# Use CPU
python run_experiment.py --device cpu

# Reduce number of workers
python run_experiment.py --num_workers 0
```

### 2. Data Not Found

**Error**: `FileNotFoundError: Dataset not found`

**Solutions**:
```bash
# Specify data directory
python run_experiment.py --data_dir /path/to/data

# Data will be downloaded automatically on first run
# Ensure internet connection for download
```

### 3. Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solutions**:
```bash
# Install dependencies
pip install -r requirements.txt

# Or install package in development mode
pip install -e .
```

**Error**: `AssertionError: Torch not compiled with CUDA enabled`

This means PyTorch was installed without CUDA support (CPU-only version).

**Solution**:
```bash
# Check your GPU and CUDA version
nvidia-smi

# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall with CUDA support (use the appropriate CUDA version)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 4. Convergence Issues

**Symptoms**: Very low accuracy, no learning

**Solutions**:
```bash
# Increase learning rate
python run_experiment.py --lr 0.01

# Increase epochs
python run_experiment.py --epochs 20

# Check if model size is appropriate
python run_experiment.py --model_size small

# Try different seed
python run_experiment.py --seed 123
```

### 5. Strategy-Specific Errors

#### EWC: "Fisher matrix computation failed"
```bash
# Reduce Fisher sample size
python run_experiment.py --strategy ewc --fisher_sample_size 100

# Adjust lambda
python run_experiment.py --strategy ewc --lambda_ewc 1000
```

#### GEM: "Memory buffer full"
```bash
# Increase memory size
python run_experiment.py --strategy gem --memory_size 512
```

## Error Recovery

The runner includes automatic error recovery:

1. **Batch-level errors**: Skipped automatically, training continues
2. **Task-level errors**: Logged and training continues to next task
3. **Experiment-level errors**: Results saved before exit

All errors are logged in `results.json` under the `errors` field:

```json
{
  "errors": [
    {
      "task": 2,
      "phase": "training",
      "error": "CUDA out of memory",
      "traceback": "..."
    }
  ]
}
```

## Checkpoint Recovery

To resume from a checkpoint:

```python
from utils.runner import load_checkpoint
import torch

# Load checkpoint
model = create_model()
optimizer = torch.optim.AdamW(model.parameters())
task_id = load_checkpoint('results/exp/checkpoints/task_2_checkpoint.pt', model, optimizer)

print(f"Resumed from task {task_id}")
```

## Performance Monitoring

### Memory Usage
```bash
# Monitor GPU memory
watch -n 1 nvidia-smi

# Or use smaller batches
python run_experiment.py --batch_size 16
```

### Training Speed
```bash
# Increase workers for faster data loading
python run_experiment.py --num_workers 8

# Use pin_memory for GPU
# (automatically enabled when device=cuda)

# Reduce logging frequency
python run_experiment.py --log_interval 100
```

### Disk Space
```bash
# Disable checkpoints if disk space is limited
python run_experiment.py --no_checkpoints

# Checkpoints are ~500MB for base model
# ~100MB for small model
# ~20MB for tiny model
```

## Advanced Usage

### Custom Experiment Name
```bash
# Use descriptive names for organization
python run_experiment.py \
    --experiment_name "vit_tiny_ewc_lambda5000_cifar10_seed42"
```

### Multiple Seeds for Statistical Significance
```bash
for seed in 42 123 456 789 1024; do
    python run_experiment.py \
        --strategy ewc \
        --seed $seed \
        --experiment_name "ewc_seed_${seed}"
done
```

### Hyperparameter Search
```bash
# Search over EWC lambda values
for lambda in 1000 5000 10000 50000; do
    python run_experiment.py \
        --strategy ewc \
        --lambda_ewc $lambda \
        --experiment_name "ewc_lambda_${lambda}"
done
```

## Programmatic Usage

You can also use the runner programmatically:

```python
import torch
from vision_transformer_nested_learning import ViTNestedLearning, ViTNestedConfig
from continual_learning.rivalry_strategies import EWCStrategy
from utils.runner import ExperimentRunner, RunConfig
from data.datasets import SplitCIFAR10

# Create configuration
config = RunConfig(
    method_name="ViT-Nested",
    model_size="tiny",
    num_classes=10,
    num_epochs=5,
    batch_size=32,
    strategy_type="ewc",
    dataset_name="split_cifar10",
    num_tasks=5,
    experiment_name="my_experiment",
)

# Create model and strategy
vit_config = ViTNestedConfig(img_size=32, patch_size=4, num_classes=10, dim=192, depth=6)
model = ViTNestedLearning(vit_config)
strategy = EWCStrategy(model, device='cuda', lambda_ewc=5000.0)

# Load data
dataset = SplitCIFAR10(root='./data', num_tasks=5)
task_loaders = []
for i in range(5):
    train_ds, test_ds = dataset.get_task(i)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
    task_loaders.append((train_loader, test_loader))

# Run experiment
runner = ExperimentRunner(config)
results = runner.run(model, strategy, task_loaders)

# Access results
print(f"Average Accuracy: {results['metrics']['average_accuracy']:.2f}%")
print(f"Forgetting: {results['metrics']['forgetting']:.2f}%")
```

## Troubleshooting Checklist

- [ ] PyTorch installed with correct CUDA version
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space for data and checkpoints
- [ ] Sufficient GPU memory (or use `--device cpu`)
- [ ] Dataset downloaded (happens automatically on first run)
- [ ] Correct Python version (3.8+)
- [ ] No conflicting package versions

## Getting Help

If you encounter issues:

1. Check error messages in terminal output
2. Review `results.json` for logged errors
3. Try with `--model_size tiny` and `--epochs 1` for quick testing
4. Use `--device cpu` to rule out GPU issues
5. Check [README.md](README.md) for general project information
6. Check [GUIDE.md](GUIDE.md) for detailed usage examples

## Best Practices

1. **Always set seed** for reproducibility: `--seed 42`
2. **Use descriptive experiment names** for organization
3. **Start with tiny model** for quick testing
4. **Monitor first task** - if it doesn't learn, check hyperparameters
5. **Save results** - checkpoints enable analysis and resumption
6. **Compare multiple seeds** - run 3-5 seeds for statistical significance
7. **Check plots** - visual inspection reveals issues quickly
8. **Read errors** - the runner logs detailed error information

## Expected Results

### Baseline Performance (5 epochs per task, tiny model)

| Dataset | Strategy | Avg Accuracy | Forgetting |
|---------|----------|--------------|------------|
| Split-CIFAR10 | Naive | ~50% | ~30% |
| Split-CIFAR10 | EWC | ~60% | ~20% |
| Split-CIFAR10 | LwF | ~62% | ~18% |
| Split-CIFAR10 | GEM | ~65% | ~15% |
| Split-MNIST | Naive | ~75% | ~15% |
| Split-MNIST | EWC | ~85% | ~8% |

*Note: Actual results vary with hyperparameters, model size, and random seed.*

## Measurement Protocol

The runner follows this measurement protocol:

1. **Train** on Task T for N epochs
2. **Measure** accuracy on all tasks 0...T after training
3. **Update** strategy state (compute Fisher, save exemplars, etc.)
4. **Save** checkpoint and intermediate results
5. **Plot** current progress
6. **Repeat** for next task

This ensures:
- ✅ Fair comparison across methods
- ✅ Tracking of catastrophic forgetting
- ✅ Intermediate results saved (resilient to crashes)
- ✅ Progress visualization after each task

## Time Estimates

Approximate time per task (CIFAR-10, 5 epochs, tiny model):

- **CPU**: ~30-45 minutes per task
- **GPU (GTX 1080)**: ~5-8 minutes per task
- **GPU (RTX 3090)**: ~2-4 minutes per task

Total experiment time for 5 tasks:
- **CPU**: ~2.5-4 hours
- **GPU (GTX 1080)**: ~25-40 minutes
- **GPU (RTX 3090)**: ~10-20 minutes

Scaling factors:
- Small model: 2-3x slower
- Base model: 8-10x slower
- 10 epochs: 2x slower
- CIFAR-100: 1.5x slower

## Next Steps

After running experiments:

1. **Analyze results**: Check `results.json` and plots
2. **Compare strategies**: Run multiple experiments
3. **Tune hyperparameters**: Adjust learning rate, regularization
4. **Scale up**: Try larger models or more tasks
5. **Publish**: Use plots and metrics in papers/reports
