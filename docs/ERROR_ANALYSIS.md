# Error Analysis & Handling

## Overview

This document catalogs potential errors in the continual learning pipeline and provides solutions. The experiment runner includes automatic error handling for most scenarios.

## Error Categories

### 1. Installation & Environment Errors

#### Error: `ModuleNotFoundError: No module named 'torch'`
**Cause**: PyTorch not installed
**Solution**:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Error: `ImportError: cannot import name 'X' from 'Y'`
**Cause**: Outdated or missing package
**Solution**:
```bash
pip install --upgrade -r requirements.txt
```

#### Error: Version mismatch warnings
**Cause**: Incompatible package versions
**Solution**:
```bash
# Create fresh environment
conda create -n continual python=3.9
conda activate continual
pip install -r requirements.txt
```

### 2. Memory Errors

#### Error: `RuntimeError: CUDA out of memory`
**Cause**: Model + batch too large for GPU
**Impact**: Training crashes
**Recovery**: Automatic restart from checkpoint
**Solutions**:
1. Reduce batch size:
   ```bash
   python run_experiment.py --batch_size 16  # or 8, 4
   ```

2. Use smaller model:
   ```bash
   python run_experiment.py --model_size tiny
   ```

3. Enable gradient checkpointing (future feature):
   ```python
   model = ViTNestedLearning(config, use_checkpoint=True)
   ```

4. Use CPU:
   ```bash
   python run_experiment.py --device cpu
   ```

5. Clear cache periodically:
   ```python
   torch.cuda.empty_cache()
   ```

**Prevention**:
- Start with tiny model and small batch
- Monitor GPU memory: `nvidia-smi`
- Use mixed precision training (future feature)

#### Error: `RuntimeError: CUDA error: out of memory` during evaluation
**Cause**: Test batch size too large
**Solution**:
The runner uses `batch_size * 2` for testing. If OOM occurs during eval:
```python
# In run_experiment.py, modify test_loader creation:
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,  # Changed from batch_size * 2
    shuffle=False,
)
```

#### Error: `MemoryError` (CPU RAM)
**Cause**: Dataset or model too large for system RAM
**Solutions**:
1. Reduce number of workers:
   ```bash
   python run_experiment.py --num_workers 0
   ```

2. Enable memory pinning (only with GPU):
   ```python
   pin_memory = True if device == 'cuda' else False
   ```

### 3. Data Loading Errors

#### Error: `FileNotFoundError: Dataset not found at <path>`
**Cause**: Dataset not downloaded
**Solution**: 
Datasets download automatically. Ensure:
- Internet connection available
- Write permissions to data directory
- Sufficient disk space (CIFAR: ~200MB, ImageNet: ~150GB)

Manual download:
```python
from torchvision import datasets
datasets.CIFAR10(root='./data', download=True)
datasets.CIFAR100(root='./data', download=True)
datasets.MNIST(root='./data', download=True)
```

#### Error: `RuntimeError: DataLoader worker (pid X) is killed`
**Cause**: Too many workers or memory issue
**Solution**:
```bash
python run_experiment.py --num_workers 2  # or 0
```

#### Error: `ConnectionError` during download
**Cause**: Network issue or mirror down
**Solution**:
1. Retry with better connection
2. Download manually and place in `./data`
3. Use alternative mirror:
   ```python
   # In datasets.py, add mirror parameter
   datasets.CIFAR10(root, download=True, mirror='https://...')
   ```

#### Error: `Broken pipe` in DataLoader
**Cause**: Worker process crash
**Solution**:
```bash
# Set workers to 0 to debug
python run_experiment.py --num_workers 0

# If it works, gradually increase workers
python run_experiment.py --num_workers 2
```

### 4. Model Errors

#### Error: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
**Cause**: Input size mismatch
**Check**:
- Image size: CIFAR=32x32, MNIST=28x28
- Number of channels: RGB=3, Grayscale=1
- Patch size must divide image size evenly

**Solution**:
```python
# Correct configurations
# CIFAR: img_size=32, patch_size=4 (32/4=8 patches)
# MNIST: img_size=28, patch_size=4 (28/4=7 patches) or patch_size=7 (28/7=4)

config = ViTNestedConfig(
    img_size=32,  # Match dataset
    patch_size=4,  # Must divide img_size
    in_channels=3,  # 3 for RGB, 1 for grayscale
)
```

#### Error: `RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same`
**Cause**: Model and data on different devices
**Solution**: Runner handles this automatically via `.to(device)`. If custom code:
```python
model = model.to(device)
x, y = x.to(device), y.to(device)
```

#### Error: `KeyError: 'Unexpected key(s) in state_dict'`
**Cause**: Loading checkpoint from different model architecture
**Solution**:
```python
# Use strict=False for partial loading
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# Or check model architecture matches checkpoint
print(checkpoint['config'])
```

#### Error: NaN loss or gradients
**Cause**: Learning rate too high, numerical instability
**Symptoms**:
- Loss becomes NaN
- Accuracy drops to 0% or random
- Gradients explode

**Solutions**:
1. Reduce learning rate:
   ```bash
   python run_experiment.py --lr 0.0001
   ```

2. Add gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. Check for invalid operations:
   ```python
   # Add assertions
   assert not torch.isnan(loss), "Loss is NaN!"
   assert not torch.isinf(loss), "Loss is inf!"
   ```

4. Use mixed precision with loss scaling (future):
   ```python
   scaler = torch.cuda.amp.GradScaler()
   ```

### 5. Strategy-Specific Errors

#### EWC: `RuntimeError: Fisher matrix has NaN values`
**Cause**: Numerical instability during Fisher computation
**Solution**:
```python
# Add epsilon for stability
fisher = fisher + 1e-8

# Reduce sample size
python run_experiment.py --strategy ewc --fisher_sample_size 100

# Adjust lambda
python run_experiment.py --lambda_ewc 1000
```

#### EWC: `MemoryError` during Fisher computation
**Cause**: Fisher matrix too large
**Solution**:
```python
# In rivalry_strategies.py, use sparse Fisher
# Or compute Fisher blockwise
# Or sample fewer parameters
```

#### LwF: `RuntimeError: Old model outputs don't match`
**Cause**: Model architecture changed between tasks
**Solution**: Ensure consistent architecture across tasks

#### GEM: `ValueError: Memory buffer index out of range`
**Cause**: Accessing invalid memory slot
**Solution**:
```python
# Increase memory size
python run_experiment.py --strategy gem --memory_size 512

# Or check memory indexing logic
```

#### PackNet: `RuntimeError: Cannot prune non-existent weights`
**Cause**: Pruning ratio too aggressive
**Solution**:
```python
python run_experiment.py --strategy packnet --prune_ratio 0.3  # Less aggressive
```

### 6. Training Errors

#### Error: No learning / accuracy stays at random
**Possible causes**:
1. Learning rate too low/high
2. Model too small/large
3. Data not normalized
4. Wrong loss function
5. Incorrect labels

**Diagnosis**:
```python
# Check data
print(x.min(), x.max(), x.mean())  # Should be normalized
print(y.min(), y.max())  # Should be valid class indices

# Check outputs
outputs = model(x)
print(outputs.shape)  # Should be [batch_size, num_classes]
print(outputs.min(), outputs.max())  # Should be reasonable

# Check loss
print(loss.item())  # Should decrease over time
```

**Solutions**:
1. Adjust learning rate:
   ```bash
   python run_experiment.py --lr 0.01  # Higher
   python run_experiment.py --lr 0.0001  # Lower
   ```

2. Increase epochs:
   ```bash
   python run_experiment.py --epochs 20
   ```

3. Check data normalization:
   ```python
   # In datasets.py, ensure transforms include:
   transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
   ```

#### Error: `RuntimeError: grad can be implicitly created only for scalar outputs`
**Cause**: Calling backward on non-scalar
**Solution**:
```python
# Ensure loss is scalar
loss = criterion(outputs, targets)  # Scalar
loss.backward()  # OK

# Not:
losses = criterion(outputs, targets, reduction='none')  # Tensor
losses.backward()  # ERROR
```

#### Error: `RuntimeError: Trying to backward through the graph a second time`
**Cause**: Calling backward twice without retaining graph
**Solution**:
```python
# For GEM or other multi-gradient methods
loss.backward(retain_graph=True)

# Or clear gradients between backward calls
optimizer.zero_grad()
```

### 7. Evaluation Errors

#### Error: Accuracy calculation gives values > 100%
**Cause**: Bug in accuracy computation
**Check**:
```python
def accuracy(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    total = target.size(0)
    return 100.0 * correct / total  # Should be 0-100
```

#### Error: Different accuracies on same data
**Cause**: Stochastic behavior (dropout, batch norm in train mode)
**Solution**:
```python
# Ensure evaluation mode
model.eval()

# Disable gradients
with torch.no_grad():
    outputs = model(x)

# Disable dropout/batch norm stochasticity
# (automatic with model.eval())
```

### 8. Checkpoint & I/O Errors

#### Error: `PermissionError: [Errno 13] Permission denied`
**Cause**: No write permission to save directory
**Solution**:
```bash
# Check permissions
ls -la results/

# Change ownership
sudo chown -R $USER:$USER results/

# Or save to different directory
python run_experiment.py --save_dir ~/my_results
```

#### Error: `OSError: [Errno 28] No space left on device`
**Cause**: Disk full
**Solution**:
```bash
# Check disk space
df -h

# Disable checkpoints
python run_experiment.py --no_checkpoints

# Clean old results
rm -rf results/old_experiment

# Use smaller model (smaller checkpoints)
python run_experiment.py --model_size tiny
```

#### Error: `RuntimeError: Unable to save checkpoint`
**Cause**: Serialization error or disk full
**Solution**:
```python
# Try saving only model weights
torch.save(model.state_dict(), path)

# Instead of full checkpoint
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    # ... other state
}, path)
```

### 9. Visualization Errors

#### Error: `ImportError: No module named 'matplotlib'`
**Cause**: Missing visualization library
**Solution**:
```bash
pip install matplotlib seaborn
```

#### Error: `UserWarning: Matplotlib is currently using agg, which is a non-GUI backend`
**Cause**: No display available (SSH/server)
**Solution**:
```python
# In runner.py, use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

#### Error: Plots are empty or malformed
**Cause**: Data format issue
**Check**:
```python
# Ensure accuracy_matrix is 2D numpy array
assert isinstance(accuracy_matrix, np.ndarray)
assert accuracy_matrix.ndim == 2

# Check for NaN values
assert not np.any(np.isnan(accuracy_matrix))
```

### 10. Concurrency Errors

#### Error: `RuntimeError: CUDA error: device-side assert triggered`
**Cause**: Invalid operation on GPU (often index out of bounds)
**Debug**:
```bash
# Run on CPU to get better error messages
python run_experiment.py --device cpu

# Enable CUDA error checking
export CUDA_LAUNCH_BLOCKING=1
python run_experiment.py
```

#### Error: `RuntimeError: CUDA error: an illegal memory access was encountered`
**Cause**: Memory corruption or invalid indices
**Solutions**:
1. Check for out-of-bounds indexing:
   ```python
   assert (y >= 0).all() and (y < num_classes).all()
   ```

2. Reduce batch size
3. Run on CPU to isolate issue

## Error Recovery Features

The `ExperimentRunner` includes automatic error recovery:

### 1. Batch-Level Recovery
```python
try:
    loss = strategy.train_step(x, y, optimizer)
except Exception as e:
    logger.warning(f"Error in batch {batch_idx}: {e}")
    continue  # Skip batch, continue training
```

### 2. Task-Level Recovery
```python
try:
    self._train_task(model, strategy, train_loader, optimizer, task_id)
except Exception as e:
    logger.error(f"Error in task {task_id}: {e}")
    self.results['errors'].append({
        'task': task_id,
        'error': str(e),
        'traceback': traceback.format_exc()
    })
    continue  # Continue to next task
```

### 3. Checkpoint Recovery
```python
# Automatically saves after each task
if self.config.save_checkpoints:
    self._save_checkpoint(model, optimizer, strategy, task_id)

# Can resume from checkpoint
task_id = load_checkpoint('path/to/checkpoint.pt', model, optimizer)
```

### 4. Results Preservation
```python
# Results saved after each task
self._save_results()  # Even if experiment crashes later

# Plots generated incrementally
self._plot_results(accuracy_matrix[:task_id+1], task_id)
```

## Debugging Strategies

### 1. Minimal Test Case
```bash
# Simplest possible run
python run_experiment.py \
    --model_size tiny \
    --num_tasks 2 \
    --epochs 1 \
    --batch_size 8 \
    --device cpu
```

### 2. Enable Debug Logging
```python
# In runner.py or run_experiment.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 3. Add Assertions
```python
# After each critical operation
assert not torch.isnan(loss), f"Loss is NaN at step {step}"
assert accuracy <= 100.0, f"Invalid accuracy: {accuracy}"
```

### 4. Profile Performance
```python
import torch.utils.benchmark as benchmark

# Time operations
t = benchmark.Timer(
    stmt='model(x)',
    globals={'model': model, 'x': x}
)
print(t.timeit(100))
```

### 5. Memory Profiling
```python
import torch
torch.cuda.memory_summary()
torch.cuda.max_memory_allocated()
```

## Prevention Best Practices

1. **Validate inputs** before training
2. **Start small** (tiny model, few epochs)
3. **Check intermediate outputs** for sanity
4. **Save frequently** (auto-handled by runner)
5. **Monitor resources** (GPU memory, disk space)
6. **Use error handling** (try-except blocks)
7. **Log everything** (runner does this automatically)
8. **Test on CPU first** if GPU issues suspected

## Common Warning Messages

### Warning: `UserWarning: Mixed memory format inputs detected`
**Meaning**: PyTorch detected non-contiguous tensors
**Impact**: Slightly slower, but safe
**Solution**: Usually safe to ignore, or:
```python
x = x.contiguous()
```

### Warning: `UserWarning: The dataloader uses num_workers > 0`
**Meaning**: Multi-process data loading
**Impact**: None (informational)
**Solution**: Ignore or set `--num_workers 0`

### Warning: `DeprecationWarning: XXXXX is deprecated`
**Meaning**: Using old API
**Impact**: Will break in future versions
**Solution**: Update code to use new API

## Getting Help

If none of these solutions work:

1. **Check error logs**: `results/experiment_name/results.json`
2. **Run minimal test**: Single task, tiny model, CPU
3. **Check versions**: `pip list | grep torch`
4. **Search issues**: Likely someone encountered this before
5. **Create reproducible example**: Minimal code that shows the error

## Error Checklist

Before reporting an error:

- [ ] Tried on CPU
- [ ] Tried with smaller model/batch
- [ ] Checked disk space
- [ ] Checked GPU memory
- [ ] Updated dependencies
- [ ] Reviewed error logs
- [ ] Tried minimal test case
- [ ] Can reproduce consistently

## Summary

The experiment runner is designed to be robust:
- ✅ Automatic error recovery at multiple levels
- ✅ Detailed error logging
- ✅ Checkpoint preservation
- ✅ Graceful degradation
- ✅ Incremental result saving

Most errors are recoverable. The runner will log errors and continue when possible, ensuring partial results are always saved.
