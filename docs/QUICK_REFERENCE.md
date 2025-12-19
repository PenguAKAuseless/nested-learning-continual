# Quick Reference Card

## 🚀 Quick Commands

### Validate Framework
```bash
python validate_framework.py
```

### Run Quick Test (1-2 min)
```bash
python run_experiment.py --model_size tiny --epochs 1 --num_tasks 2
```

### Run Full Experiment (10-30 min)
```bash
python run_experiment.py --strategy ewc --epochs 5
```

### Compare Strategies
```bash
# Naive
python run_experiment.py --strategy naive --experiment_name cmp_naive

# EWC
python run_experiment.py --strategy ewc --experiment_name cmp_ewc

# LwF
python run_experiment.py --strategy lwf --experiment_name cmp_lwf

# GEM
python run_experiment.py --strategy gem --experiment_name cmp_gem
```

## 📁 Output Files

```
results/experiment_name/
├── results.json                    # All metrics & accuracy
├── config.json                     # Experiment settings
├── plots/
│   ├── accuracy_matrix_final.png   # ⭐ Main result
│   ├── task_evolution_final.png    # Forgetting plot
│   └── losses_final.png            # Training curves
└── checkpoints/
    └── task_*.pt                   # Model checkpoints
```

## 📊 Key Metrics

In `results.json`:
- `average_accuracy` - Overall performance
- `forgetting` - How much old tasks degrade
- `forward_transfer` - Help to future tasks
- `backward_transfer` - Impact on old tasks

## 🎛️ Common Options

### Model Sizes
```bash
--model_size tiny    # Fast, 192-dim (recommended for testing)
--model_size small   # Balanced, 384-dim
--model_size base    # Large, 768-dim
```

### Strategies
```bash
--strategy naive     # No regularization (baseline)
--strategy ewc       # Elastic Weight Consolidation
--strategy lwf       # Learning without Forgetting
--strategy gem       # Gradient Episodic Memory
--strategy packnet   # Pruning-based
--strategy si        # Synaptic Intelligence
```

### Datasets
```bash
--dataset split_cifar10    # CIFAR-10, 5 tasks (default)
--dataset split_cifar100   # CIFAR-100, 10 tasks
--dataset split_mnist      # MNIST, 5 tasks
--dataset permuted_mnist   # MNIST permutations
--dataset rotated_mnist    # MNIST rotations
```

### Training
```bash
--epochs 5          # Epochs per task (default: 5)
--batch_size 32     # Batch size (default: 32)
--lr 0.001          # Learning rate (default: 0.001)
--num_tasks 5       # Number of tasks (default: 5)
```

### Strategy Parameters
```bash
# EWC
--lambda_ewc 5000           # Regularization strength

# LwF
--lambda_lwf 1.0            # Distillation weight
--temperature 2.0           # Temperature

# GEM
--memory_size 256           # Memory buffer size
```

## 🐛 Quick Troubleshooting

### Out of Memory
```bash
python run_experiment.py --batch_size 16 --model_size tiny
# Or
python run_experiment.py --device cpu
```

### Not Learning
```bash
# Increase learning rate or epochs
python run_experiment.py --lr 0.01 --epochs 10
```

### Import Errors
```bash
pip install -r requirements.txt
```

## 📚 Documentation

- **USAGE_SUMMARY.md** - ⭐ Start here for complete guide
- **EXPERIMENT_GUIDE.md** - Detailed instructions
- **ERROR_ANALYSIS.md** - Error solutions
- **README.md** - Project overview

## 🎯 Typical Workflow

```bash
# 1. Validate setup
python validate_framework.py

# 2. Quick test
python run_experiment.py --epochs 1 --num_tasks 2

# 3. Full run
python run_experiment.py --strategy ewc --experiment_name my_exp

# 4. Check results
ls results/my_exp/plots/
cat results/my_exp/results.json
```

## 📈 Expected Performance (CIFAR-10, tiny model, 5 epochs)

| Strategy | Avg Accuracy | Forgetting | Time (GPU) |
|----------|--------------|------------|------------|
| Naive    | ~50%         | ~30%       | 10 min     |
| EWC      | ~60%         | ~20%       | 15 min     |
| LwF      | ~62%         | ~18%       | 18 min     |
| GEM      | ~65%         | ~15%       | 20 min     |

## 💡 Tips

- ✅ Always set `--seed 42` for reproducibility
- ✅ Start with `--model_size tiny` for quick tests
- ✅ Use descriptive `--experiment_name`
- ✅ Check `plots/accuracy_matrix_final.png` first
- ✅ Run multiple seeds for statistical significance
- ✅ GPU recommended (10-50x faster than CPU)

## 🆘 Help

If issues arise:
1. Check ERROR_ANALYSIS.md
2. Run `python validate_framework.py`
3. Try minimal test with CPU
4. Check terminal logs
5. Review `results.json` errors field

## 🔗 Quick Links

- [Complete Usage Guide](USAGE_SUMMARY.md)
- [Experiment Instructions](EXPERIMENT_GUIDE.md)
- [Error Solutions](ERROR_ANALYSIS.md)
- [Project Summary](IMPLEMENTATION_SUMMARY.md)
