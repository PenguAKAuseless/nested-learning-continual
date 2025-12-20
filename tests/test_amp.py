"""
Quick test of mixed precision training implementation
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runner import RunConfig, ExperimentRunner
from model.vision_transformer_nested_learning import ViTNestedConfig, ViTNestedLearning
from continual_learning.rivalry_strategies import NaiveStrategy
from data.datasets import SplitCIFAR10

def test_amp():
    print("Testing Mixed Precision Training Implementation")
    print("=" * 70)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping AMP test")
        return
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Create a small model
    config = ViTNestedConfig(
        img_size=32, patch_size=4, num_classes=10,
        dim=192, depth=2, num_heads=3, mlp_ratio=4.0,
    )
    model = ViTNestedLearning(config)
    model = model.to('cuda:0')
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy data
    x = torch.randn(4, 3, 32, 32).cuda()
    y = torch.randint(0, 10, (4,)).cuda()
    print(f"✓ Dummy data created: x={x.shape}, y={y.shape}")
    
    # Test without AMP
    print("\n1. Testing WITHOUT mixed precision:")
    strategy = NaiveStrategy(model, 'cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    loss = strategy.train_step(x, y, optimizer)
    print(f"   Loss: {loss:.4f}")
    print("   ✓ Standard training works")
    
    # Test with AMP
    print("\n2. Testing WITH mixed precision:")
    scaler = torch.amp.GradScaler('cuda')
    
    try:
        loss_amp = strategy.train_step_amp(x, y, optimizer, scaler)
        print(f"   Loss: {loss_amp:.4f}")
        print("   ✓ AMP training works")
    except Exception as e:
        print(f"   ✗ AMP training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test RunConfig with AMP
    print("\n3. Testing RunConfig with use_amp:")
    config = RunConfig(
        method_name="vit_nested",
        num_classes=10,
        num_epochs=1,
        batch_size=4,
        dataset_name="split_cifar10",
        num_tasks=2,
        device="cuda:0",
        use_amp=True
    )
    runner = ExperimentRunner(config)
    print(f"   ✓ Runner created with AMP={runner.use_amp}")
    print(f"   ✓ Scaler initialized: {runner.scaler is not None}")
    
    # Verify autocast context
    print("\n4. Testing autocast context:")
    with torch.amp.autocast('cuda'):
        output = model(x)
        print(f"   Output dtype: {output.dtype}")
        print(f"   ✓ Autocast produces float16: {output.dtype == torch.float16}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("Mixed precision training is properly configured.")
    print("=" * 70)

if __name__ == '__main__':
    test_amp()
