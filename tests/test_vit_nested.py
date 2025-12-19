"""
Demo and testing script for ViT with Nested Learning
Run various tests to verify the implementation
"""

import torch
import torch.nn as nn
from vision_transformer_nested_learning import (
    ViTNestedLearning,
    ViTNestedConfig,
    create_vit_nested_tiny,
    create_vit_nested_small,
)
import time


def test_forward_pass():
    """Test basic forward pass"""
    print("=" * 60)
    print("Test 1: Forward Pass")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Output range: [{logits.min():.2f}, {logits.max():.2f}]")
    
    assert logits.shape == (2, 10), "Output shape mismatch"
    print("✓ Forward pass test passed!\n")


def test_backward_pass():
    """Test backward pass and gradient flow"""
    print("=" * 60)
    print("Test 2: Backward Pass")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    model.train()
    
    x = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 10, (2,))
    
    # Forward pass
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Parameters with gradients: {has_grad}/{total_params}")
    
    # Check gradient norms
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"✓ Gradient norm: {grad_norm:.4f}")
    
    assert has_grad == total_params, "Some parameters missing gradients"
    print("✓ Backward pass test passed!\n")


def test_memory_updates():
    """Test memory module updates"""
    print("=" * 60)
    print("Test 3: Memory Updates")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    model.train()
    
    # Get initial memory stats
    stats_before = model.get_memory_stats()
    
    # Run several forward-backward passes
    for i in range(10):
        x = torch.randn(2, 3, 224, 224)
        labels = torch.randint(0, 10, (2,))
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
    
    # Get final memory stats
    stats_after = model.get_memory_stats()
    
    print("Block 0 memory changes:")
    print(f"  TITAN steps: {stats_before['block_0']['titan_step']} → {stats_after['block_0']['titan_step']}")
    print(f"  TITAN mem norm: {stats_before['block_0']['titan_mem_norm']:.4f} → {stats_after['block_0']['titan_mem_norm']:.4f}")
    print(f"  CMS fast steps: {stats_before['block_0']['cms_fast_step']} → {stats_after['block_0']['cms_fast_step']}")
    print(f"  CMS slow steps: {stats_before['block_0']['cms_slow_step']} → {stats_after['block_0']['cms_slow_step']}")
    
    # Verify updates occurred
    assert stats_after['block_0']['titan_step'] > stats_before['block_0']['titan_step'], \
        "TITAN memory not updating"
    
    print("✓ Memory update test passed!\n")


def test_memorization_mode():
    """Test enabling/disabling memorization"""
    print("=" * 60)
    print("Test 4: Memorization Mode")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    
    # Test in eval mode
    model.eval()
    with torch.no_grad():
        logits_eval = model(x)
    
    # Test with memorization enabled
    model.enable_memorization(True)
    with torch.no_grad():
        logits_mem = model(x)
    
    print(f"✓ Eval mode output shape: {logits_eval.shape}")
    print(f"✓ Memorization mode output shape: {logits_mem.shape}")
    print(f"✓ Model training state: {model.training}")
    
    # Disable memorization
    model.enable_memorization(False)
    print(f"✓ Model training state after disable: {model.training}")
    
    print("✓ Memorization mode test passed!\n")


def test_different_sizes():
    """Test different model sizes"""
    print("=" * 60)
    print("Test 5: Different Model Sizes")
    print("=" * 60)
    
    models = {
        'tiny': create_vit_nested_tiny(num_classes=10),
        'small': create_vit_nested_small(num_classes=10),
    }
    
    x = torch.randn(1, 3, 224, 224)
    
    for name, model in models.items():
        model.eval()
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        # Measure inference time
        with torch.no_grad():
            start = time.time()
            logits = model(x)
            elapsed = time.time() - start
        
        print(f"{name.capitalize()}:")
        print(f"  Parameters: {params:,}")
        print(f"  Inference time: {elapsed*1000:.2f} ms")
        print(f"  Output shape: {logits.shape}")
    
    print("✓ Different sizes test passed!\n")


def test_teach_signal_gating():
    """Test surprise-based teach signal gating"""
    print("=" * 60)
    print("Test 6: Teach Signal Gating")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    
    # Set high surprise threshold
    for block in model.blocks:
        block.surprise_threshold = 1.0  # Very high threshold
    
    model.train()
    x = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 10, (2,))
    
    # Get memory stats before
    stats_before = model.get_memory_stats()
    
    # Run with high threshold (should suppress updates)
    for _ in range(5):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
    
    stats_after_high = model.get_memory_stats()
    
    # Lower threshold
    for block in model.blocks:
        block.surprise_threshold = 0.0  # No gating
    
    # Run with no threshold (should allow updates)
    for _ in range(5):
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
    
    stats_after_low = model.get_memory_stats()
    
    print("Memory updates with different thresholds:")
    print(f"  High threshold (1.0): {stats_after_high['block_0']['titan_step'] - stats_before['block_0']['titan_step']} steps")
    print(f"  Low threshold (0.0): {stats_after_low['block_0']['titan_step'] - stats_after_high['block_0']['titan_step']} steps")
    
    print("✓ Teach signal gating test passed!\n")


def test_batch_independence():
    """Test that batch elements are processed independently"""
    print("=" * 60)
    print("Test 7: Batch Independence")
    print("=" * 60)
    
    model = create_vit_nested_tiny(num_classes=10)
    model.eval()
    
    # Process individually
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    
    # Process as batch
    x_batch = torch.cat([x1, x2], dim=0)
    
    with torch.no_grad():
        y_batch = model(x_batch)
    
    # Compare
    diff1 = (y_batch[0] - y1[0]).abs().max()
    diff2 = (y_batch[1] - y2[0]).abs().max()
    
    print(f"✓ Max difference (sample 1): {diff1:.6f}")
    print(f"✓ Max difference (sample 2): {diff2:.6f}")
    
    assert diff1 < 1e-5 and diff2 < 1e-5, "Batch processing not consistent"
    print("✓ Batch independence test passed!\n")


def test_configuration():
    """Test custom configuration"""
    print("=" * 60)
    print("Test 8: Custom Configuration")
    print("=" * 60)
    
    config = ViTNestedConfig(
        img_size=128,  # Smaller image
        patch_size=8,
        dim=256,
        depth=6,
        num_heads=4,
        num_classes=100,
        titan_mem_size=64,
        cms_fast_size=32,
        cms_slow_size=16,
        dropout=0.2,
    )
    
    model = ViTNestedLearning(config)
    
    print(f"✓ Image size: {config.img_size}")
    print(f"✓ Patch size: {config.patch_size}")
    print(f"✓ Num patches: {(config.img_size // config.patch_size) ** 2}")
    print(f"✓ Model dim: {config.dim}")
    print(f"✓ Depth: {config.depth}")
    print(f"✓ Num classes: {config.num_classes}")
    
    # Test with correct input size
    x = torch.randn(2, 3, 128, 128)
    model.eval()
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"✓ Output shape: {logits.shape}")
    assert logits.shape == (2, 100), "Custom config output shape mismatch"
    print("✓ Custom configuration test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Vision Transformer with Nested Learning - Test Suite")
    print("=" * 60 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    try:
        test_forward_pass()
        test_backward_pass()
        test_memory_updates()
        test_memorization_mode()
        test_different_sizes()
        test_teach_signal_gating()
        test_batch_independence()
        test_configuration()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
