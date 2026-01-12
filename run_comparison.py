"""
Run comparison experiment across all three models
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

def run_experiment(model, num_tasks=2, epochs=3, batch_size=16):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {model}")
    print(f"{'='*60}\n")
    
    # Get Python executable from current environment
    python_exe = sys.executable
    
    cmd = [
        python_exe,
        "run_experiment.py",
        "--model", model,
        "--num_tasks", str(num_tasks),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", "1e-4"
    ]
    
    # Add model-specific args
    if model == "vit_cms":
        cmd.extend(["--cms_levels", "3", "--k", "2"])
    elif model == "vit_simple":
        cmd.extend(["--head_layers", "2"])
    elif model == "cnn_replay":
        cmd.extend(["--buffer_size", "1000"])
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nWARNING: {model} experiment failed!")
        return False
    
    print(f"\n✓ {model} experiment completed successfully!")
    return True

def main():
    """Run all comparison experiments"""
    print("="*60)
    print("CONTINUAL LEARNING MODEL COMPARISON")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    models = ["vit_cms", "vit_simple", "cnn_replay"]
    num_tasks = 2  # Start with 2 tasks for testing
    epochs = 3     # 3 epochs for quick testing
    batch_size = 16  # Smaller batch for GPU memory safety
    
    print(f"\nConfiguration:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Tasks: {num_tasks}")
    print(f"  Epochs per task: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"\nEstimated time: ~{len(models) * num_tasks * epochs} minutes")
    print("\nPress Ctrl+C to cancel...")
    
    try:
        input("\nPress Enter to start...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return
    
    # Run experiments
    results = {}
    for model in models:
        success = run_experiment(model, num_tasks, epochs, batch_size)
        results[model] = success
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {model:20s}: {status}")
    
    print(f"\nCheck ./results/ directory for detailed results")
    print("Compare metrics in the results.json files")

if __name__ == "__main__":
    main()
