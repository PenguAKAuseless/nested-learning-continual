"""
Run comparison experiment across all three models
"""

import subprocess
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List

def run_experiment(model, num_tasks=2, epochs=3, batch_size=16, seed=42):
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
        "--learning_rate", "1e-4",
        "--seed", str(seed)
    ]
    
    # Add model-specific args
    if model == "vit_cms":
        cmd.extend(["--cms_levels", "3", "--k", "5"])
    elif model == "vit_simple":
        cmd.extend(["--head_layers", "3"])
    elif model == "vit_replay":
        cmd.extend(["--head_layers", "3", "--buffer_size", "1000"])
    elif model == "cnn_replay":
        cmd.extend(["--buffer_size", "1000"])
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nWARNING: {model} experiment failed!")
        return False
    
    print(f"\n✓ {model} experiment completed successfully!")
    return True

def generate_comparison_analysis(results_dirs: Dict[str, Path], num_tasks: int) -> Path:
    """Generate comparison tables and figures from all model results."""
    # Create comparison directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = Path('./results') / f'comparison_{timestamp}'
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results from all models
    all_results = {}
    for model_name, results_dir in results_dirs.items():
        results_file = results_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                all_results[model_name] = json.load(f)
    
    if not all_results:
        print("No results found to compare!")
        return comparison_dir
    
    # 1. Create end-task performance comparison table
    print("\nCreating performance comparison table...")
    table_data = []
    for model_name, results in all_results.items():
        final_eval = results.get('final_evaluation', {})
        for task_id in range(num_tasks):
            if task_id in final_eval:
                task_metrics = final_eval[task_id]
                table_data.append({
                    'Model': model_name,
                    'Task': task_id,
                    'Accuracy (%)': f"{task_metrics['accuracy']:.2f}",
                    'F1 Score (%)': f"{task_metrics['f1']:.2f}",
                    'Precision (%)': f"{task_metrics['precision']:.2f}",
                    'Recall (%)': f"{task_metrics['recall']:.2f}"
                })
    
    df_table = pd.DataFrame(table_data)
    
    if df_table.empty:
        print("\nNo task performance data found to compare!")
        return comparison_dir
    
    table_file = comparison_dir / 'comparison_table.csv'
    df_table.to_csv(table_file, index=False)
    
    # Print table to console
    print("\n" + "="*80)
    print("FINAL TASK PERFORMANCE COMPARISON")
    print("="*80)
    print(df_table.to_string(index=False))
    
    # Calculate and print average performance
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE ACROSS ALL TASKS")
    print("="*80)
    for model_name in all_results.keys():
        model_data = df_table[df_table['Model'] == model_name]
        if not model_data.empty:
            avg_acc = model_data['Accuracy (%)'].astype(float).mean()
            avg_f1 = model_data['F1 Score (%)'].astype(float).mean()
            print(f"{model_name:20s}: Acc={avg_acc:.2f}%, F1={avg_f1:.2f}%")
    
    # 2. Create forgetting analysis figure
    print("\nCreating forgetting analysis figure...")
    create_forgetting_figure(all_results, num_tasks, comparison_dir)
    
    return comparison_dir


def create_forgetting_figure(all_results: Dict, num_tasks: int, output_dir: Path):
    """Create figure showing how well each model remembers previous tasks."""
    fig, axes = plt.subplots(1, num_tasks, figsize=(5*num_tasks, 5), sharey=True)
    if num_tasks == 1:
        axes = [axes]
    
    colors = {'vit_cms': 'blue', 'vit_simple': 'green', 'vit_replay': 'orange', 'cnn_replay': 'red'}
    
    for task_id in range(num_tasks):
        ax = axes[task_id]
        
        for model_name, results in all_results.items():
            per_task_history = results.get('per_task_history', {})
            
            if str(task_id) in per_task_history:
                history = per_task_history[str(task_id)]
                # Extract accuracy over time
                training_steps = [h['after_training_task'] for h in history]
                accuracies = [h['accuracy'] for h in history]
                
                ax.plot(training_steps, accuracies, 
                       marker='o', label=model_name, 
                       color=colors.get(model_name, 'black'),
                       linewidth=2, markersize=8)
        
        ax.set_xlabel('After Training Task', fontsize=12)
        if task_id == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Task {task_id} Memory Retention', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(range(num_tasks))
    
    plt.suptitle('Catastrophic Forgetting Analysis: Task Performance Over Time', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_file = output_dir / 'forgetting_analysis.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved forgetting analysis figure: {fig_file}")


def main():
    """Run all comparison experiments"""
    print("="*60)
    print("CONTINUAL LEARNING MODEL COMPARISON")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    models = ["vit_cms", "vit_simple", "vit_replay", "cnn_replay"]
    num_tasks = 5  # Start with 5 tasks for testing
    epochs = 2     # 2 epochs for testing
    batch_size = 16  # Smaller batch for GPU memory safety
    seed = 42      # Fixed seed for reproducibility
    
    print(f"\nConfiguration:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Tasks: {num_tasks}")
    print(f"  Epochs per task: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Random seed: {seed}")
    print(f"\nEstimated time: ~{len(models) * num_tasks * epochs} minutes")
    print("\nPress Ctrl+C to cancel...")
    
    try:
        input("\nPress Enter to start...")
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return
    
    # Run experiments and track results directories
    results = {}
    results_dirs = {}
    for model in models:
        success = run_experiment(model, num_tasks, epochs, batch_size, seed)
        results[model] = success
        # Find the most recent results directory for this model
        if success:
            results_dir = Path('./results')
            model_dirs = sorted([d for d in results_dir.glob(f"{model}_*") if d.is_dir()], 
                              key=lambda x: x.stat().st_mtime, reverse=True)
            if model_dirs:
                results_dirs[model] = model_dirs[0]
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {model:20s}: {status}")
    
    # Generate comparison analysis if we have results
    if results_dirs:
        print("\n" + "="*60)
        print("GENERATING COMPARISON ANALYSIS")
        print("="*60)
        comparison_dir = generate_comparison_analysis(results_dirs, num_tasks)
        print(f"\nComparison analysis saved to: {comparison_dir}")
        print(f"  - comparison_table.csv: End-task performance for all models")
        print(f"  - forgetting_analysis.png: Forgetting curves for all models")
    else:
        print(f"\nCheck ./results/ directory for detailed results")
        print("Compare metrics in the results.json files")

if __name__ == "__main__":
    main()
