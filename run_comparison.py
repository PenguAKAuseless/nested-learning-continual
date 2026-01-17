"""
Run comparison experiment across ALL 4 models
Function: EXECUTION ONLY (No Visualization)
Updated: Auto-skip completed models
"""

import subprocess
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# --- CẤU HÌNH ĐỒNG BỘ ---
BACKBONE = "vit_base_patch16_224" 
BATCH_SIZE = "8"                   
CMS_K = "20"                       
CMS_LEVELS = "3"

def check_experiment_completed(model_name: str, num_tasks: int) -> Optional[Path]:
    """
    Kiểm tra xem model này đã chạy xong chưa.
    Trả về Path tới thư mục kết quả nếu đã xong, ngược lại trả về None.
    """
    results_root = Path('./results')
    if not results_root.exists():
        return None
        
    # Tìm tất cả các thư mục khớp với pattern: task_{num_tasks}_{model}_*
    pattern = f"task_{num_tasks}_{model_name}_*"
    candidates = sorted(list(results_root.glob(pattern)), key=lambda x: x.stat().st_mtime, reverse=True)
    
    for folder in candidates:
        json_path = folder / 'results.json'
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Kiểm tra xem đã chạy xong chưa (có final_evaluation)
                if 'final_evaluation' in data and data['final_evaluation']:
                    # Check config cơ bản
                    config = data.get('config', {})
                    if str(config.get('num_tasks')) == str(num_tasks):
                        return folder
            except Exception:
                continue 
    return None

def run_experiment(model, num_tasks=5, epochs=3):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Checking status for: {model}")
    
    # [SMART SKIP] Kiểm tra xem đã chạy xong chưa
    existing_result_dir = check_experiment_completed(model, num_tasks)
    if existing_result_dir:
        print(f"✅ Found completed experiment at: {existing_result_dir}")
        print(f"   Skipping training for {model}...")
        return existing_result_dir # Trả về đường dẫn đã xong
    
    print(f"Running NEW experiment: {model}")
    print(f"Config: {BACKBONE} | Batch: {BATCH_SIZE} | Tasks: {num_tasks}")
    print(f"{'='*60}\n")
    
    python_exe = sys.executable
    
    cmd = [
        python_exe,
        "run_experiment.py",
        "--model", model,
        "--num_tasks", str(num_tasks),
        "--epochs", str(epochs),
        "--batch_size", BATCH_SIZE,
        "--learning_rate", "1e-4",
        "--backbone", BACKBONE,
        "--no_checkpoint" 
    ]
    
    if model == "vit_cms":
        cmd.extend(["--cms_levels", CMS_LEVELS, "--k", CMS_K])
    elif model == "vit_simple":
        cmd.extend(["--head_layers", "3"])
    elif model == "vit_replay":
        cmd.extend(["--head_layers", "3", "--buffer_size", "500", "--replay_batch_size", BATCH_SIZE])
    elif model == "cnn_replay":
        cmd.extend(["--buffer_size", "500", "--replay_batch_size", BATCH_SIZE])
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        # Chạy lệnh thực thi
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"\nWARNING: {model} experiment failed/crashed!")
            return None
    except Exception as e:
        print(f"Error running subprocess: {e}")
        return None
    
    # Sau khi chạy xong, tìm lại thư mục vừa tạo
    # (Vì run_experiment.py tự sinh tên folder theo timestamp, ta cần tìm cái mới nhất)
    res_path = Path('./results')
    dirs = sorted([d for d in res_path.glob(f"task_{num_tasks}_{model}_*") if d.is_dir()], 
                  key=lambda x: x.stat().st_mtime, reverse=True)
    
    if dirs:
        print(f"\n✓ {model} experiment completed successfully at: {dirs[0]}")
        return dirs[0]
    else:
        print(f"\n⚠ {model} finished but result folder not found!")
        return None

def main():
    print("="*60)
    print("CONTINUAL LEARNING BENCHMARK RUNNER")
    print("Function: Execution Only (Use visualize_final.py for plotting)")
    print("="*60)
    
    models = ["vit_cms", "vit_simple", "vit_replay", "cnn_replay"]
    num_tasks = 5
    epochs = 3
    
    print(f"Models: {models}")
    print(f"Tasks: {num_tasks} | Epochs: {epochs}")
    print(f"Backbone: {BACKBONE}")

    final_paths = {}
    
    for model in models:
        result_dir = run_experiment(model, num_tasks, epochs)
        if result_dir:
            final_paths[model] = str(result_dir / "results.json")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETED. HERE ARE YOUR RESULT FILES:")
    print("="*80)
    print("Copy the dictionary below into 'visualize_final.py':\n")
    
    print("files = {")
    for model, path in final_paths.items():
        # Format tên model cho đẹp
        pretty_name = model.replace("vit_", "ViT-").replace("cnn_", "CNN-").replace("simple", "Simple").replace("replay", "Replay").replace("cms", "CMS")
        print(f"    '{pretty_name}': '{path}',")
    print("}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()