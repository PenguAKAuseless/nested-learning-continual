import subprocess
import sys

# --- CẤU HÌNH CHUNG ---
cmd_template = [
    sys.executable, "run_experiment.py",
    "--num_tasks", "2",           # Chạy 2 task để test chuyển task
    "--epochs", "1",              # 1 epoch cho nhanh
    "--batch_size", "4",          # Batch nhỏ
    "--backbone", "vit_tiny_patch16_224", # Dùng Tiny cho nhẹ
    "--no_checkpoint"
]

print("=== BẮT ĐẦU KIỂM TRA LỖI (DRY RUN) ===")
print("Mục tiêu: Test xem ViT_CMS có chạy được cùng Replay Buffer không.\n")

# --- 1. TEST ƯU TIÊN: ViT + CMS + REPLAY BUFFER ---
print(f"{'='*60}")
print("🔥 TEST 1 (QUAN TRỌNG NHẤT): ViT + CMS + Replay Buffer")
print(f"{'='*60}")

cmd_hybrid = cmd_template + [
    "--model", "vit_cms",
    "--cms_levels", "2", 
    "--k", "2",
    # Thêm tham số Buffer vào model CMS
    "--buffer_size", "10", 
    "--replay_batch_size", "4" 
]

try:
    subprocess.run(cmd_hybrid, check=True)
    print("✅ TEST 1 PASSED: ViT + CMS + Replay chạy ngon lành!\n")
except subprocess.CalledProcessError:
    print("\n❌ TEST 1 FAILED: ViT_CMS không chạy được với Replay Buffer.")
    print("   -> Khả năng cao là class ViT_CMS thiếu hàm 'add_to_buffer' hoặc 'sample_from_buffer'.")
    print("   -> Nếu bạn muốn fix, hãy copy các hàm buffer từ cnn_baseline.py sang vit_cms.py.")
    sys.exit(1) # Dừng ngay lập tức


# --- 2. CÁC MODEL CÒN LẠI ---
other_models = ["cnn_replay", "vit_replay", "vit_simple"] # Đã bỏ vit_cms thường vì test ở trên rồi

for model in other_models:
    print(f"{'='*60}")
    print(f"TEST TIẾP THEO: {model}")
    print(f"{'='*60}")
    
    cmd = cmd_template + ["--model", model]
    
    # Cấu hình riêng
    if "replay" in model:
        cmd.extend(["--buffer_size", "10", "--replay_batch_size", "4"])
    # vit_simple không cần thêm gì đặc biệt

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Model {model}: OK!\n")
    except subprocess.CalledProcessError:
        print(f"❌ Model {model}: GẶP LỖI!")
        sys.exit(1)

print("\n" + "="*60)
print("🎉 CHÚC MỪNG: TẤT CẢ CÁC MODEL ĐỀU KHÔNG CÓ LỖI CODE!")
print("="*60)