import matplotlib.pyplot as plt
import json
import os
import numpy as np
import seaborn as sns

# ==========================================
# CẤU HÌNH GIAO DIỆN (STYLE)
# ==========================================
# Sử dụng style sạch, chuyên nghiệp cho báo cáo khoa học
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 22,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'font.family': 'sans-serif',
    'figure.titlesize': 24
})

def save_plot(fig, filename):
    """Lưu ảnh tỉ lệ 16:9 (1920x1080) chất lượng cao"""
    fig.set_size_inches(16, 9)
    # Tinh chỉnh layout để không bị cắt chữ
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)

def parse_results(file_path):
    """Đọc và parse file results.json theo cấu trúc bạn cung cấp"""
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        data = json.load(f)

    history = data.get('per_task_history', {})
    
    # Xác định số lượng task dựa trên key của history (ví dụ "0", "1" -> 2 tasks)
    tasks_ids = sorted([int(k) for k in history.keys()])
    num_tasks = len(tasks_ids)
    
    # Tạo ma trận Acc: Hàng=Training_Step, Cột=Eval_Task
    # Khởi tạo bằng NaN để không vẽ những điểm chưa học
    acc_matrix = np.full((num_tasks, num_tasks), np.nan)

    for eval_task_str, records in history.items():
        eval_task = int(eval_task_str)
        for rec in records:
            train_step = rec['after_training_task']
            acc = rec['accuracy']
            
            # Chỉ điền vào ma trận nếu index hợp lệ
            if train_step < num_tasks and eval_task < num_tasks:
                acc_matrix[train_step, eval_task] = acc

    # Tính Average Accuracy tại mỗi bước (chỉ tính các task đã học đến lúc đó)
    avg_acc_list = []
    task0_acc_list = []
    
    for t in range(num_tasks):
        # Lấy dòng t: kết quả sau khi train xong task t
        # Chỉ lấy các cột từ 0 đến t (các task đã học)
        valid_accs = acc_matrix[t, :t+1]
        valid_accs = valid_accs[~np.isnan(valid_accs)] # Loại bỏ nan
        
        if len(valid_accs) > 0:
            avg_acc_list.append(np.mean(valid_accs))
        else:
            avg_acc_list.append(0)
            
        task0_acc_list.append(acc_matrix[t, 0])

    return {
        'tasks': tasks_ids,
        'avg_acc': avg_acc_list,
        'task0_acc': task0_acc_list,
        'matrix': acc_matrix
    }

def visualize_all(cms_file, baseline_file=None):
    print(f"Processing CMS results from: {cms_file}")
    cms_data = parse_results(cms_file)
    if not cms_data: return

    tasks = cms_data['tasks']
    num_tasks = len(tasks)

    # ==========================================
    # 1. FREQUENCY HEATMAP (Concept Visualization)
    # ==========================================
    # Đây là biểu đồ minh họa cơ chế hoạt động (như yêu cầu của bạn)
    fig, ax = plt.subplots()
    steps = 100
    # Level 0 (Fast): Cập nhật liên tục (dày đặc)
    level_0 = np.random.rand(steps) * 0.9 + 0.1
    # Level 1 (Mid): Cập nhật thưa hơn
    level_1 = np.zeros(steps)
    level_1[::5] = np.random.rand(20) * 0.7 + 0.2
    # Level 2 (Slow): Cập nhật rất hiếm
    level_2 = np.zeros(steps)
    level_2[::25] = np.random.rand(4) * 0.5 + 0.3
    
    heatmap_data = np.vstack([level_0, level_1, level_2])
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='magma', interpolation='nearest')
    ax.set_yticks([0, 1, 2], labels=['Level 0 (Fast Weights)', 'Level 1 (Mid Weights)', 'Level 2 (Slow Weights)'])
    ax.set_xlabel('Training Iterations (Time)')
    ax.set_title('Concept: Multi-Time Scale Update Frequencies in CMS')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Gradient Magnitude')
    
    save_plot(fig, '1_frequency_heatmap.png')

    # ==========================================
    # 2. COMPARISON (CMS vs BASELINE)
    # ==========================================
    fig, ax = plt.subplots()
    ax.plot(tasks, cms_data['avg_acc'], 'g-o', linewidth=4, markersize=12, label='ViT-CMS (Ours)')
    
    if baseline_file and os.path.exists(baseline_file):
        base_data = parse_results(baseline_file)
        if base_data:
            ax.plot(tasks, base_data['avg_acc'], 'r--s', linewidth=3, markersize=10, alpha=0.6, label='Baseline')
    
    ax.set_title('Performance Comparison: Average Accuracy', fontweight='bold')
    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(tasks)
    ax.legend()
    ax.grid(True, linestyle='--')
    
    save_plot(fig, '2_comparison_cms_vs_base.png')

    # ==========================================
    # 3. AVERAGE ACCURACY EVOLUTION
    # ==========================================
    fig, ax = plt.subplots()
    ax.plot(tasks, cms_data['avg_acc'], color='#2ca02c', linewidth=4, marker='o', markersize=12)
    ax.fill_between(tasks, cms_data['avg_acc'], alpha=0.1, color='#2ca02c')
    
    ax.set_title('Average Accuracy over Learning Steps', fontweight='bold')
    ax.set_xlabel('Task ID')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(tasks)
    
    # Annotate giá trị
    for i, v in enumerate(cms_data['avg_acc']):
        ax.annotate(f"{v:.1f}%", (tasks[i], v), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')

    save_plot(fig, '3_avg_accuracy.png')

    # ==========================================
    # 4. TASK 0 STABILITY (Forgetting Curve)
    # ==========================================
    fig, ax = plt.subplots()
    ax.plot(tasks, cms_data['task0_acc'], color='#d62728', linewidth=4, marker='s', markersize=12)
    
    ax.set_title('Stability Analysis: Task 0 Forgetting Curve', fontweight='bold')
    ax.set_xlabel('Task ID (Training Stage)')
    ax.set_ylabel('Task 0 Accuracy (%)')
    ax.set_ylim(0, 105) # Để chừa chỗ cho text 100%
    ax.set_xticks(tasks)
    
    for i, v in enumerate(cms_data['task0_acc']):
        ax.annotate(f"{v:.1f}%", (tasks[i], v), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=12, fontweight='bold')

    save_plot(fig, '4_task0_stability.png')

    # ==========================================
    # 5. PER-TASK EVOLUTION (Spaghetti Plot)
    # ==========================================
    fig, ax = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    
    for t_id in tasks:
        # Lấy lịch sử của task t_id (cột t_id trong ma trận)
        # Chỉ vẽ từ thời điểm t_id trở đi (acc_matrix[t_id:, t_id])
        acc_history = cms_data['matrix'][t_id:, t_id]
        steps = tasks[t_id:]
        
        if len(steps) > 0:
            ax.plot(steps, acc_history, 'o-', linewidth=3, label=f'Task {t_id}', color=colors[t_id])

    ax.set_title('Performance Evolution of Individual Tasks', fontweight='bold')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(-5, 105)
    ax.set_xticks(tasks)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    save_plot(fig, '5_per_task_evolution.png')

    # ==========================================
    # 6. FORGETTING MATRIX 
    # ==========================================
    fig, ax = plt.subplots()
    matrix = cms_data['matrix']
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    sns.heatmap(matrix, mask=mask, annot=True, fmt=".1f", cmap="YlGnBu", 
                vmin=0, vmax=100, square=True, linewidths=1, 
                annot_kws={"size": 16, "weight": "bold"}, cbar_kws={"label": "Accuracy (%)"}, ax=ax)
    
    ax.set_title('Backward Transfer Matrix (Forgetting Heatmap)', fontweight='bold')
    ax.set_xlabel('Evaluated Task ID')
    ax.set_ylabel('Trained Task ID')
    
    save_plot(fig, '6_forgetting_matrix.png')

if __name__ == "__main__":
    cms_results_file = "results/task_5_vit_cms_cifar10_20260116_233659/results.json" 
    baseline_results_file = None 

    print("Generating visualizations...")
    visualize_all(cms_results_file, baseline_results_file)
    print("Done.")