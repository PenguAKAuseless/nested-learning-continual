# import matplotlib.pyplot as plt
# import json
# import os
# import numpy as np
# import seaborn as sns

# plt.style.use('seaborn-v0_8-whitegrid')
# plt.rcParams.update({
#     'font.size': 14,
#     'axes.titlesize': 20,
#     'axes.labelsize': 16,
#     'xtick.labelsize': 14,
#     'ytick.labelsize': 14,
#     'legend.fontsize': 14,
#     'font.family': 'sans-serif',
#     'lines.linewidth': 3
# })

# MODEL_COLORS = {
#     'ViT-CMS': '#2ca02c',
#     'ViT-Simple': '#d62728',
#     'ViT-Replay': '#1f77b4',
#     'CNN-Replay': '#ff7f0e'
# }

# def save_plot(fig, filename):
#     """Lưu ảnh tỉ lệ 16:9 (1920x1080) chất lượng cao"""
#     fig.set_size_inches(16, 9)
#     plt.tight_layout()
#     fig.savefig(filename, dpi=300, bbox_inches='tight')
#     print(f"Saved: {filename}")
#     plt.close(fig)

# def parse_results(file_path):
#     """Đọc file results.json và trích xuất metrics"""
#     if not file_path or not os.path.exists(file_path):
#         return None

#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
            
#         history = data.get('per_task_history', {})
#         tasks_ids = sorted([int(k) for k in history.keys()])
#         num_tasks = len(tasks_ids)
        
#         acc_matrix = np.full((num_tasks, num_tasks), np.nan)
#         for eval_task_str, records in history.items():
#             eval_task = int(eval_task_str)
#             for rec in records:
#                 train_step = rec['after_training_task']
#                 if train_step < num_tasks and eval_task < num_tasks:
#                     acc_matrix[train_step, eval_task] = rec['accuracy']

#         avg_acc_list = []
#         task0_acc_list = []
        
#         for t in range(num_tasks):
#             valid_accs = acc_matrix[t, :t+1]
#             valid_accs = valid_accs[~np.isnan(valid_accs)]
#             avg_acc_list.append(np.mean(valid_accs) if len(valid_accs) > 0 else 0)
#             task0_acc_list.append(acc_matrix[t, 0])

#         return {
#             'tasks': tasks_ids,
#             'avg_acc': avg_acc_list,
#             'task0_acc': task0_acc_list,
#             'matrix': acc_matrix
#         }
#     except Exception as e:
#         print(f"Error parsing {file_path}: {e}")
#         return None

# def visualize_multiple(model_files):
#     """
#     model_files: Dictionary {'Model Name': 'path/to/results.json'}
#     """
    
#     all_data = {}
#     for name, path in model_files.items():
#         print(f"Loading {name} from {path}...")
#         parsed = parse_results(path)
#         if parsed:
#             all_data[name] = parsed
#         else:
#             print(f"Warning: Could not load data for {name}")

#     if not all_data:
#         print("No valid data found to visualize!")
#         return

#     cms_name = next((n for n in all_data if 'CMS' in n), list(all_data.keys())[0])
#     cms_data = all_data[cms_name]
#     tasks = cms_data['tasks']
#     num_tasks = len(tasks)

#     # ==========================================
#     # 1. FREQUENCY HEATMAP
#     # ==========================================
#     fig, ax = plt.subplots()
#     steps = 100
#     level_0 = np.random.rand(steps) * 0.9 + 0.1
#     level_1 = np.zeros(steps); level_1[::5] = np.random.rand(20) * 0.7 + 0.2
#     level_2 = np.zeros(steps); level_2[::25] = np.random.rand(4) * 0.5 + 0.3
    
#     im = ax.imshow(np.vstack([level_0, level_1, level_2]), aspect='auto', cmap='magma', interpolation='nearest')
#     ax.set_yticks([0, 1, 2], labels=['Level 0 (Fast)', 'Level 1 (Mid)', 'Level 2 (Slow)'])
#     ax.set_xlabel('Training Iterations')
#     ax.set_title('Concept: Multi-Time Scale Update Frequencies in CMS', fontweight='bold')
#     plt.colorbar(im, ax=ax, label='Update Strength')
#     save_plot(fig, '1_frequency_heatmap.png')

#     # ==========================================
#     # 2. COMPARISON: AVERAGE ACCURACY
#     # ==========================================
#     fig, ax = plt.subplots()
#     for name, data in all_data.items():
#         color = MODEL_COLORS.get(name, 'gray')
#         lw = 5 if 'CMS' in name else 3
#         marker = 'o' if 'CMS' in name else 's'
#         alpha = 1.0 if 'CMS' in name else 0.7
        
#         ax.plot(data['tasks'], data['avg_acc'], marker=marker, linewidth=lw, 
#                 label=name, color=color, alpha=alpha, markersize=10)
    
#     ax.set_title('Performance Comparison: Average Accuracy', fontweight='bold')
#     ax.set_xlabel('Number of Tasks Learned')
#     ax.set_ylabel('Average Accuracy (%)')
#     ax.set_ylim(0, 100)
#     ax.set_xticks(tasks)
#     ax.legend()
#     ax.grid(True, linestyle='--')
#     save_plot(fig, '2_comparison_avg_acc.png')

#     # ==========================================
#     # 2b. COMPARISON: TASK 0 STABILITY
#     # ==========================================
#     fig, ax = plt.subplots()
#     for name, data in all_data.items():
#         color = MODEL_COLORS.get(name, 'gray')
#         lw = 5 if 'CMS' in name else 3
#         marker = 'D' if 'CMS' in name else '^'
        
#         ax.plot(data['tasks'], data['task0_acc'], marker=marker, linewidth=lw, 
#                 label=name, color=color, markersize=10)

#     ax.set_title('Stability Comparison: Task 0 Forgetting Curve', fontweight='bold')
#     ax.set_xlabel('Tasks Learned')
#     ax.set_ylabel('Task 0 Accuracy (%)')
#     ax.set_ylim(0, 105)
#     ax.set_xticks(tasks)
#     ax.legend()
#     ax.grid(True, linestyle='--')
#     save_plot(fig, '2b_comparison_task0_stability.png')

#     # ==========================================
#     # 3. DETAILED: AVG ACCURACY (CMS ONLY)
#     # ==========================================
#     fig, ax = plt.subplots()
#     color = MODEL_COLORS.get(cms_name, 'green')
#     ax.plot(tasks, cms_data['avg_acc'], color=color, linewidth=5, marker='o', markersize=12)
#     ax.fill_between(tasks, cms_data['avg_acc'], alpha=0.1, color=color)
    
#     ax.set_title(f'{cms_name}: Average Accuracy Evolution', fontweight='bold')
#     ax.set_xlabel('Task ID')
#     ax.set_ylabel('Average Accuracy (%)')
#     ax.set_ylim(0, 100)
#     ax.set_xticks(tasks)
    
#     for i, v in enumerate(cms_data['avg_acc']):
#         ax.annotate(f"{v:.1f}%", (tasks[i], v), xytext=(0, 15), 
#                     textcoords='offset points', ha='center', fontweight='bold')
#     save_plot(fig, '3_cms_avg_accuracy.png')

#     # ==========================================
#     # 4. DETAILED: TASK 0 STABILITY (CMS ONLY)
#     # ==========================================
#     fig, ax = plt.subplots()
#     color = '#d62728'
#     ax.plot(tasks, cms_data['task0_acc'], color=color, linewidth=5, marker='s', markersize=12)
    
#     ax.set_title(f'{cms_name}: Stability Analysis (Task 0)', fontweight='bold')
#     ax.set_xlabel('Task ID')
#     ax.set_ylabel('Task 0 Accuracy (%)')
#     ax.set_ylim(0, 105)
#     ax.set_xticks(tasks)
    
#     for i, v in enumerate(cms_data['task0_acc']):
#         ax.annotate(f"{v:.1f}%", (tasks[i], v), xytext=(0, 15), 
#                     textcoords='offset points', ha='center', fontweight='bold')
#     save_plot(fig, '4_cms_task0_stability.png')

#     # ==========================================
#     # 5. DETAILED: SPAGHETTI PLOT (CMS ONLY)
#     # ==========================================
#     fig, ax = plt.subplots()
#     colors = plt.cm.tab10(np.linspace(0, 1, num_tasks))
    
#     for t_id in tasks:
#         acc_history = cms_data['matrix'][t_id:, t_id]
#         steps = tasks[t_id:]
#         if len(steps) > 0:
#             ax.plot(steps, acc_history, 'o-', linewidth=3, label=f'Task {t_id}', color=colors[t_id])

#     ax.set_title(f'{cms_name}: Performance Evolution of Each Task', fontweight='bold')
#     ax.set_xlabel('Training Step')
#     ax.set_ylabel('Accuracy (%)')
#     ax.set_ylim(-5, 105)
#     ax.set_xticks(tasks)
#     ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     ax.grid(True, alpha=0.5)
#     save_plot(fig, '5_cms_per_task_evolution.png')

#     # ==========================================
#     # 6. DETAILED: MATRIX HEATMAP (CMS ONLY)
#     # ==========================================
#     fig, ax = plt.subplots()
#     matrix = cms_data['matrix']
#     mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
#     sns.heatmap(matrix, mask=mask, annot=True, fmt=".1f", cmap="YlGnBu", 
#                 vmin=0, vmax=100, square=True, linewidths=1, 
#                 annot_kws={"size": 16, "weight": "bold"}, 
#                 cbar_kws={"label": "Accuracy (%)"}, ax=ax)
    
#     ax.set_title(f'{cms_name}: Backward Transfer Matrix', fontweight='bold')
#     ax.set_xlabel('Evaluated Task ID')
#     ax.set_ylabel('Trained Task ID')
#     save_plot(fig, '6_cms_forgetting_matrix.png')

# if __name__ == "__main__":
    
#     base_dir = "results"
#     files = {
#         'ViT-CMS':    "results/task_5_vit_cms_cifar10_20260117_035349/results.json",
#         'ViT-Simple': "results/task_5_vit_simple_cifar10_20260117_051009/results.json",
#         'ViT-Replay': "results/task_5_vit_replay_cifar10_20260117_062150/results.json",
#         'CNN-Replay': "results/task_5_cnn_replay_cifar10_20260117_062454/results.json"
#     }

#     valid_files = {k: v for k, v in files.items() if v and os.path.exists(v)}
    
#     print(f"Found {len(valid_files)} valid result files.")
#     if len(valid_files) < 4:
#         print("Make sure to update the paths in 'files' dictionary at the bottom of the script.")

#     print("Generating visualizations...")
#     visualize_multiple(valid_files)
#     print("Done! Images saved in current directory.")
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# --- CẤU HÌNH GIAO DIỆN ĐẸP ---
# --- CẤU HÌNH GIAO DIỆN ĐẸP (ĐÃ SỬA LỖI) ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'lines.linewidth': 4,
    'lines.markersize': 10,  # <--- SỬA DÒNG NÀY (Thêm 'lines.')
    'font.family': 'sans-serif'
})

# Màu sắc chuẩn
COLORS = {
    'ViT-CMS': '#2ca02c',    # Xanh lá (Ours)
    'ViT-Simple': '#d62728', # Đỏ (Baseline)
    'ViT-Replay': '#1f77b4', # Xanh dương
    'CNN-Replay': '#ff7f0e'  # Cam
}

MARKERS = {
    'ViT-CMS': 'o',
    'ViT-Simple': 's',
    'ViT-Replay': '^',
    'CNN-Replay': 'D'
}

def parse_data(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        history = data.get('per_task_history', {})
        tasks = sorted([int(k) for k in history.keys()])
        num_tasks = len(tasks)
        
        # Mảng lưu Average F1 qua từng giai đoạn
        avg_f1_list = []
        # Mảng lưu Accuracy riêng của Task 0 qua từng giai đoạn
        task0_acc_list = []

        for t in range(num_tasks):
            # Tính Avg F1 tại thời điểm t
            f1_sum = 0
            count = 0
            # Lấy Acc của Task 0 tại thời điểm t
            t0_acc = 0

            # Duyệt qua các task đã học (từ 0 đến t)
            for eval_task_id in range(t + 1):
                records = history.get(str(eval_task_id), [])
                # Tìm record tương ứng với việc vừa train xong task t
                rec = next((r for r in records if r['after_training_task'] == t), None)
                
                if rec:
                    # Lấy F1 (nếu ko có key f1 thì lấy acc tạm)
                    val = rec.get('f1_score', rec.get('accuracy', 0))
                    f1_sum += val
                    count += 1
                    
                    # Nếu đang check Task 0, lưu lại Acc của nó
                    if eval_task_id == 0:
                        t0_acc = rec.get('accuracy', 0)

            avg_f1 = f1_sum / count if count > 0 else 0
            avg_f1_list.append(avg_f1)
            task0_acc_list.append(t0_acc)
            
        return tasks, avg_f1_list, task0_acc_list
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

def plot_chart(data_dict, metric_idx, title, ylabel, filename, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for name, vals in data_dict.items():
        tasks, y_vals = vals[0], vals[metric_idx]
        
        # Làm nổi bật ViT-CMS
        alpha = 1.0 if 'CMS' in name else 0.7
        ls = '-' if 'CMS' in name else '--'
        zorder = 10 if 'CMS' in name else 1
        
        ax.plot(tasks, y_vals, label=name, color=COLORS.get(name, 'black'),
                marker=MARKERS.get(name, 'o'), linestyle=ls, alpha=alpha, zorder=zorder)

        # Annotate số liệu cuối cùng
        last_y = y_vals[-1]
        ax.annotate(f"{last_y:.1f}", (tasks[-1], last_y), 
                    xytext=(10, 0), textcoords='offset points', 
                    color=COLORS.get(name, 'black'), fontweight='bold')

    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel(ylabel)
    ax.set_xticks(tasks)
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        # Tự động zoom vào vùng có dữ liệu để thấy sự khác biệt
        ax.set_ylim(40, 100) 

    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Đã lưu: {filename}")
    plt.close()

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # SỬA ĐƯỜNG DẪN Ở ĐÂY CHO ĐÚNG FILE CỦA BẠN
    files = {
        'ViT-CMS':    "results/task_5_vit_cms_cifar10_20260117_035349/results.json",
        'ViT-Simple': "results/task_5_vit_simple_cifar10_20260117_051009/results.json",
        'ViT-Replay': "results/task_5_vit_replay_cifar10_20260117_062150/results.json",
        'CNN-Replay': "results/task_5_cnn_replay_cifar10_20260117_062454/results.json"
    }

    parsed_data = {}
    for name, path in files.items():
        res = parse_data(path)
        if res:
            parsed_data[name] = res # (tasks, avg_f1, task0_acc)

    if parsed_data:
        # 1. Vẽ Average F1 Score (Theo yêu cầu)
        # index 1 là avg_f1
        plot_chart(parsed_data, 1, 
                   "Performance Comparison: Average F1-Score", 
                   "Average F1-Score (%)", 
                   "comparison_avg_f1.png", ylim=(50, 100))

        # 2. Vẽ Task 0 Forgetting (Cái này mới thấy khác biệt rõ)
        # index 2 là task0_acc
        plot_chart(parsed_data, 2, 
                   "Stability Analysis: Forgetting on Task 0", 
                   "Task 0 Accuracy (%)", 
                   "comparison_task0_forgetting.png", ylim=(0, 105))
    else:
        print("Không tìm thấy dữ liệu hợp lệ!")