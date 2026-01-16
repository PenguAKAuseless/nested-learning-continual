import torch
import torch.nn as nn
import torch.optim as optim
from models.cms import CMS

class Trainer:
    def __init__(self, model, device, learning_rate=1e-3, use_replay=False, replay_batch_size=16):
        self.model = model
        self.device = device
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.criterion = nn.CrossEntropyLoss()
        
        # --- CẢI TIẾN 1: Tinh chỉnh Optimizer theo từng Level ---
        self.optimizer = self._create_hierarchical_optimizer(model, learning_rate)

    def _create_hierarchical_optimizer(self, model, base_lr):
        """
        Tạo Optimizer với Learning Rate và Weight Decay khác nhau cho từng Level của CMS.
        Nguyên lý:
        - Level 0 (Fast): Học nhanh (High LR), Quên nhanh (High Decay) -> Plasticity
        - Level N (Slow): Học chậm (Low LR), Nhớ lâu (Low Decay) -> Stability
        """
        params_groups = []
        memo = set()

        # 1. Tách tham số của CMS
        for module_name, module in model.named_modules():
            if isinstance(module, CMS):
                for level_idx, level_module in enumerate(module.levels):
                    # Tính toán hệ số điều chỉnh dựa trên level
                    # Level càng cao -> k^i càng lớn -> Tần số cập nhật thấp
                    # Scaling factor: Giảm LR theo lũy thừa của k (hoặc căn bậc 2)
                    
                    # Giả sử k=module.k (ví dụ k=2)
                    # Level 0: scale = 1.0
                    # Level 1: scale = 0.5
                    # Level 2: scale = 0.25
                    k_factor = module.k
                    scale = 1.0 / (k_factor ** level_idx) # Hoặc 1.0 / sqrt(k**i)
                    
                    decay_scale = 1.0 
                    # Fast weights (Level 0) cần decay mạnh để tránh bias task cũ?
                    # Hoặc Slow weights cần decay mạnh? 
                    # Theo bài báo: Slow weights cần ổn định => Weight Decay thấp hơn.
                    
                    wd_scale = scale # Decay cũng giảm theo level
                    
                    level_params = []
                    for p in level_module.parameters():
                        if p not in memo:
                            level_params.append(p)
                            memo.add(p)
                    
                    if level_params:
                        params_groups.append({
                            'params': level_params,
                            'lr': base_lr * scale,
                            'weight_decay': 1e-4 * wd_scale,
                            'name': f"cms_level_{level_idx}"
                        })

        # 2. Các tham số còn lại (Backbone không phải CMS, Head, Norm...)
        backbone_params = []
        for p in model.parameters():
            if p not in memo:
                backbone_params.append(p)
        
        if backbone_params:
            params_groups.append({
                'params': backbone_params,
                'lr': base_lr, # Base LR cho các phần tĩnh
                'weight_decay': 1e-4,
                'name': "backbone_standard"
            })
            
        print(f"[Trainer] Optimizer initialized with {len(params_groups)} groups.")
        for g in params_groups:
            print(f"  - {g['name']}: lr={g['lr']:.6f}, wd={g['weight_decay']:.6f}")

        return optim.AdamW(params_groups)

    def train_task(self, train_loader, task_id, epochs=1, verbose=True):
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        print(f"Training Task {task_id} for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Replay Logic (Nếu có)
                if self.use_replay and hasattr(self.model, 'sample_from_buffer') and self.model.get_buffer_size() > 0:
                    buf_data, buf_target = self.model.sample_from_buffer(self.replay_batch_size)
                    buf_data, buf_target = buf_data.to(self.device), buf_target.to(self.device)
                    data = torch.cat([data, buf_data])
                    target = torch.cat([target, buf_target])
                
                self.optimizer.zero_grad()
                
                # --- Quan trọng: CMS forward sẽ tự update step_counter bên trong ---
                output = self.model(data)
                
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient Clipping để ổn định huấn luyện mạng sâu
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%")
        
        # Add to replay buffer after task completion (nếu model hỗ trợ)
        if self.use_replay and hasattr(self.model, 'add_to_buffer'):
            print("Updating Replay Buffer...")
            # Lấy một subset mẫu từ train_loader để lưu
            # (Code giản lược, thực tế cần lấy ngẫu nhiên)
            count = 0
            for data, target in train_loader:
                if count >= 200: break # Lưu khoảng 200 mẫu mỗi task
                self.model.add_to_buffer(data, target, task_id)
                count += len(data)

        return {'loss': avg_loss, 'accuracy': accuracy, 'history': history}