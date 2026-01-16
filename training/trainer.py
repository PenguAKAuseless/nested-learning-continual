import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, device, optimizer=None, learning_rate=1e-3, use_replay=False, replay_batch_size=16):
        """
        Args:
            model: PyTorch model
            device: Computing device (cpu/cuda)
            optimizer: Passed optimizer (can be CMSOptimizerWrapper). If None, creates a default AdamW.
            learning_rate: Used only if optimizer is None.
            use_replay: Enable experience replay
            replay_batch_size: Number of replay samples per batch
        """
        self.model = model
        self.device = device
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        self.criterion = nn.CrossEntropyLoss()
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    def train_task(self, train_loader, task_id, epochs=1, verbose=True):
        self.model.train()
        history = {'loss': [], 'accuracy': []}
        
        if verbose:
            print(f"Training Task {task_id} for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Replay Logic
                if self.use_replay and hasattr(self.model, 'sample_from_buffer') and self.model.get_buffer_size() > 0:
                    try:
                        buf_data, buf_target = self.model.sample_from_buffer(self.replay_batch_size)
                        buf_data, buf_target = buf_data.to(self.device), buf_target.to(self.device)
                        data = torch.cat([data, buf_data])
                        target = torch.cat([target, buf_target])
                    except ValueError:
                        pass # Bỏ qua nếu buffer chưa đủ dữ liệu
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Gradient Clipping (khuyên dùng cho mạng sâu/CMS)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Step (CMSOptimizerWrapper sẽ tự xử lý việc chặn cập nhật các tầng chậm)
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
        
        # Add to replay buffer after task completion
        if self.use_replay and hasattr(self.model, 'add_to_buffer'):
            print("Updating Replay Buffer...")
            count = 0
            self.model.eval() # Chuyển sang eval để tránh ảnh hưởng batchnorm khi lấy mẫu
            with torch.no_grad():
                for data, target in train_loader:
                    if count >= 200: break
                    self.model.add_to_buffer(data, target, task_id)
                    count += len(data)
            self.model.train()

        return {'loss': avg_loss, 'accuracy': accuracy, 'history': history}