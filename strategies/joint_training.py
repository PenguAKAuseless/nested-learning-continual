import torch
import torch.nn as nn
import torch.optim as optim
# LƯU Ý: torch.amp là API mới (PyTorch 2.4+). 
# Nếu bạn dùng bản cũ hơn, hãy dùng: from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast 
from tqdm.auto import tqdm
import os

class JointTrainer:
    def __init__(self, model, device, cfg):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        
        opt_cfg = cfg['training']['optimizer']
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=opt_cfg['lr'], 
            momentum=opt_cfg['momentum'], 
            weight_decay=opt_cfg['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['training']['epochs_per_task']
        )
        self.scaler = GradScaler()
        
    def train(self, loader, epochs):
        self.model.train()

        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                with autocast(device_type=device_type, dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                
                pbar.set_postfix({'Loss': total_loss/(pbar.n+1), 'Acc': 100.*correct/total})
            
            self.scheduler.step()

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        
        return 100. * correct / total if total > 0 else 0.0

    def save_checkpoint(self, task_id, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"checkpoint_task_{task_id}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")