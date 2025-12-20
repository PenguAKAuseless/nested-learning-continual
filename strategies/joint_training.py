import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class JointTrainer:
    def __init__(self, model, device, cfg):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        opt_cfg = cfg['training']['optimizer']
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=opt_cfg['lr'], 
            momentum=opt_cfg['momentum'], 
            weight_decay=opt_cfg['weight_decay']
        )
        
        sched_cfg = cfg['training']['scheduler']
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=sched_cfg['step_size'], 
            gamma=sched_cfg['gamma']
        )

    def train(self, loader, epochs):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                
                pbar.set_postfix({'Loss': total_loss/len(loader), 'Acc': 100.*correct/total})
            
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