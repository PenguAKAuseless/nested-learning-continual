import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Trainer for neural networks with continual learning support"""
    def __init__(self, model, data_loader, optimizer, criterion, device):
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        self.losses = []
        self.accuracies = []

    def train_one_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.data_loader, desc=f'Epoch {self.epoch}')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        average_loss = total_loss / len(self.data_loader)
        accuracy = 100 * correct / total
        self.losses.append(average_loss)
        self.accuracies.append(accuracy)
        return average_loss, accuracy

    def train(self, num_epochs):
        """Train the model for multiple epochs"""
        print(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.epoch += 1
            average_loss, accuracy = self.train_one_epoch()
            print(f'Epoch [{self.epoch}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
        print("Training completed!")
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        average_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        return average_loss, accuracy