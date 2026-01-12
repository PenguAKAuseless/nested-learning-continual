"""
Training Pipeline for Continual Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm


class Trainer:
    """
    Trainer for continual learning experiments.
    
    Args:
        model: The model to train
        device: Device to use (cuda/cpu)
        learning_rate: Learning rate
        use_replay: Whether to use replay buffer (for CNN_Replay)
        replay_batch_size: Batch size for replay samples
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        use_replay: bool = False,
        replay_batch_size: int = 32
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.use_replay = use_replay
        self.replay_batch_size = replay_batch_size
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
    def train_task(
        self,
        train_loader: DataLoader,
        task_id: int,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train on a single task.
        
        Args:
            train_loader: DataLoader for the task
            task_id: Task identifier
            epochs: Number of epochs to train
            verbose: Print progress
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            pbar = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}/{epochs}") if verbose else train_loader
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Add replay samples if enabled
                if self.use_replay and hasattr(self.model, 'sample_from_buffer'):
                    replay_images, replay_labels, _ = self.model.sample_from_buffer(self.replay_batch_size)
                    if replay_images is not None:
                        replay_images = replay_images.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_outputs = self.model(replay_images)
                        replay_loss = self.criterion(replay_outputs, replay_labels)
                        loss = loss + replay_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Add to replay buffer if enabled
                if self.use_replay and hasattr(self.model, 'add_to_buffer'):
                    # Add subset of current batch to buffer
                    if batch_idx % 5 == 0:  # Add every 5th batch
                        self.model.add_to_buffer(images, labels, task_id)
                
                # Track metrics
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                
                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_samples += labels.size(0)
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100. * correct / labels.size(0):.2f}%'
                    })
            
            # Epoch summary
            epoch_acc = 100. * epoch_correct / epoch_samples
            if verbose:
                print(f"Task {task_id} Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Acc={epoch_acc:.2f}%")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # Return metrics
        avg_loss = total_loss / (epochs * len(train_loader))
        avg_acc = 100. * total_correct / total_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
    
    def set_learning_rate(self, lr: float):
        """Update learning rate."""
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
