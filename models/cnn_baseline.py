"""
Simple CNN Baseline with Replay Buffer for Continual Learning
"""

import torch
import torch.nn as nn
import random
from typing import List, Tuple


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10.
    Used as baseline for comparison with ViT models.
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels (3 for RGB)
        hidden_dim: Hidden dimension scaling factor
    """
    def __init__(self, num_classes=10, input_channels=3, hidden_dim=64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate feature dimension (for 32x32 input: 32->16->8->4)
        self.feature_dim = hidden_dim * 4 * 4 * 4
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x):
        """Extract features before classification."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ReplayBuffer:
    """
    Experience Replay Buffer for continual learning.
    Stores samples from previous tasks to prevent catastrophic forgetting.
    
    Args:
        buffer_size: Maximum number of samples to store
        sampling_strategy: How to sample ('random', 'balanced')
    """
    def __init__(self, buffer_size=1000, sampling_strategy='balanced'):
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        self.buffer = []  # List of (image, label, task_id) tuples
        
    def add_samples(self, images: torch.Tensor, labels: torch.Tensor, task_id: int):
        """
        Add samples to the replay buffer.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            task_id: Current task identifier
        """
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                ))
            else:
                # Replace random sample if buffer is full
                idx = random.randint(0, self.buffer_size - 1)
                self.buffer[idx] = (
                    images[i].cpu().clone(),
                    labels[i].cpu().clone(),
                    task_id
                )
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Sample a batch from the replay buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            
        Returns:
            Tuple of (images, labels, task_ids)
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        sample_size = min(batch_size, len(self.buffer))
        
        if self.sampling_strategy == 'random':
            samples = random.sample(self.buffer, sample_size)
        elif self.sampling_strategy == 'balanced':
            # Try to balance across tasks
            task_ids = list(set([s[2] for s in self.buffer]))
            per_task = sample_size // len(task_ids)
            remainder = sample_size % len(task_ids)
            
            samples = []
            for task_id in task_ids:
                task_samples = [s for s in self.buffer if s[2] == task_id]
                n_samples = per_task + (1 if remainder > 0 else 0)
                remainder -= 1
                samples.extend(random.sample(task_samples, min(n_samples, len(task_samples))))
            
            # If we didn't get enough samples, add more random ones
            if len(samples) < sample_size:
                remaining = [s for s in self.buffer if s not in samples]
                samples.extend(random.sample(remaining, sample_size - len(samples)))
        else:
            samples = random.sample(self.buffer, sample_size)
        
        # Unpack samples
        images = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        task_ids = [s[2] for s in samples]
        
        return images, labels, task_ids
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the replay buffer."""
        self.buffer = []


class CNN_Replay(nn.Module):
    """
    CNN with integrated replay buffer for continual learning.
    
    Args:
        num_classes: Number of output classes
        buffer_size: Replay buffer size
        hidden_dim: CNN hidden dimension scaling
    """
    def __init__(self, num_classes=10, buffer_size=1000, hidden_dim=64):
        super().__init__()
        
        self.cnn = SimpleCNN(num_classes=num_classes, hidden_dim=hidden_dim)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.cnn(x)
    
    def get_features(self, x):
        return self.cnn.get_features(x)
    
    def add_to_buffer(self, images, labels, task_id):
        """Add samples to replay buffer."""
        self.replay_buffer.add_samples(images, labels, task_id)
    
    def sample_from_buffer(self, batch_size):
        """Sample from replay buffer."""
        return self.replay_buffer.sample(batch_size)
