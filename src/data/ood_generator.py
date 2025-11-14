import torch
import numpy as np
from torch.utils.data import Dataset


class OODNoiseDataset(Dataset):
    """
    Out-of-distribution (OOD) noise dataset for continual learning.
    Generates random noise images or simple patterns as OOD samples.
    """
    def __init__(self, num_samples=1000, image_size=(32, 32), num_channels=3, 
                 noise_type='gaussian', num_classes=100):
        """
        Parameters:
        - num_samples: Number of OOD samples to generate
        - image_size: Size of images (height, width)
        - num_channels: Number of image channels
        - noise_type: Type of noise ('gaussian', 'uniform', 'pattern')
        - num_classes: Number of classes (for random labels)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.noise_type = noise_type
        self.num_classes = num_classes
        
        # Pre-generate OOD samples for consistency
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        """Generate OOD samples"""
        samples = []
        
        for _ in range(self.num_samples):
            if self.noise_type == 'gaussian':
                # Gaussian noise
                image = torch.randn(self.num_channels, *self.image_size)
            elif self.noise_type == 'uniform':
                # Uniform noise
                image = torch.rand(self.num_channels, *self.image_size)
            elif self.noise_type == 'pattern':
                # Simple patterns (checkerboard, stripes, etc.)
                pattern_type = np.random.choice(['checkerboard', 'stripes', 'gradient'])
                image = self._generate_pattern(pattern_type)
            else:
                # Default to Gaussian
                image = torch.randn(self.num_channels, *self.image_size)
            
            # Normalize to reasonable range
            image = torch.clamp(image, -3, 3)
            
            # Random label (shouldn't match actual classes, but we use -1 to mark as OOD)
            label = -1  # Special label for OOD
            
            samples.append((image, label))
        
        return samples
    
    def _generate_pattern(self, pattern_type):
        """Generate simple patterns"""
        h, w = self.image_size
        
        if pattern_type == 'checkerboard':
            # Checkerboard pattern
            x = np.arange(w)
            y = np.arange(h)
            xx, yy = np.meshgrid(x, y)
            pattern = ((xx // 4 + yy // 4) % 2) * 2 - 1
            image = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).repeat(self.num_channels, 1, 1)
            
        elif pattern_type == 'stripes':
            # Vertical or horizontal stripes
            if np.random.rand() > 0.5:
                pattern = np.tile(np.sin(np.arange(w) * 0.5), (h, 1))
            else:
                pattern = np.tile(np.sin(np.arange(h) * 0.5).reshape(-1, 1), (1, w))
            image = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).repeat(self.num_channels, 1, 1)
            
        elif pattern_type == 'gradient':
            # Linear gradient
            if np.random.rand() > 0.5:
                pattern = np.linspace(-1, 1, w).reshape(1, -1).repeat(h, axis=0)
            else:
                pattern = np.linspace(-1, 1, h).reshape(-1, 1).repeat(w, axis=1)
            image = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).repeat(self.num_channels, 1, 1)
        else:
            image = torch.randn(self.num_channels, *self.image_size)
        
        return image
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]


def create_ood_dataset(num_samples=1000, image_size=(32, 32), num_channels=3, 
                       noise_type='gaussian', num_classes=100):
    """
    Factory function to create OOD dataset.
    
    Parameters:
    - num_samples: Number of OOD samples
    - image_size: Size of images
    - num_channels: Number of channels
    - noise_type: Type of noise ('gaussian', 'uniform', 'pattern', 'mixed')
    - num_classes: Number of classes
    
    Returns:
    - OODNoiseDataset instance
    """
    if noise_type == 'mixed':
        # Create mixed OOD dataset
        datasets = []
        for nt in ['gaussian', 'uniform', 'pattern']:
            datasets.append(OODNoiseDataset(
                num_samples=num_samples // 3,
                image_size=image_size,
                num_channels=num_channels,
                noise_type=nt,
                num_classes=num_classes
            ))
        # Concatenate
        from torch.utils.data import ConcatDataset
        return ConcatDataset(datasets)
    else:
        return OODNoiseDataset(
            num_samples=num_samples,
            image_size=image_size,
            num_channels=num_channels,
            noise_type=noise_type,
            num_classes=num_classes
        )
