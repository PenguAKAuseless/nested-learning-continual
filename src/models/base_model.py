import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base model class for continual learning models"""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.model = None

    def forward(self, x):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Forward method must be implemented in subclasses.")

    def get_features(self, x):
        """Extract features - to be implemented by subclasses"""
        raise NotImplementedError("Get features method must be implemented in subclasses.")
    
    def freeze_layers(self, num_layers):
        """Freeze first num_layers for transfer learning"""
        count = 0
        for name, param in self.named_parameters():
            if count < num_layers:
                param.requires_grad = False
                count += 1
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params