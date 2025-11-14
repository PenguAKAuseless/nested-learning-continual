import torch
import torch.nn as nn
import torch.nn.functional as F


class NestedBlock(nn.Module):
    """A nested block with residual connections for hierarchical feature learning"""
    def __init__(self, in_channels, out_channels, depth=2):
        super(NestedBlock, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        
        # Create nested layers
        for i in range(depth):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.residual(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        return F.relu(out + identity)


class NestedNetwork(nn.Module):
    """Nested Learning Network for Continual Learning"""
    def __init__(self, input_channels=3, num_classes=100, base_channels=64):
        super(NestedNetwork, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Nested blocks with increasing depth for hierarchical learning
        self.nested_block1 = NestedBlock(base_channels, base_channels, depth=2)
        self.nested_block2 = NestedBlock(base_channels, base_channels * 2, depth=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.nested_block3 = NestedBlock(base_channels * 2, base_channels * 4, depth=3)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.nested_block4 = NestedBlock(base_channels * 4, base_channels * 8, depth=2)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Adaptive pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.nested_block1(x)
        x = self.nested_block2(x)
        x = self.pool2(x)
        
        x = self.nested_block3(x)
        x = self.pool3(x)
        
        x = self.nested_block4(x)
        x = self.pool4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_features(self, x):
        """Extract features for continual learning tasks"""
        x = self.conv1(x)
        x = self.nested_block1(x)
        x = self.nested_block2(x)
        x = self.pool2(x)
        x = self.nested_block3(x)
        x = self.pool3(x)
        x = self.nested_block4(x)
        x = self.pool4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x