import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ResNet18CL(nn.Module):
    def __init__(self, num_classes=256, pretrained=False, remove_bias=False):
        super(ResNet18CL, self).__init__()
        
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )
        self.feature_dim = base_model.fc.in_features
        
        # Classifier head
        self.head = nn.Linear(self.feature_dim, num_classes, bias = not remove_bias)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: Input tensor [Batch, 3, 224, 224]
            return_features (bool): Nếu True, trả về cả features vector (trước lớp cuối).
        """
        # Feature extraction
        x = self.encoder(x)
        
        # Flatten: [Batch, 512, 1, 1] -> [Batch, 512]
        features = torch.flatten(x, 1)
        
        # Classification
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits

def get_backbone(cfg):
    if cfg['model']['architecture'] == 'resnet18':
        return ResNet18CL(
            num_classes=cfg['model']['num_classes'],
            pretrained=cfg['model']['pretrained']
        )
    else:
        raise NotImplementedError(f"Architecture {cfg['model']['architecture']} not supported yet.")