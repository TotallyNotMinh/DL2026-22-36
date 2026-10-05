import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34,
    ResNet18_Weights, ResNet34_Weights
)

class ResNetEncoder(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=False, in_channels=1):
        super().__init__()
        
        # 1. Initialize backbone
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.out_dim = 512
        elif model_name == "resnet34":
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            backbone = resnet34(weights=weights)
            self.out_dim = 512
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from ['resnet18', 'resnet34']")

        # 2. Modify conv1 from 3 channels to in_channels (e.g. 1 for Spectrogram)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            if pretrained:
                with torch.no_grad():
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv

        # 3. Extract convolution layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

    def forward(self, x):
        # Input x: (B, 1, 128, 500) or (B, 1, 128, 1000)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)            # (B, out_dim, 1, 1)
        feat = torch.flatten(x, 1)     # (B, out_dim)
        return feat
