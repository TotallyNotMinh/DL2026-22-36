import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 backbone for audio spectrogram feature extraction.
    Adapts first conv layer to accept 1-channel (or in_channels) spectrograms
    while preserving ImageNet pretrained weights via channel averaging.
    """
    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        
        if in_channels != 3:
            old_conv = backbone.features[0][0]
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
            backbone.features[0][0] = new_conv

        self.features = backbone.features
        self.out_dim = 1280

    def forward(self, x: torch.Tensor):
        # x: (B, in_channels, Freq, Time)
        return self.features(x)
