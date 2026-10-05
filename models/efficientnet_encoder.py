import torch
import torch.nn as nn
import torchvision.models as tv_models

class EfficientNetEncoder(nn.Module):
    """
    EfficientNet (B0 or B4) backbone for audio spectrogram feature extraction.
    Adapts first conv layer to accept 1-channel (or in_channels) spectrograms
    while preserving ImageNet pretrained weights via channel averaging.
    """
    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True, in_channels: int = 1):
        super().__init__()
        model_name = model_name.lower().replace("-", "_")
        if model_name in ("efficientnet_b4", "b4"):
            weights = tv_models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            backbone = tv_models.efficientnet_b4(weights=weights)
            self.out_dim = 1792
        else:
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = tv_models.efficientnet_b0(weights=weights)
            self.out_dim = 1280
        
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

    def forward(self, x: torch.Tensor):
        # x: (B, in_channels, Freq, Time)
        return self.features(x)
