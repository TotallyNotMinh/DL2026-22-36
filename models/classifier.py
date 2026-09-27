import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder
from torchinfo import summary

class Classifer(nn.Module):
    """
    AST Classifier for FSD50K multi-label sound-event classification.
    Initial placeholder implementation matching (128, 500) AST spectrogram input and 200 classes.
    Can be replaced or extended as needed.
    """
    def __init__(
        self,
        tok_dim=768,
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 500),
        num_head=12,
        num_layer=12,
        dropout=0.1,
        use_dino=True,
        pretrained_dino=True,
    ):
        super().__init__()
        self.encoder = ASTEncoder(
            tok_dim=tok_dim,
            c_in=c_in,
            overlap=overlap,
            patch_size=patch_size,
            size=size,
            num_head=num_head,
            num_layer=num_layer,
            use_dino=use_dino,
            pretrained_dino=pretrained_dino,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(tok_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, 500)
        tokens = self.encoder(x)          # (B, 588, 768)
        pooled = tokens.mean(dim=1)       # (B, 768) Global Average Pooling
        pooled = self.dropout(pooled)
        logits = self.head(pooled)        # (B, num_classes)
        return logits


if __name__ == "__main__":
    x = torch.rand([1, 1, 128, 500], device="cuda")
    model = Classifer().to(device="cuda")
    print(model(x).shape)
    summary(model, [1, 1, 128, 500])