import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder
try:
    from torchinfo import summary
except ImportError:
    summary = None

class Classifer(nn.Module):
    """
    AST Classifier for FSD50K multi-label sound-event classification.
    Matching AST spectrogram input and 200 classes.
    """
    def __init__(
        self,
        tok_dim=192,
        num_classes=200,
        c_in=1,
        overlap=6,
        patch_size=16,
        size=(128, 1000),
        num_head=3,
        num_layer=12,
        dropout=0.1,
        use_dino=True,
        pretrained_dino=True,
        pretrained_url=None,
        use_cls_dist=True,
    ):
        super().__init__()
        self.use_cls_dist = use_cls_dist
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
            pretrained_url=pretrained_url,
        )
        self.head_norm = nn.LayerNorm(tok_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(tok_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, 128, target_frames)
        tokens = self.encoder(x)          # (B, 2 + N, tok_dim)
        if self.use_cls_dist and tokens.shape[1] >= 2:
            # Dual-token average from AST (CLS + DIST)
            pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0
        else:
            pooled = tokens[:, 2:].mean(dim=1) if tokens.shape[1] > 2 else tokens.mean(dim=1)
        pooled = self.head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.head(pooled)        # (B, num_classes)
        return logits


Classifier = Classifer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([1, 1, 128, 500], device=device)
    model = Classifer().to(device=device)
    print("Output shape:", model(x).shape)
    summary(model, [1, 1, 128, 500])