import torch
import torch.nn as nn
from models.encoder import ASTEncoder


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
        num_head=8,
        num_layer=12,
        dropout=0.1,
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
