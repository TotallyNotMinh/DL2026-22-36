import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from models.encoder import ASTEncoder
from models.resnet_encoder import ResNetEncoder
from models.efficientnet_encoder import EfficientNetEncoder

try:
    from torchinfo import summary
except ImportError:
    summary = None


class AttentionPool(nn.Module):
    """Learned softmax-weighted sum over patch tokens (excludes CLS/DIST)."""
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                     # x: (B, N, D) patch tokens only
        w = self.score(x).softmax(dim=1)      # (B, N, 1)
        return (w * x).sum(dim=1)             # (B, D)


class Classifer(nn.Module):
    """
    Unified Audio Classifier for FSD50K multi-label sound-event classification.
    Supports AST (ViT-Tiny/Small/Base), ResNet (18/34), and EfficientNet-B0 backbones.
    Default initialization produces standard AST with (CLS+DIST)/2 dual-token pooling.
    """
    def __init__(
        self,
        encoder_type="ast",
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
        pooling=None,
    ):
        super().__init__()
        self.encoder_type = encoder_type.lower() if isinstance(encoder_type, str) else "ast"
        self.use_cls_dist = use_cls_dist
        self.pooling = pooling

        if self.encoder_type == "ast":
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
            if pooling == "attention":
                self.pool = AttentionPool(tok_dim)
            head_in = 2 * tok_dim if pooling == "gap_max" else tok_dim
            self.head_norm = nn.LayerNorm(head_in, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(head_in, num_classes)

        elif self.encoder_type in ("resnet18", "resnet34"):
            self.encoder = ResNetEncoder(
                model_name=self.encoder_type,
                pretrained=pretrained_dino,
                in_channels=c_in,
            )
            feat_dim = self.encoder.out_dim
            self.head_norm = None
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(feat_dim, num_classes)

        elif self.encoder_type in ("efficientnet", "efficientnet_b0", "efficientnet-b0"):
            self.encoder = EfficientNetEncoder(
                pretrained=pretrained_dino,
                in_channels=c_in,
            )
            feat_dim = self.encoder.out_dim
            self.head_norm = nn.LayerNorm(feat_dim, eps=1e-6)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(feat_dim, num_classes)
        else:
            raise ValueError(f"Unknown encoder_type: '{encoder_type}'. Choose from ['ast', 'resnet18', 'resnet34', 'efficientnet_b0']")

    def forward(self, x, return_features: bool = False):
        # x: (B, 1, 128, target_frames)
        if self.encoder_type == "ast":
            tokens = self.encoder(x)          # (B, 2 + N, tok_dim)
            if self.pooling in ("gap", "gap_max", "attention"):
                patch_tokens = tokens[:, 2:]  # (B, N, tok_dim), CLS/DIST dropped
                if self.pooling == "gap":
                    pooled = patch_tokens.mean(dim=1)
                elif self.pooling == "gap_max":
                    pooled = torch.cat([patch_tokens.mean(dim=1), patch_tokens.amax(dim=1)], dim=-1)
                else:  # "attention"
                    pooled = self.pool(patch_tokens)
            elif self.use_cls_dist and tokens.shape[1] >= 2:
                # Dual-token average from AST (CLS + DIST)
                pooled = (tokens[:, 0] + tokens[:, 1]) / 2.0
            else:
                pooled = tokens[:, 2:].mean(dim=1) if tokens.shape[1] > 2 else tokens.mean(dim=1)

            pooled = self.head_norm(pooled)
            features = pooled
            pooled = self.dropout(pooled)
            logits = self.head(pooled)

        elif self.encoder_type in ("resnet18", "resnet34"):
            pooled = self.encoder(x)          # (B, 512)
            features = pooled
            pooled = self.dropout(pooled)
            logits = self.head(pooled)

        elif self.encoder_type in ("efficientnet", "efficientnet_b0", "efficientnet-b0"):
            feat_map = self.encoder(x)        # (B, 1280, H, W)
            pooled = feat_map.mean(dim=(-2, -1))
            pooled = self.head_norm(pooled)
            features = pooled
            pooled = self.dropout(pooled)
            logits = self.head(pooled)

        if return_features:
            return logits, features
        return logits


# Alias for backward compatibility
Classifier = Classifer


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand([2, 1, 128, 500], device=device)
    
    # Test AST default
    m_ast = Classifer(pretrained_dino=False).to(device)
    print("AST default logits shape:", m_ast(x).shape)

    # Test AST pooling variants
    for p in ["gap", "gap_max", "attention"]:
        m_pool = Classifer(pooling=p, pretrained_dino=False).to(device)
        print(f"AST pooling={p} shape:", m_pool(x).shape)

    # Test ResNet34
    m_res = Classifer(encoder_type="resnet34", pretrained_dino=False).to(device)
    print("ResNet-34 logits shape:", m_res(x).shape)

    # Test EfficientNet
    m_eff = Classifer(encoder_type="efficientnet_b0", pretrained_dino=False).to(device)
    print("EfficientNet-B0 logits shape:", m_eff(x).shape)