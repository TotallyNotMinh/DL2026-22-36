import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

class PatchEmbedder(nn.Module):
    def __init__(self, tok_dim, c_in, overlap, patch_size, size):
        super().__init__()
        (H, W) = size

        self.patch_embedder = nn.Conv2d(in_channels=c_in, 
                                        out_channels=tok_dim, 
                                        kernel_size=patch_size, 
                                        stride=(patch_size-overlap))

        self.stride = patch_size - overlap
        self.H_out = (H - patch_size) // self.stride + 1
        self.W_out = (W - patch_size) // self.stride + 1

        self.pos_embdder = nn.Parameter(torch.randn(1, tok_dim, self.H_out, self.W_out) * 0.02)
        self.positional_dropout = nn.Dropout2d(0.1)

    def _get_pos_embed(self, pos_param, h, w):
        if pos_param.shape[-2:] == (h, w):
            return pos_param
        return F.interpolate(pos_param, size=(h, w), mode="bicubic", align_corners=False)
    
    def forward(self, x):
        patch = self.patch_embedder(x)
        h, w = patch.shape[-2:]
        pe = self._get_pos_embed(self.pos_embdder, h, w)
        tok = self.positional_dropout((patch + pe)).flatten(2).transpose(1, 2)
        return tok

class TransformerEncoder(nn.Module):
    def __init__(self, tok_dim, num_head, num_layer, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=tok_dim, nhead=num_head, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

    def forward(self, x):
        return self.encoder(x)

class ASTEncoder(nn.Module):
    def __init__(self, tok_dim=768, c_in=1, overlap=6, patch_size=16, size=(128, 500), num_head=8, num_layer=12, num_class=20):
        super().__init__()
        self.patch_embedder = PatchEmbedder(tok_dim=tok_dim, c_in=c_in, overlap=overlap, patch_size=patch_size, size=size)
        self.transformer_encoder = TransformerEncoder(tok_dim=tok_dim, num_head=num_head, num_layer=num_layer)

    def forward(self, x):
        tokens = self.patch_embedder(x) # (B, 588, 768)
        features = self.transformer_encoder(tokens)
        return features

if __name__ == "__main__":
    x = torch.rand([1, 1, 128, 500], device="cuda")
    model = AST().to(device="cuda")
    summary(model, input_size=(1, 1, 128, 500))