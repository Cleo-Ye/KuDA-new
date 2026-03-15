"""
SharedPath: Shared-Evidence Path. 输入 H_R_in [B, L_total, 256], alpha_r [B,1]；
内部 N 层 TransformerEncoderBlock + residual gate g_r = 1 + lambda_r * alpha_r；
末端 mean pooling -> F_R [B, 256].
"""
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class SharedPath(nn.Module):
    def __init__(self, hidden_dim=256, nhead=4, num_layers=2, dropout=0.1, lambda_r=0.5):
        super(SharedPath, self).__init__()
        self.hidden_dim = hidden_dim
        self.lambda_r = lambda_r
        layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.encoder = TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, H_R_in, alpha_r, mask=None):
        """
        H_R_in: [B, L_total, 256], alpha_r: [B, 1]
        mask: [B, L_total] 有效=1, padding=0；None 则整序列 mean
        Returns F_R: [B, 256]
        """
        x = H_R_in
        g_r = 1.0 + self.lambda_r * alpha_r   # [B, 1]
        g_r = g_r.unsqueeze(-1)   # [B, 1, 1] for broadcasting over (L, D)

        for layer in self.encoder.layers:
            delta = layer(x)
            x = x + g_r * delta

        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        if mask is not None:
            valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            F_R = (x * mask.unsqueeze(-1)).sum(dim=1) / valid_lengths
        else:
            F_R = x.mean(dim=1)
        return F_R
