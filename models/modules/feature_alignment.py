"""
FeatureAlignment: 接收编码器序列输出，保留序列长度，将 T/V/A 统一投影到 hidden_dim。
输入: H_T_raw [B, L_T, 768], H_V_raw [B, L_V, 128], H_A_raw [B, L_A, 128]
输出: H_T, H_V, H_A [B, L_*, hidden_dim]
"""
import torch
from torch import nn


class FeatureAlignment(nn.Module):
    def __init__(self, dim_t=768, dim_v=128, dim_a=128, hidden_dim=256):
        super(FeatureAlignment, self).__init__()
        self.proj_T = nn.Linear(dim_t, hidden_dim)
        self.proj_V = nn.Linear(dim_v, hidden_dim)
        self.proj_A = nn.Linear(dim_a, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, H_T_raw, H_V_raw, H_A_raw):
        """
        H_T_raw: [B, L_T, 768], H_V_raw: [B, L_V, 128], H_A_raw: [B, L_A, 128]
        Returns H_T, H_V, H_A: [B, L_T, d], [B, L_V, d], [B, L_A, d] with d=hidden_dim
        """
        H_T = self.proj_T(H_T_raw)   # [B, L_T, 256]
        H_V = self.proj_V(H_V_raw)   # [B, L_V, 256]
        H_A = self.proj_A(H_A_raw)   # [B, L_A, 256]
        return H_T, H_V, H_A
