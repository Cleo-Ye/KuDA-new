"""
CrossAttentionBlock: 轻量级 cross-attention 块，供 JointGainPath 的 TA/TV/AV 三路复用。
输入 query [B, L_q, d], context [B, L_k, d] -> 输出 [B, L_q, d].
"""
import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, use_ffn=True):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )
            self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        query: [B, L_q, d], context: [B, L_k, d]
        Returns: [B, L_q, d]
        """
        # cross-attn: query from query, key/value from context
        attn_out, _ = self.cross_attn(query, context, context)  # [B, L_q, d]
        query = self.norm1(query + self.dropout(attn_out))
        if self.use_ffn:
            query = query + self.ffn(query)
            query = self.norm2(query)
        return query
