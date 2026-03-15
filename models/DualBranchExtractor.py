"""
Dual-branch contrastive decoupling (Strategy 3).
Consistency branch: standard cross-modal attention -> F_cons.
Conflict branch: synergy-aware (S-weighted) attention -> F_conf.
Decoupling is enforced by L_diff and L_ortho during training, not by structure.
"""
import torch
from torch import nn


class DualBranchExtractor(nn.Module):
    def __init__(self, hidden_dim=256, nhead=4, dropout=0.1):
        super(DualBranchExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead

        # Consistency branch: T as query, V&A stacked as key/value
        self.cons_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.cons_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Conflict branch: same structure, query modulated by S
        self.conf_attn = nn.MultiheadAttention(
            hidden_dim, nhead, dropout=dropout, batch_first=True
        )
        self.conf_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, Z_T, Z_V, Z_A, S=None):
        """
        Z_T, Z_V, Z_A: [B, D]. S: [B] optional synergy from PID.
        Returns F_cons [B, D], F_conf [B, D].
        """
        B = Z_T.size(0)
        device = Z_T.device

        # [B, 1, D] for query; [B, 2, D] for key/value (V and A stacked)
        q_T = Z_T.unsqueeze(1)
        kv = torch.stack([Z_V, Z_A], dim=1)  # [B, 2, D]

        # Consistency branch: standard cross-attn
        cons_out, _ = self.cons_attn(q_T, kv, kv)  # [B, 1, D]
        F_cons = self.cons_proj(self.norm(cons_out.squeeze(1)))

        # Conflict branch: S-modulate query (S higher -> focus on disagreement)
        if S is not None:
            S_w = S.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            q_conf = q_T * (1.0 + S_w)
        else:
            q_conf = q_T
        conf_out, _ = self.conf_attn(q_conf, kv, kv)
        F_conf = self.conf_proj(self.norm(conf_out.squeeze(1)))

        return F_cons, F_conf
