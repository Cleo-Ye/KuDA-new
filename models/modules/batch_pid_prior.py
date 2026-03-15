"""
BatchPIDPrior: batch-level instantaneous prior from 6 probe heads (3 unimodal, 3 bimodal),
EMA-updated r_global, s_global. Input: H_T, H_V, H_A [B, L, 256], labels.
Probe input uses detach() of pooled Z_* to avoid coupling backbone.
"""
import torch
import torch.nn.functional as F
from torch import nn


class BatchPIDPrior(nn.Module):
    def __init__(self, hidden_dim=256, ema_beta=0.9, opt=None):
        super(BatchPIDPrior, self).__init__()
        self.hidden_dim = hidden_dim
        self.ema_beta = ema_beta if opt is None else getattr(opt, 'pid_prior_ema_beta', 0.9)
        # Unimodal probe heads
        self.head_T = nn.Linear(hidden_dim, 1)
        self.head_A = nn.Linear(hidden_dim, 1)
        self.head_V = nn.Linear(hidden_dim, 1)
        # Bimodal probe heads (concat 2 * hidden_dim -> 1)
        self.head_TA = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.head_TV = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.head_AV = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.register_buffer('r_global_buf', torch.tensor(0.5))
        self.register_buffer('s_global_buf', torch.tensor(0.5))

    def forward(self, H_T, H_V, H_A, labels):
        """
        H_T, H_V, H_A: [B, L_*, 256]. labels: [B, 1] regression target.
        Returns r_global [B,1], s_global [B,1], aux_pid_loss scalar.
        """
        B = H_T.size(0)
        device = H_T.device
        # Pool and detach for probe
        Z_T = H_T.mean(dim=1).detach()   # [B, 256]
        Z_A = H_A.mean(dim=1).detach()
        Z_V = H_V.mean(dim=1).detach()

        pred_T = self.head_T(Z_T).squeeze(-1)   # [B]
        pred_A = self.head_A(Z_A).squeeze(-1)
        pred_V = self.head_V(Z_V).squeeze(-1)
        pred_TA = self.head_TA(torch.cat([Z_T, Z_A], dim=-1)).squeeze(-1)
        pred_TV = self.head_TV(torch.cat([Z_T, Z_V], dim=-1)).squeeze(-1)
        pred_AV = self.head_AV(torch.cat([Z_A, Z_V], dim=-1)).squeeze(-1)

        label_flat = labels.view(-1)
        L_T = F.l1_loss(pred_T, label_flat, reduction='mean')
        L_A = F.l1_loss(pred_A, label_flat, reduction='mean')
        L_V = F.l1_loss(pred_V, label_flat, reduction='mean')
        L_TA = F.l1_loss(pred_TA, label_flat, reduction='mean')
        L_TV = F.l1_loss(pred_TV, label_flat, reduction='mean')
        L_AV = F.l1_loss(pred_AV, label_flat, reduction='mean')

        aux_pid_loss = (L_T + L_A + L_V + L_TA + L_TV + L_AV) / 6.0

        # MMI-PID 严格公式 (论文 3.3): I ≈ -L; R = min(I_i, I_j); S = I_ij - max(I_i, I_j); 再 sigmoid 归一化
        I_T, I_A, I_V = -L_T, -L_A, -L_V
        I_TA, I_TV, I_AV = -L_TA, -L_TV, -L_AV
        R_TA_raw = torch.minimum(I_T, I_A)
        R_TV_raw = torch.minimum(I_T, I_V)
        R_AV_raw = torch.minimum(I_A, I_V)
        S_TA_raw = I_TA - torch.maximum(I_T, I_A)
        S_TV_raw = I_TV - torch.maximum(I_T, I_V)
        S_AV_raw = I_AV - torch.maximum(I_A, I_V)
        R_TA = torch.sigmoid(R_TA_raw)
        R_TV = torch.sigmoid(R_TV_raw)
        R_AV = torch.sigmoid(R_AV_raw)
        S_TA = torch.sigmoid(S_TA_raw)
        S_TV = torch.sigmoid(S_TV_raw)
        S_AV = torch.sigmoid(S_AV_raw)
        r_tilde = (R_TA + R_TV + R_AV) / 3.0
        s_tilde = (S_TA + S_TV + S_AV) / 3.0

        # EMA update (in-place so buffer stays registered)
        r_val = (self.ema_beta * self.r_global_buf + (1.0 - self.ema_beta) * r_tilde.detach()).to(self.r_global_buf.device)
        s_val = (self.ema_beta * self.s_global_buf + (1.0 - self.ema_beta) * s_tilde.detach()).to(self.s_global_buf.device)
        self.r_global_buf.copy_(r_val)
        self.s_global_buf.copy_(s_val)

        r_global = torch.full((B, 1), self.r_global_buf.item(), device=device, dtype=H_T.dtype)
        s_global = torch.full((B, 1), self.s_global_buf.item(), device=device, dtype=H_T.dtype)
        return r_global, s_global, aux_pid_loss
