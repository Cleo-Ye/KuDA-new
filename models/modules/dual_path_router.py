"""
DualPathRouter: 用 r_global, s_global 与 q_r, q_s 得到 alpha_r, alpha_s；
用修正后的 mask 公式 M_R = alpha_r * sigmoid(e), M_S = alpha_s * sigmoid(e) 生成双路径视图；
在序列维拼接得到 H_R_in, H_S_in [B, L_total, 256].
"""
import torch
import torch.nn.functional as F
from torch import nn


class DualPathRouter(nn.Module):
    def __init__(self, tau=1.0):
        super(DualPathRouter, self).__init__()
        self.tau = tau

    def forward(self, r_global, s_global, q_r, q_s, e_scores, H_T, H_A, H_V, mask_T=None, mask_A=None, mask_V=None):
        """
        r_global, s_global: [B, 1] or scalar/broadcastable
        q_r, q_s: [B, 1]
        e_scores: dict with e_T [B,L_T], e_A [B,L_A], e_V [B,L_V]
        H_T, H_A, H_V: [B, L_*, 256]
        mask_T, mask_A, mask_V: [B, L_*] 有效=1, padding=0；若为 None 则视为全有效
        Returns alpha_r [B,1], alpha_s [B,1], H_R_in [B, L_total, 256], H_S_in [B, L_total, 256], mask_combined [B, L_total]
        """
        B = H_T.size(0)
        device = H_T.device
        # Ensure [B, 1] for broadcasting
        if not isinstance(r_global, torch.Tensor) or r_global.dim() == 0:
            r_global = torch.full((B, 1), float(r_global), device=device, dtype=H_T.dtype)
        elif r_global.dim() == 1:
            r_global = r_global.unsqueeze(1)
        if not isinstance(s_global, torch.Tensor) or s_global.dim() == 0:
            s_global = torch.full((B, 1), float(s_global), device=device, dtype=H_T.dtype)
        elif s_global.dim() == 1:
            s_global = s_global.unsqueeze(1)

        # 1. Routing weights: alpha = softmax([r_global*q_r, s_global*q_s] / tau)
        logits = torch.cat([r_global * q_r, s_global * q_s], dim=1)   # [B, 2]
        alpha = F.softmax(logits / self.tau, dim=1)
        alpha_r = alpha[:, 0:1]   # [B, 1]
        alpha_s = alpha[:, 1:2]   # [B, 1]

        e_T = e_scores['e_T']   # [B, L_T]
        e_A = e_scores['e_A']   # [B, L_A]
        e_V = e_scores['e_V']   # [B, L_V]

        # 2. Masks: M_R = alpha_r * sigmoid(e), M_S = alpha_s * sigmoid(e)
        M_R_T = alpha_r * torch.sigmoid(e_T)   # [B, L_T]
        M_S_T = alpha_s * torch.sigmoid(e_T)
        M_R_A = alpha_r * torch.sigmoid(e_A)
        M_S_A = alpha_s * torch.sigmoid(e_A)
        M_R_V = alpha_r * torch.sigmoid(e_V)
        M_S_V = alpha_s * torch.sigmoid(e_V)

        # 3. Apply masks: H_R_T = M_R_T.unsqueeze(-1) * H_T
        H_R_T = M_R_T.unsqueeze(-1) * H_T   # [B, L_T, 256]
        H_S_T = M_S_T.unsqueeze(-1) * H_T
        H_R_A = M_R_A.unsqueeze(-1) * H_A
        H_S_A = M_S_A.unsqueeze(-1) * H_A
        H_R_V = M_R_V.unsqueeze(-1) * H_V
        H_S_V = M_S_V.unsqueeze(-1) * H_V

        # 4. Concatenate along sequence dim: T then V then A (plan: L_total = L_T + L_V + L_A)
        H_R_in = torch.cat([H_R_T, H_R_V, H_R_A], dim=1)   # [B, L_total, 256]
        H_S_in = torch.cat([H_S_T, H_S_V, H_S_A], dim=1)

        # 5. 拼接 padding mask（1=有效, 0=padding），供末端掩码均值池化
        L_T, L_A, L_V = H_T.size(1), H_A.size(1), H_V.size(1)
        if mask_T is not None and mask_A is not None and mask_V is not None:
            mask_combined = torch.cat([mask_T, mask_V, mask_A], dim=1)   # [B, L_total]
        else:
            mask_combined = torch.ones(B, L_T + L_V + L_A, device=device, dtype=H_T.dtype)

        return alpha_r, alpha_s, H_R_in, H_S_in, mask_combined
