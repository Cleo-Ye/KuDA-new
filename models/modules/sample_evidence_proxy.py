"""
SampleEvidenceProxy: token-level evidence score -> evidence distribution -> evidence center ->
local support (bounded similarity) -> modal-level support S -> concentration C -> q_r, q_s.
输入: H_T, H_V, H_A [B, L_*, 256]
输出: q_r [B,1], q_s [B,1], e_scores dict {e_T [B,L_T], e_A [B,L_A], e_V [B,L_V]}
"""
import math
import torch
import torch.nn.functional as F
from torch import nn


class SampleEvidenceProxy(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SampleEvidenceProxy, self).__init__()
        self.hidden_dim = hidden_dim
        self.score_T = nn.Linear(hidden_dim, 1)
        self.score_A = nn.Linear(hidden_dim, 1)
        self.score_V = nn.Linear(hidden_dim, 1)

    def forward(self, H_T, H_A, H_V):
        """
        H_T: [B, L_T, 256], H_A: [B, L_A, 256], H_V: [B, L_V, 256]
        Returns q_r [B,1], q_s [B,1], e_scores dict
        """
        B, L_T, _ = H_T.shape
        _, L_A, _ = H_A.shape
        _, L_V, _ = H_V.shape
        device = H_T.device
        eps = 1e-8

        # 1. Evidence logits per token
        e_T = self.score_T(H_T).squeeze(-1)   # [B, L_T]
        e_A = self.score_A(H_A).squeeze(-1)   # [B, L_A]
        e_V = self.score_V(H_V).squeeze(-1)   # [B, L_V]

        # 2. Evidence distribution (softmax over sequence)
        pi_T = F.softmax(e_T, dim=1)   # [B, L_T]
        pi_A = F.softmax(e_A, dim=1)   # [B, L_A]
        pi_V = F.softmax(e_V, dim=1)   # [B, L_V]

        # 3. Evidence center: weighted sum of features
        hbar_T = (pi_T.unsqueeze(-1) * H_T).sum(dim=1)   # [B, 256]
        hbar_A = (pi_A.unsqueeze(-1) * H_A).sum(dim=1)
        hbar_V = (pi_V.unsqueeze(-1) * H_V).sum(dim=1)

        # 4. Local support: bounded similarity (sigmoid-scaled dot product) token vs other-modal centers
        # For each token in T: similarity to hbar_A and hbar_V center -> then combine
        # Spec: sim = sigmoid((x * y).sum(-1) / sqrt(d)); c_T [B, L_T]
        d = self.hidden_dim
        scale = math.sqrt(d)
        # Token T_i vs [hbar_A, hbar_V]: use mean of (T to A center) and (T to V center) per token
        dot_TA = (H_T * hbar_A.unsqueeze(1)).sum(dim=-1) / scale   # [B, L_T]
        dot_TV = (H_T * hbar_V.unsqueeze(1)).sum(dim=-1) / scale
        c_T = torch.sigmoid((dot_TA + dot_TV) / 2.0)   # [B, L_T]
        dot_AV = (H_A * hbar_V.unsqueeze(1)).sum(dim=-1) / scale
        dot_AT = (H_A * hbar_T.unsqueeze(1)).sum(dim=-1) / scale
        c_A = torch.sigmoid((dot_AV + dot_AT) / 2.0)   # [B, L_A]
        dot_VA = (H_V * hbar_A.unsqueeze(1)).sum(dim=-1) / scale
        dot_VT = (H_V * hbar_T.unsqueeze(1)).sum(dim=-1) / scale
        c_V = torch.sigmoid((dot_VA + dot_VT) / 2.0)   # [B, L_V]

        # 5. Modal-level support: S_m = (pi_m * c_m).sum(dim=1), S = (S_T + S_A + S_V)/3
        S_T = (pi_T * c_T).sum(dim=1, keepdim=True)   # [B, 1]
        S_A = (pi_A * c_A).sum(dim=1, keepdim=True)
        S_V = (pi_V * c_V).sum(dim=1, keepdim=True)
        S = (S_T + S_A + S_V) / 3.0   # [B, 1]

        # 6. Evidence concentration: C = 1 - entropy / log(L); 使用 1e-9 避免 log(0) 导致 NaN
        log_eps = 1e-9
        entropy_T = -(pi_T * torch.log(pi_T + log_eps)).sum(dim=1, keepdim=True)
        C_T = 1.0 - entropy_T / (math.log(L_T + eps) + eps)
        entropy_A = -(pi_A * torch.log(pi_A + log_eps)).sum(dim=1, keepdim=True)
        C_A = 1.0 - entropy_A / (math.log(L_A + eps) + eps)
        entropy_V = -(pi_V * torch.log(pi_V + log_eps)).sum(dim=1, keepdim=True)
        C_V = 1.0 - entropy_V / (math.log(L_V + eps) + eps)
        C = (C_T + C_A + C_V) / 3.0   # [B, 1]

        # 7. Local proxies
        q_r = S   # [B, 1]
        q_s = (1.0 - S) * C   # [B, 1]

        e_scores = {'e_T': e_T, 'e_A': e_A, 'e_V': e_V}
        return q_r, q_s, e_scores
