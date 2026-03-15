"""
Differentiable PID-inspired synergy estimator (Strategy 1).
Input: sample-level triple (Z_T, Z_V, Z_A). Output: synergy S, redundancy R, L_PID.

S is computed by a dedicated MLP head directly from cross-modal features, avoiding the
Sinkhorn marginal collapse (doubly-stochastic → r_i=1 → H_r=0 → MI=0 → S≡0.55).
Sinkhorn is retained only for L_PID regularization.
"""
import torch
import torch.nn.functional as F
from torch import nn


class PIDEstimator(nn.Module):
    def __init__(self, input_dim, K=16, n_iter=5, eps=1e-8, sigmoid_scale=2.0):
        super(PIDEstimator, self).__init__()
        # Sinkhorn transport network (used only for L_PID regularization)
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, K * K),
        )
        # Dedicated S head: directly learns synergy from cross-modal conflict signals.
        # Input: concat of pairwise diff + abs-diff features, avoiding the H_r=0 dead-lock.
        self.s_head = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.K = K
        self.n_iter = n_iter
        self.eps = eps
        self.sigmoid_scale = sigmoid_scale

    def forward(self, Z_T, Z_V, Z_A):
        """
        Z_T, Z_V, Z_A: [B, input_dim]. Returns S [B], R [B], L_PID scalar.
        """
        B = Z_T.size(0)
        x = torch.cat([Z_T, Z_V, Z_A], dim=-1)  # [B, input_dim*3]

        # --- S head: per-sample synergy score ---
        # Features: pairwise absolute differences capture cross-modal conflict directly.
        diff_tv = (Z_T - Z_V).abs()
        diff_ta = (Z_T - Z_A).abs()
        diff_va = (Z_V - Z_A).abs()
        s_input = torch.cat([diff_tv, diff_ta, diff_va], dim=-1)  # [B, input_dim*3]
        S_logit = self.s_head(s_input).squeeze(-1)                 # [B]
        S = S_logit.sigmoid() * 0.9 + 0.1                         # S ∈ [0.1, 1.0]

        # --- Sinkhorn transport (for L_PID regularization only) ---
        logits = self.net(x).view(B, self.K, self.K)
        log_A = logits
        for _ in range(self.n_iter):
            log_A = log_A - torch.logsumexp(log_A, dim=-1, keepdim=True)
            log_A = log_A - torch.logsumexp(log_A, dim=-2, keepdim=True)

        # L_PID: encourage the transport plan to be non-trivial (low entropy)
        # clamp to avoid -log(0)=inf and gradient explosion
        log_A_safe = log_A.clamp(min=1e-8)
        L_PID = torch.clamp(-log_A_safe.mean(), max=10.0)

        # R: placeholder redundancy (kept for API compatibility)
        R = torch.zeros(B, device=Z_T.device)

        return S, R, L_PID
