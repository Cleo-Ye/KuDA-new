"""
JointGainPath: Joint-Gain Path. 支持两种模式：
- use_pairwise=True: 将 H_S_in 按 len_t, len_v, len_a 切分为 H_T, H_V, H_A，做 TA/TV/AV 三对双向 cross-attention，
  每向 mean pooling 后两向取平均得 F_TA/F_TV/F_AV，再 concat+MLP 得 F_S。每 sub-block 应用 residual gate g_s。
- use_pairwise=False: 简单 TransformerEncoder + gate + mean pooling（与原版一致）。
"""
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from .attention_blocks import CrossAttentionBlock


class JointGainPath(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        lambda_s=0.5,
        use_pairwise=True,
        use_residual_gate=True,
        debug=False,
        return_pairwise_debug=False,
    ):
        super(JointGainPath, self).__init__()
        self.hidden_dim = hidden_dim
        self.lambda_s = lambda_s
        self.use_pairwise = use_pairwise
        self.use_residual_gate = use_residual_gate
        self.debug = debug
        self.return_pairwise_debug = return_pairwise_debug

        if use_pairwise:
            # TA: T<->A, TV: T<->V, AV: A<->V. 每对两个方向各一个 block.
            self.ta_t_from_a = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            self.ta_a_from_t = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            self.tv_t_from_v = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            self.tv_v_from_t = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            self.av_a_from_v = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            self.av_v_from_a = CrossAttentionBlock(hidden_dim, nhead=nhead, dropout=dropout)
            # [B, 3d] -> [B, d]
            self.fuse_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
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

    def forward(self, H_S_in, alpha_s, len_t=None, len_v=None, len_a=None, mask_T=None, mask_A=None, mask_V=None, mask_combined=None):
        """
        H_S_in: [B, L_total, d], alpha_s: [B, 1]
        len_t, len_v, len_a: 仅 use_pairwise=True 时必需，满足 len_t+len_v+len_a == L_total.
        mask_T, mask_A, mask_V: [B, L_*] 有效=1, padding=0；pairwise 时用于各段掩码均值
        mask_combined: [B, L_total] encoder 模式时用于掩码均值；None 则整序列 mean
        Returns F_S [B, d], 若 return_pairwise_debug=True 则返回 (F_S, dict(F_TA=..., F_TV=..., F_AV=...)).
        """
        if self.use_pairwise:
            return self._forward_pairwise(H_S_in, alpha_s, len_t, len_v, len_a, mask_T, mask_A, mask_V)
        return self._forward_encoder(H_S_in, alpha_s, mask_combined)

    def _forward_encoder(self, H_S_in, alpha_s, mask_combined=None):
        """回退路径：TransformerEncoder + gate + mean pooling。mask_combined [B, L_total] 有效=1、padding=0；None 则整序列 mean。"""
        x = H_S_in
        g_s = 1.0 + self.lambda_s * alpha_s if self.use_residual_gate else torch.ones_like(alpha_s)
        g_s = g_s.unsqueeze(-1)
        for layer in self.encoder.layers:
            delta = layer(x)
            x = x + g_s * delta
        if self.encoder.norm is not None:
            x = self.encoder.norm(x)
        if mask_combined is not None:
            valid_lengths = mask_combined.sum(dim=1, keepdim=True).clamp(min=1e-9)
            F_S = (x * mask_combined.unsqueeze(-1)).sum(dim=1) / valid_lengths
        else:
            F_S = x.mean(dim=1)
        return F_S

    @staticmethod
    def _masked_mean(x, mask, dim=1):
        """x [B, L, d], mask [B, L] 有效=1；返回 [B, d]"""
        if mask is None:
            return x.mean(dim=dim)
        valid = mask.sum(dim=dim, keepdim=True).clamp(min=1e-9)
        return (x * mask.unsqueeze(-1)).sum(dim=dim) / valid

    def _forward_pairwise(self, H_S_in, alpha_s, len_t, len_v, len_a, mask_T=None, mask_A=None, mask_V=None):
        B, L_total, d = H_S_in.shape
        if len_t is None or len_v is None or len_a is None:
            raise ValueError("JointGainPath pairwise mode requires len_t, len_v, len_a")
        if len_t + len_v + len_a != L_total:
            raise AssertionError(
                f"Length mismatch: len_t({len_t}) + len_v({len_v}) + len_a({len_a}) != H_S_in.size(1)({L_total})"
            )
        # 切分顺序与 router 一致: T, V, A
        H_T = H_S_in[:, :len_t, :]           # [B, L_t, d]
        H_V = H_S_in[:, len_t:len_t + len_v, :]   # [B, L_v, d]
        H_A = H_S_in[:, len_t + len_v:, :]        # [B, L_a, d]

        g_s = (1.0 + self.lambda_s * alpha_s) if self.use_residual_gate else torch.ones_like(alpha_s)
        g_s = g_s.unsqueeze(-1)   # [B, 1, 1] 用于广播到 [B, L, d]

        # TA: T<-A, A<-T; 掩码均值池化; F_TA = (F_T_from_A + F_A_from_T)/2
        delta_T_from_A = self.ta_t_from_a(H_T, H_A)   # [B, L_t, d]
        delta_A_from_T = self.ta_a_from_t(H_A, H_T)   # [B, L_a, d]
        H_T_updated = H_T + g_s * delta_T_from_A
        H_A_updated = H_A + g_s * delta_A_from_T
        F_T_from_A = self._masked_mean(H_T_updated, mask_T)   # [B, d]
        F_A_from_T = self._masked_mean(H_A_updated, mask_A)
        F_TA = (F_T_from_A + F_A_from_T) * 0.5   # [B, d]

        # TV
        delta_T_from_V = self.tv_t_from_v(H_T, H_V)
        delta_V_from_T = self.tv_v_from_t(H_V, H_T)
        H_T_updated = H_T + g_s * delta_T_from_V
        H_V_updated = H_V + g_s * delta_V_from_T
        F_T_from_V = self._masked_mean(H_T_updated, mask_T)
        F_V_from_T = self._masked_mean(H_V_updated, mask_V)
        F_TV = (F_T_from_V + F_V_from_T) * 0.5

        # AV
        delta_A_from_V = self.av_a_from_v(H_A, H_V)
        delta_V_from_A = self.av_v_from_a(H_V, H_A)
        H_A_updated = H_A + g_s * delta_A_from_V
        H_V_updated = H_V + g_s * delta_V_from_A
        F_A_from_V = self._masked_mean(H_A_updated, mask_A)
        F_V_from_A = self._masked_mean(H_V_updated, mask_V)
        F_AV = (F_A_from_V + F_V_from_A) * 0.5

        if self.debug:
            print("JointGainPath debug: H_S_in", H_S_in.shape, "H_T/H_V/H_A", H_T.shape, H_V.shape, H_A.shape,
                  "F_TA/F_TV/F_AV", F_TA.shape, F_TV.shape, F_AV.shape)

        # [B, 3d] -> MLP -> [B, d]
        F_cat = torch.cat([F_TA, F_TV, F_AV], dim=-1)   # [B, 3d]
        F_S = self.fuse_mlp(F_cat)   # [B, d]

        if self.debug:
            print("JointGainPath debug: F_S", F_S.shape)

        if self.return_pairwise_debug:
            return F_S, {'F_TA': F_TA, 'F_TV': F_TV, 'F_AV': F_AV}
        return F_S
