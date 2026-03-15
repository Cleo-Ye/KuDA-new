"""
Route B: PID Dual-Path MSA. 不推翻 UnimodalEncoder / dataset；编码器输出不 mean(1)，经 FeatureAlignment
-> BatchPIDPrior（可选）-> SampleEvidenceProxy -> DualPathRouter -> SharedPath / JointGainPath
-> Dynamic Fusion -> Prediction Heads.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.Encoder_KIAdapter import UnimodalEncoder
from models.modules import FeatureAlignment, SampleEvidenceProxy, DualPathRouter, SharedPath, JointGainPath


class PIDDualPathMSA(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased', use_batch_pid_prior=False):
        super(PIDDualPathMSA, self).__init__()
        self.opt = opt
        self.use_batch_pid_prior = use_batch_pid_prior
        hidden_size = getattr(opt, 'hidden_size', 256)
        dim_t, dim_v, dim_a = 768, 128, 128

        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)
        self.aligner = FeatureAlignment(dim_t=dim_t, dim_v=dim_v, dim_a=dim_a, hidden_dim=hidden_size)
        self.batch_pid_prior = None
        if use_batch_pid_prior:
            from models.modules.batch_pid_prior import BatchPIDPrior
            self.batch_pid_prior = BatchPIDPrior(hidden_dim=hidden_size, opt=opt)

        self.sample_proxy = SampleEvidenceProxy(hidden_dim=hidden_size)
        self.router = DualPathRouter(tau=getattr(opt, 'router_tau', 1.0))
        self.shared_path = SharedPath(
            hidden_dim=hidden_size,
            nhead=getattr(opt, 'path_nhead', 4),
            num_layers=getattr(opt, 'path_layers', 2),
            dropout=getattr(opt, 'dropout', 0.3),
            lambda_r=getattr(opt, 'lambda_r', 0.5),
        )
        self.joint_path = JointGainPath(
            hidden_dim=hidden_size,
            nhead=getattr(opt, 'path_nhead', 4),
            num_layers=getattr(opt, 'path_layers', 2),
            dropout=getattr(opt, 'dropout', 0.3),
            lambda_s=getattr(opt, 'lambda_s', 0.5),
            use_pairwise=getattr(opt, 'use_pairwise_joint_path', True),
            use_residual_gate=getattr(opt, 'use_residual_gate_in_joint_path', True),
            debug=getattr(opt, 'debug_joint_path', False),
            return_pairwise_debug=getattr(opt, 'return_pairwise_debug', False),
        )
        self.return_pairwise_debug = getattr(opt, 'return_pairwise_debug', False)
        self.main_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.aux_head_R = nn.Linear(hidden_size, 1)
        self.aux_head_S = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs_data_mask, multi_senti, gt_modal_labels=None):
        device = next(self.parameters()).device
        # 1. Unimodal encoder (no mean(1))
        uni_fea, uni_senti, posteriors, senti_scores = self.UniEncKI(inputs_data_mask)
        H_T_raw = uni_fea['T']   # [B, L_t, 768]
        H_V_raw = uni_fea['V']   # [B, L_v, 128]
        H_A_raw = uni_fea['A']   # [B, L_a, 128]

        # 2. Feature alignment
        H_T, H_V, H_A = self.aligner(H_T_raw, H_V_raw, H_A_raw)   # [B, L_*, 256]

        # 3. Batch PID prior (Phase 1: skip, use fixed 0.5)
        if self.batch_pid_prior is not None and multi_senti is not None:
            r_global, s_global, aux_pid_loss = self.batch_pid_prior(H_T, H_V, H_A, multi_senti)
        else:
            B = H_T.size(0)
            r_global = torch.full((B, 1), 0.5, device=device, dtype=H_T.dtype)
            s_global = torch.full((B, 1), 0.5, device=device, dtype=H_T.dtype)
            aux_pid_loss = torch.tensor(0.0, device=device)

        # 4. Sample evidence proxy
        q_r, q_s, e_scores = self.sample_proxy(H_T, H_A, H_V)

        # 4.5 构建 padding mask：数据集中 True=padding，转为 1=有效、0=padding
        B, len_t, len_v, len_a = H_T.size(0), H_T.size(1), H_V.size(1), H_A.size(1)
        mask_dict = inputs_data_mask.get('mask', {}) if isinstance(inputs_data_mask, dict) else getattr(inputs_data_mask, 'mask', {})
        if mask_dict.get('V') is not None and mask_dict['V'].numel() > 0:
            mask_V = (~mask_dict['V'].bool().to(device)).float()   # [B, L_v]
            if mask_V.size(1) != len_v:
                mask_V = mask_V[:, :len_v] if mask_V.size(1) >= len_v else F.pad(mask_V, (0, len_v - mask_V.size(1)), value=0.0)
        else:
            mask_V = torch.ones(B, len_v, device=device, dtype=H_T.dtype)
        if mask_dict.get('A') is not None and mask_dict['A'].numel() > 0:
            mask_A = (~mask_dict['A'].bool().to(device)).float()
            if mask_A.size(1) != len_a:
                mask_A = mask_A[:, :len_a] if mask_A.size(1) >= len_a else F.pad(mask_A, (0, len_a - mask_A.size(1)), value=0.0)
        else:
            mask_A = torch.ones(B, len_a, device=device, dtype=H_T.dtype)
        if mask_dict.get('T') is not None and isinstance(mask_dict['T'], torch.Tensor) and mask_dict['T'].numel() > 0:
            mask_T = (~mask_dict['T'].bool().to(device)).float()
            if mask_T.size(1) != len_t:
                mask_T = mask_T[:, :len_t] if mask_T.size(1) >= len_t else F.pad(mask_T, (0, len_t - mask_T.size(1)), value=0.0)
        else:
            mask_T = torch.ones(B, len_t, device=device, dtype=H_T.dtype)

        # 5. Dual path router
        alpha_r, alpha_s, H_R_in, H_S_in, mask_combined = self.router(
            r_global, s_global, q_r, q_s, e_scores, H_T, H_A, H_V,
            mask_T=mask_T, mask_A=mask_A, mask_V=mask_V,
        )

        # 6. Dual path（末端掩码均值池化）
        F_R = self.shared_path(H_R_in, alpha_r, mask_combined)   # [B, 256]
        joint_out = self.joint_path(
            H_S_in, alpha_s, len_t=len_t, len_v=len_v, len_a=len_a,
            mask_T=mask_T, mask_A=mask_A, mask_V=mask_V, mask_combined=mask_combined,
        )
        if isinstance(joint_out, tuple):
            F_S, pairwise_debug = joint_out
        else:
            F_S = joint_out
            pairwise_debug = None

        # 7. Dynamic fusion
        F_final = alpha_r * F_R + alpha_s * F_S

        # 8. Prediction heads
        pred = self.main_head(F_final)
        pred_R = self.aux_head_R(F_R)
        pred_S = self.aux_head_S(F_S)
        logit_cls = self.cls_head(F_final)

        senti_aux_loss = torch.tensor(0.0, device=device)
        if multi_senti is not None and hasattr(self, '_compute_senti_aux_loss'):
            senti_aux_loss = self._compute_senti_aux_loss(posteriors, multi_senti)

        out = {
            'pred': pred,
            'pred_R': pred_R,
            'pred_S': pred_S,
            'F_R': F_R,
            'F_S': F_S,
            'alpha_r': alpha_r,
            'alpha_s': alpha_s,
            'r_global': r_global,
            's_global': s_global,
            'aux_pid_loss': aux_pid_loss,
            'logit_cls': logit_cls,
            'senti_aux_loss': senti_aux_loss,
            'posteriors': posteriors,
        }
        if self.return_pairwise_debug and pairwise_debug is not None:
            out['F_TA'] = pairwise_debug['F_TA']
            out['F_TV'] = pairwise_debug['F_TV']
            out['F_AV'] = pairwise_debug['F_AV']
        return out

    def _compute_senti_aux_loss(self, posteriors, labels):
        """Auxiliary CE loss on mean posteriors vs binned labels (optional)."""
        labels = labels.view(-1)
        num_classes = getattr(self.opt, 'senti_num_classes', 7)
        boundaries = torch.linspace(-1.0, 1.0, num_classes + 1, device=labels.device)[1:-1]
        label_bins = torch.bucketize(labels, boundaries).clamp(0, num_classes - 1)
        total_loss = torch.tensor(0.0, device=labels.device)
        for m in ['T', 'A', 'V']:
            avg_posterior = posteriors[m].mean(dim=1)
            log_posterior = torch.log(avg_posterior + 1e-8)
            total_loss = total_loss + F.nll_loss(log_posterior, label_bins)
        return total_loss / 3.0
