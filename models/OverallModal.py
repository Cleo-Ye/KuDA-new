"""
KuDA refactored: PID estimator (Strategy 1) + contrastive dual-branch (Strategy 3).
IEC and ICR modules are fully replaced. Single forward path: UniEncKI -> pool -> PID -> dual_branch -> S-weighted fusion -> regressor.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.Encoder_KIAdapter import UnimodalEncoder
from models.PIDEstimator import PIDEstimator
from models.DualBranchExtractor import DualBranchExtractor


class KMSA(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        self.opt = opt
        hidden_size = getattr(opt, 'hidden_size', 256)

        # Unimodal Encoder (unchanged)
        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)

        # 模态维度对齐到 hidden_size
        # T: BERT 输出 768；V/A: TfEncoder 默认 proj_fea_dim=128
        self.pool_proj_T = nn.Linear(768, hidden_size)
        self.pool_proj_V = nn.Linear(128, hidden_size)
        self.pool_proj_A = nn.Linear(128, hidden_size)

        # PID estimator
        self.pid_estimator = PIDEstimator(
            input_dim=hidden_size,
            K=getattr(opt, 'pid_K', 16),
            n_iter=getattr(opt, 'sinkhorn_iters', 5),
            sigmoid_scale=getattr(opt, 'sigmoid_scale', 2.0),
        )

        # Dual-branch extractor
        self.dual_branch = DualBranchExtractor(
            hidden_dim=hidden_size,
            nhead=4,
            dropout=getattr(opt, 'dropout', 0.3),
        )

        # Regression head (replaces SentiCLS)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # 辅助二分类头：L_classification，>0 为正类、<0 为负类，用于提升 Acc-2/F1
        self.cls_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs_data_mask, multi_senti, gt_modal_labels=None):
        device = next(self.parameters()).device

        # Stage 1: Unimodal encoding
        uni_fea, uni_senti, posteriors, senti_scores = self.UniEncKI(inputs_data_mask)
        hidden_T = uni_fea['T']   # [B, L_t, 1536]
        hidden_V = uni_fea['V']   # [B, L_v, 256]
        hidden_A = uni_fea['A']   # [B, L_a, 256]

        # Pool (mean over time) + 维度对齐
        Z_T = self.pool_proj_T(hidden_T.mean(1))  # [B, hidden_size]
        Z_V = self.pool_proj_V(hidden_V.mean(1))  # [B, hidden_size]
        Z_A = self.pool_proj_A(hidden_A.mean(1))  # [B, hidden_size]

        # Stage 2: PID estimation（消融 w/o PID Routing 时关闭，S 固定 0.5，L_PID=0）
        if getattr(self.opt, 'ablation_no_pid_routing', False):
            B = Z_T.size(0)
            S = torch.full((B,), 0.5, device=device, dtype=Z_T.dtype)
            R = torch.zeros(B, device=device)
            L_PID = torch.tensor(0.0, device=device)
        else:
            S, R, L_PID = self.pid_estimator(Z_T, Z_V, Z_A)  # S: [B]

        # Stage 3: Dual-branch extraction
        F_cons, F_conf = self.dual_branch(Z_T, Z_V, Z_A, S=S)  # [B, D]

        # Stage 4: Fusion（消融时改为静态 0.5/0.5 或仅 F_cons）
        if getattr(self.opt, 'ablation_no_pid_routing', False):
            F_fusion = 0.5 * F_cons + 0.5 * F_conf
        elif getattr(self.opt, 'ablation_single_branch', False):
            F_fusion = F_cons
        else:
            S_w = S.unsqueeze(-1)
            F_fusion = (1.0 - S_w) * F_cons + S_w * F_conf
        prediction = self.regressor(F_fusion)
        logit_cls = self.cls_head(F_fusion)  # [B, 1]，用于 L_classification（BCE，>0 正类）

        # Auxiliary sentiment loss (from posteriors)
        senti_aux_loss = torch.tensor(0.0, device=device)
        if multi_senti is not None:
            senti_aux_loss = self._compute_senti_aux_loss(posteriors, multi_senti)

        # 供可视化/统计复用：将 S 暴露为 last_conflict_intensity（协同度即“冲突/分歧”强度）
        self.last_conflict_intensity = S.detach()

        return prediction, senti_aux_loss, L_PID, F_cons, F_conf, S, logit_cls

    def _compute_senti_aux_loss(self, posteriors, labels):
        """Auxiliary CE loss on mean posteriors vs binned labels."""
        labels = labels.view(-1)
        num_classes = getattr(self.opt, 'senti_num_classes', 7)
        boundaries = torch.linspace(-1.0, 1.0, num_classes + 1, device=labels.device)[1:-1]
        label_bins = torch.bucketize(labels, boundaries).clamp(0, num_classes - 1)

        total_loss = torch.tensor(0.0, device=labels.device)
        count = 0
        for m in ['T', 'A', 'V']:
            avg_posterior = posteriors[m].mean(dim=1)
            log_posterior = torch.log(avg_posterior + 1e-8)
            total_loss = total_loss + F.nll_loss(log_posterior, label_bins)
            count += 1
        return total_loss / count



def build_model(opt):
    if 'sims' in opt.datasetName:
        l_pretrained = './pretrainedModel/BERT/bert-base-chinese'
    else:
        l_pretrained = './pretrainedModel/BERT/bert-base-uncased'
    model_type = getattr(opt, 'model_type', 'kmsa')
    if model_type == 'pid_dualpath':
        from models.pid_dualpath_msa import PIDDualPathMSA
        use_batch_pid_prior = getattr(opt, 'use_batch_pid_prior', False)
        model = PIDDualPathMSA(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained, use_batch_pid_prior=use_batch_pid_prior)
        return model
    model = KMSA(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained)
    return model
