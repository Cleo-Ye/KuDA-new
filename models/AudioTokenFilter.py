"""
Audio 轻量压缩模块 (Phase 2 论文防守: 为何只压缩 Vision 的对照)
方案 A: 低置信过滤 (Rel < tau_rel 的帧丢弃)
方案 B: 按 Rel 保留 top-r% 帧
与 ICR 统一使用熵基可靠度 Rel = 1 - H(p)/log(C).
"""
import torch
import torch.nn as nn

from models.ConflictJS import _reliability_from_entropy


class AudioTokenFilter(nn.Module):
    """
    Audio 轻量筛选: 低置信过滤 或 top-r% 保留, 不做多样性聚类.
    输出格式与 VisionTokenPruner 一致: 压缩后序列 + padding mask, 供融合层 key_padding_mask 使用.
    """
    def __init__(self, mode='top_r', tau_rel=0.3, target_ratio=0.5, min_retain=10):
        """
        Args:
            mode: 'low_conf' (方案A: 仅保留 Rel > tau_rel) 或 'top_r' (方案B: 按 Rel 保留 top target_ratio)
            tau_rel: 低置信阈值 (方案A 使用)
            target_ratio: 方案B 保留比例; 方案A 下可忽略
            min_retain: 每样本最少保留帧数
        """
        super().__init__()
        self.mode = mode
        self.tau_rel = tau_rel
        self.target_ratio = target_ratio
        self.min_retain = min_retain

    def forward(self, hidden_a, posteriors_a):
        """
        Args:
            hidden_a: [B, L_a, D]
            posteriors_a: [B, L_a, C]
        Returns:
            hidden_a_filtered: [B, K, D]  K = max(保留数) per batch
            retained_indices: [B, K]  (for compatibility, may be padded)
            pruning_info: dict with pruned_mask [B, K], original_length, filtered_length, etc.
        """
        B, L_a, D = hidden_a.shape
        device = hidden_a.device
        # 与 ICR 统一: 熵基可靠度
        rel = _reliability_from_entropy(posteriors_a)  # [B, L_a]

        retained_indices_list = []
        for b in range(B):
            if self.mode == 'low_conf':
                keep = rel[b] > self.tau_rel
                idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                if idx.numel() < self.min_retain:
                    # 保底: 按 Rel top min_retain
                    _, top_idx = torch.topk(rel[b], min(self.min_retain, L_a))
                    idx = top_idx.sort()[0]
            else:
                # top_r: 保留 top target_ratio 的帧, K = ceil(L_a * audio_keep_ratio)
                target_num = max(int(torch.ceil(torch.tensor(L_a * self.target_ratio, device=hidden_a.device)).item()), self.min_retain)
                target_num = min(target_num, L_a)
                _, top_idx = torch.topk(rel[b], target_num)
                idx = top_idx.sort()[0]
            retained_indices_list.append(idx)

        max_retained = max(len(idx) for idx in retained_indices_list)
        max_retained = max(max_retained, 1)
        retained_indices = torch.zeros(B, max_retained, dtype=torch.long, device=device)
        pruned_mask = torch.zeros(B, max_retained, dtype=torch.bool, device=device)
        hidden_a_filtered = torch.zeros(B, max_retained, D, device=device, dtype=hidden_a.dtype)

        for b in range(B):
            idx = retained_indices_list[b]
            n = len(idx)
            retained_indices[b, :n] = idx
            pruned_mask[b, :n] = True
            hidden_a_filtered[b, :n] = hidden_a[b, idx]

        pruning_info = {
            'original_length': L_a,
            'filtered_length': max_retained,
            'pruned_mask': pruned_mask,
        }
        return hidden_a_filtered, retained_indices, pruning_info
