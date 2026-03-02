"""
视频Token筛选模块 (设计文档第3步: 单模态去冗余压缩)
- VisionTokenPruner: 原启发式 DART 风格 (保留作消融)
- TextGuidedVisionPruner: 文本引导 prune+merge (确定性实现)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedVisionPruner(nn.Module):
    """
    文本引导的 Vision IEC: 文本引导重要性 score pruning + 剩余 token merging.
    参考 PuMer/ToMe: 保留与文本/情绪相关证据, 合并冗余证据.
    """
    def __init__(self, vision_keep_ratio=0.50, hidden_dim=256, text_dim=1536, nhead=4, dropout=0.1):
        super().__init__()
        self.vision_keep_ratio = vision_keep_ratio
        self.proj_t = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )

    def forward(self, hidden_v, hidden_t, senti_t, mask_v=None, mask_t=None):
        """
        Args:
            hidden_v: [B, L_v, 256]
            hidden_t: [B, L_t, 1536]
            senti_t: [B, L_t]
            mask_v: [B, L_v] True=valid (若 None 则全有效)
            mask_t: [B, L_t] True=valid (若 None 则全有效)
        Returns:
            hidden_v_new: [B, K, 256]
            retained_indices: [B, K]
            pruning_info: dict
        """
        B, L_v, D = hidden_v.shape
        L_t = hidden_t.shape[1]
        device = hidden_v.device

        if mask_v is None:
            mask_v = torch.ones(B, L_v, dtype=torch.bool, device=device)
        if mask_t is None:
            mask_t = torch.ones(B, L_t, dtype=torch.bool, device=device)

        # 1) 维度对齐
        T = self.proj_t(hidden_t)  # [B, L_t, 256]
        V = hidden_v  # [B, L_v, 256]

        # 2) 文本权重 w_t = |senti_t| * mask_t，归一化增强区分度
        w_t = torch.abs(senti_t).float() * mask_t.float()  # [B, L_t], 范围 [0, 3]
        # 归一化到和为1，使不同样本的权重分布更一致
        w_t = w_t / (w_t.sum(dim=1, keepdim=True) + 1e-8)
        w_t = w_t * mask_t.float()  # 重新应用mask确保padding位置为0

        # 3) V->T cross-attn, 取注意力权重
        # key_padding_mask: True = padding (mask out)
        key_padding_mask = ~mask_t  # [B, L_t], True where invalid
        attn_out, attn_weights = self.cross_attn(
            V, T, T,
            key_padding_mask=key_padding_mask,
            average_attn_weights=True
        )  # attn_weights: [B, L_v, L_t]
        attn_vt = attn_weights  # [B, L_v, L_t]

        # 4) score_v = attn_vt @ w_t, 结合注意力和文本情感权重
        score_v = torch.bmm(attn_vt, w_t.unsqueeze(-1)).squeeze(-1)  # [B, L_v]
        score_v = score_v.masked_fill(~mask_v, -1e9)

        # 5) Prune: top-K
        K = max(int(torch.ceil(torch.tensor(L_v * self.vision_keep_ratio, device=device)).item()), 1)
        K = min(K, L_v)

        hidden_v_new_list = []
        retained_indices_list = []
        for b in range(B):
            _, top_idx = torch.topk(score_v[b], K)
            top_idx = top_idx.sort()[0]
            retained_indices_list.append(top_idx)
            V_keep = V[b, top_idx]  # [K, 256]
            score_keep = score_v[b, top_idx]  # [K]

            # 6) Merge: 未保留 token 按 cosine 最近邻分配到 K 个保留 token
            drop_mask = torch.ones(L_v, dtype=torch.bool, device=device)
            drop_mask[top_idx] = False
            drop_idx = torch.nonzero(drop_mask, as_tuple=False).squeeze(-1)
            if drop_idx.numel() == 0:
                hidden_v_new_list.append(V_keep)
                continue

            V_drop = V[b, drop_idx]  # [L_drop, 256]
            score_drop = score_v[b, drop_idx]  # [L_drop]

            # cosine similarity: V_drop @ V_keep^T (normalized)
            V_keep_n = F.normalize(V_keep, dim=-1)
            V_drop_n = F.normalize(V_drop, dim=-1)
            sim = torch.mm(V_drop_n, V_keep_n.t())  # [L_drop, K]
            assign = sim.argmax(dim=1)  # [L_drop], each in [0, K-1]

            # 合并: 对每个保留 token k, 把分配给它的 drop token 做 score 加权
            V_merge = V_keep.clone()
            for k in range(K):
                members = (assign == k).nonzero(as_tuple=True)[0]
                if members.numel() > 0:
                    weights = F.softmax(score_drop[members], dim=0)
                    merged = (V_drop[members] * weights.unsqueeze(-1)).sum(dim=0)
                    V_merge[k] = F.normalize(V_merge[k] + merged, dim=-1)
            hidden_v_new_list.append(V_merge)

        # 填充到相同长度
        max_K = max(x.shape[0] for x in hidden_v_new_list)
        hidden_v_new = torch.zeros(B, max_K, D, device=device, dtype=hidden_v.dtype)
        retained_indices = torch.zeros(B, max_K, dtype=torch.long, device=device)
        pruned_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)
        for b in range(B):
            n = hidden_v_new_list[b].shape[0]
            hidden_v_new[b, :n] = hidden_v_new_list[b]
            retained_indices[b, :n] = retained_indices_list[b]
            pruned_mask[b, :n] = True

        pruning_info = {
            'original_length': L_v,
            'pruned_length': max_K,
            'compression_ratio': 1.0 - (max_K / L_v),
            'pruned_mask': pruned_mask,
        }
        return hidden_v_new, retained_indices, pruning_info


class VisionTokenPruner(nn.Module):
    """
    视频token筛选: 纯视觉内部去冗余 + 低置信token过滤.
    不做冲突/一致分流(那是ConflictJS EvidenceSplitter的职责).
    使用posteriors仅用于Rel置信度过滤, 不涉及跨模态比较.
    """
    def __init__(self, tau_rel=0.1, target_ratio=0.3, min_retain=15):
        """
        Args:
            tau_rel: 置信度阈值, Rel=max(p_V_i) < tau_rel的token视为低质量噪声
            target_ratio: 最终保留的token比例
            min_retain: 每个样本最少保留的token数
        """
        super().__init__()
        self.tau_rel = tau_rel
        self.target_ratio = target_ratio
        self.min_retain = min_retain

    def forward(self, hidden_v, posteriors_v):
        """
        Args:
            hidden_v: [B, L_v, D] 视频token表征
            posteriors_v: [B, L_v, C] 视觉token的情感后验分布(用于Rel过滤)
        Returns:
            hidden_v_pruned: [B, L_v', D] 压缩后的视频token
            retained_indices: [B, L_v'] 保留的索引
            pruning_info: dict 记录压缩信息
        """
        B, L_v, D = hidden_v.shape
        device = hidden_v.device
        target_num = max(int(L_v * self.target_ratio), self.min_retain)

        # Step 1: 计算置信度 Rel_i = max(p_V_i), 过滤低置信噪声token
        rel = posteriors_v.max(dim=-1)[0]  # [B, L_v]

        retained_indices_list = []
        for b in range(B):
            # 高置信token索引
            high_rel_idx = torch.nonzero(rel[b] > self.tau_rel).squeeze(-1)
            # 关键: 如果高置信token不足target_num, 补齐到target_num
            # 这样vision_target_ratio才能真正控制保留数量
            if len(high_rel_idx) < target_num:
                k = min(target_num, L_v)
                high_rel_idx = torch.topk(rel[b], k).indices
            high_rel_idx = high_rel_idx.sort()[0]

            # Step 2: 在高置信token中做纯视觉内部去冗余
            if len(high_rel_idx) <= target_num:
                # 不需要压缩
                retained_indices_list.append(high_rel_idx)
                continue

            # DART-inspired: L2范数选pivot + 余弦相似度聚类保留多样性
            tokens = hidden_v[b, high_rel_idx, :]  # [N_high, D]
            norms = torch.norm(tokens, p=2, dim=-1)  # [N_high]
            pivot_num = min(max(target_num // 4, 1), len(high_rel_idx))
            pivot_local = torch.topk(norms, pivot_num).indices

            selected = set(pivot_local.tolist())
            available = set(range(len(high_rel_idx))) - selected
            budget = target_num - pivot_num
            topk_per_pivot = max(budget // max(pivot_num, 1), 1)

            for p_idx in pivot_local.tolist():
                if not available or len(selected) >= target_num:
                    break
                p_vec = tokens[p_idx:p_idx+1]  # [1, D]
                avail_list = list(available)
                avail_vecs = tokens[avail_list]  # [N_avail, D]
                # 选最不相似的(多样性保留), 用负余弦相似度
                cos = F.cosine_similarity(p_vec, avail_vecs, dim=-1)
                k = min(topk_per_pivot, len(avail_list))
                if k > 0:
                    # 选最不相似的k个(多样性)
                    bottom_local = torch.topk(-cos, k).indices
                    bottom_real = [avail_list[i] for i in bottom_local.tolist()]
                    selected.update(bottom_real)
                    available.difference_update(bottom_real)

            selected_list = sorted(list(selected))[:target_num]
            local_indices = torch.tensor(selected_list, dtype=torch.long, device=device)
            retained_indices_list.append(high_rel_idx[local_indices])

        # 填充到相同长度
        max_retained = max(len(idx) for idx in retained_indices_list)
        max_retained = max(max_retained, 1)
        retained_indices = torch.zeros(B, max_retained, dtype=torch.long, device=device)
        pruned_mask = torch.zeros(B, max_retained, dtype=torch.bool, device=device)

        for b in range(B):
            n = len(retained_indices_list[b])
            retained_indices[b, :n] = retained_indices_list[b]
            pruned_mask[b, :n] = True

        # 提取token
        hidden_v_pruned = torch.zeros(B, max_retained, D, device=device)
        for b in range(B):
            n = len(retained_indices_list[b])
            hidden_v_pruned[b, :n] = hidden_v[b, retained_indices[b, :n]]

        pruning_info = {
            'original_length': L_v,
            'pruned_length': max_retained,
            'compression_ratio': 1.0 - (max_retained / L_v),
            'pruned_mask': pruned_mask,
        }

        return hidden_v_pruned, retained_indices, pruning_info
