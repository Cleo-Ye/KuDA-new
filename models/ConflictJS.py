"""
Phase 2: Conflict-JS核心模块
包含: AlignmentAwareReference, 证据拆分, Evidence-level JS, 冲突强度合成
Rel 使用归一化熵: Rel=1−H(p)/log(C). 冲突参照为对齐感知局部参照.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentAwareReference(nn.Module):
    """
    对齐感知局部参照: senti_ref_mi = Σ_j attn_{i,j}·senti_Tj
    ref_V = Attn(V→T) @ senti_T, ref_A = Attn(A→T) @ senti_T
    ref_T = 0.5*(Attn(T→V)@senti_V + Attn(T→A)@senti_A)
    """
    def __init__(self, text_dim=1536, align_dim=256, nhead=4):
        super().__init__()
        self.proj_t = nn.Linear(text_dim, align_dim)
        self.attn_v2t = nn.MultiheadAttention(align_dim, nhead, batch_first=True)
        self.attn_a2t = nn.MultiheadAttention(align_dim, nhead, batch_first=True)
        self.attn_t2v = nn.MultiheadAttention(align_dim, nhead, batch_first=True)
        self.attn_t2a = nn.MultiheadAttention(align_dim, nhead, batch_first=True)

    def forward(self, hidden_T, hidden_V, hidden_A, senti_T, senti_V, senti_A,
                mask_T=None, mask_V=None, mask_A=None):
        """
        Args:
            hidden_T [B,L_t,1536], hidden_V [B,L_v,256], hidden_A [B,L_a,256]
            senti_T/V/A [B,L_m]
            mask_*: [B,L] True=padding (key_padding_mask convention)
        Returns:
            senti_ref_T [B,L_t], senti_ref_V [B,L_v], senti_ref_A [B,L_a]
        """
        B, L_t, _ = hidden_T.shape
        device = hidden_T.device
        T_proj = self.proj_t(hidden_T)  # [B, L_t, 256]
        if mask_T is None:
            mask_T = torch.zeros(B, L_t, dtype=torch.bool, device=device)
        kp_T = mask_T  # True=padding
        kp_V = mask_V if mask_V is not None else torch.zeros(B, hidden_V.shape[1], dtype=torch.bool, device=device)
        kp_A = mask_A if mask_A is not None else torch.zeros(B, hidden_A.shape[1], dtype=torch.bool, device=device)

        _, attn_vt = self.attn_v2t(hidden_V, T_proj, T_proj, key_padding_mask=kp_T, average_attn_weights=True)
        senti_ref_V = torch.bmm(attn_vt, senti_T.unsqueeze(-1)).squeeze(-1)  # [B, L_v]

        _, attn_at = self.attn_a2t(hidden_A, T_proj, T_proj, key_padding_mask=kp_T, average_attn_weights=True)
        senti_ref_A = torch.bmm(attn_at, senti_T.unsqueeze(-1)).squeeze(-1)  # [B, L_a]

        _, attn_tv = self.attn_t2v(T_proj, hidden_V, hidden_V, key_padding_mask=kp_V, average_attn_weights=True)
        ref_t_from_v = torch.bmm(attn_tv, senti_V.unsqueeze(-1)).squeeze(-1)  # [B, L_t]
        _, attn_ta = self.attn_t2a(T_proj, hidden_A, hidden_A, key_padding_mask=kp_A, average_attn_weights=True)
        ref_t_from_a = torch.bmm(attn_ta, senti_A.unsqueeze(-1)).squeeze(-1)  # [B, L_t]
        senti_ref_T = 0.5 * (ref_t_from_v + ref_t_from_a)
        return senti_ref_T, senti_ref_V, senti_ref_A


def _entropy_per_token(posteriors, eps=1e-8):
    """
    对每个 token 的分布计算熵 H(p) = -sum(p*log(p)).
    Args:
        posteriors: [B, L, C]
    Returns:
        H: [B, L]
    """
    return -torch.sum(posteriors * torch.log(posteriors + eps), dim=-1)


def _reliability_from_entropy(posteriors, eps=1e-8):
    """
    Rel_i = 1 - H(p_i) / log(C), 归一化到 [0, 1], 熵越小越可靠.
    Args:
        posteriors: [B, L, C]
    Returns:
        Rel: [B, L]
    """
    H = _entropy_per_token(posteriors, eps)
    C = posteriors.shape[-1]
    H_max = torch.log(torch.tensor(float(C), device=posteriors.device, dtype=posteriors.dtype))
    H_norm = (H / H_max.clamp(min=1e-8)).clamp(0.0, 1.0)
    return 1.0 - H_norm


class EvidenceSplitter(nn.Module):
    """
    证据拆分: valid = mask & (Rel>=rel_min), d=|senti-ref|, 分位数阈值
    conf_mask = valid & (d >= thr_conf), con_mask = valid & (d <= thr_con)
    改进: 使用更激进的分位数,增强冲突/一致证据的区分度
    """
    def __init__(self, tau_conf=0.3, tau_con=0.1, tau_rel=0.5,
                 conf_ratio=0.30, con_ratio=0.30, rel_min=0.15,
                 conf_percentile=0.3, con_percentile=0.2,
                 min_conf_ratio=0.05, min_con_ratio=0.05):
        super().__init__()
        self.tau_conf = tau_conf
        self.tau_con = tau_con
        self.tau_rel = tau_rel
        self.conf_ratio = conf_ratio  # 提升到0.30,选更多冲突证据
        self.con_ratio = con_ratio    # 提升到0.30,选更多一致证据
        self.rel_min = rel_min        # 降低到0.15,包含更多中等可靠度token
        self.conf_percentile = conf_percentile
        self.con_percentile = con_percentile
        self.min_conf_ratio = min_conf_ratio
        self.min_con_ratio = min_con_ratio

    def forward(self, posteriors_dict, senti_scores_dict, senti_ref_per_token, mask_dict=None):
        """
        Args:
            posteriors_dict: {'T'/'A'/'V': [B, L_m, C]}
            senti_scores_dict: {'T'/'A'/'V': [B, L_m]}
            senti_ref_per_token: {'T'/'A'/'V': [B, L_m]} 对齐感知局部参照
            mask_dict: {'T'/'A'/'V': [B, L_m]} True=valid (若None则全有效)
        """
        con_masks = {}
        conf_masks = {}
        for modality in ['T', 'A', 'V']:
            posteriors = posteriors_dict[modality]
            senti_scores = senti_scores_dict[modality]
            senti_ref = senti_ref_per_token[modality]
            B, L = senti_scores.shape
            device = senti_scores.device

            Rel = _reliability_from_entropy(posteriors)
            mask = mask_dict[modality] if mask_dict and modality in mask_dict and mask_dict[modality] is not None else torch.ones(B, L, dtype=torch.bool, device=device)
            valid = mask & (Rel >= self.rel_min)
            d = torch.abs(senti_scores - senti_ref)

            conf_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            con_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            for b in range(B):
                d_valid = d[b][valid[b]]
                if d_valid.numel() == 0:
                    conf_mask[b, 0] = True
                    con_mask[b, min(1, L - 1)] = True
                    continue
                thr_conf = torch.quantile(d_valid.float(), 1.0 - self.conf_ratio)
                thr_con = torch.quantile(d_valid.float(), self.con_ratio)
                conf_mask[b] = valid[b] & (d[b] >= thr_conf)
                con_mask[b] = valid[b] & (d[b] <= thr_con)
                overlap = conf_mask[b] & con_mask[b]
                if overlap.any():
                    for oi in overlap.nonzero(as_tuple=True)[0]:
                        if d[b, oi].item() > d[b][valid[b]].median().item():
                            con_mask[b, oi] = False
                        else:
                            conf_mask[b, oi] = False
                # 保底在重叠解析之后执行，确保至少各 1 个
                if conf_mask[b].sum() == 0:
                    conf_mask[b, d[b].argmax().item()] = True
                if con_mask[b].sum() == 0:
                    con_mask[b, d[b].argmin().item()] = True
            con_masks[modality] = con_mask
            conf_masks[modality] = conf_mask
        return con_masks, conf_masks


class EvidenceLevelJS(nn.Module):
    """
    Evidence-level JS计算: 在证据后验上计算Jensen-Shannon散度
    """
    def __init__(self):
        super().__init__()
    
    def aggregate_posteriors(self, posteriors, mask, confidence=None):
        """
        聚合证据集合的后验分布
        Args:
            posteriors: [B, L, C]
            mask: [B, L] bool mask
            confidence: [B, L] 置信度权重(可选)
        Returns:
            P_agg: [B, C] 聚合后验
        """
        B, L, C = posteriors.shape
        
        # 将mask外的token权重设为0
        weights = mask.float()  # [B, L]
        if confidence is not None:
            weights = weights * confidence
        
        # 检查每个样本是否有有效证据
        weight_sum = weights.sum(dim=-1, keepdim=True)  # [B, 1]
        has_evidence = (weight_sum > 1e-6)  # [B, 1]
        
        # 对有证据的样本做归一化加权平均
        safe_weights = weights / weight_sum.clamp(min=1e-8)  # [B, L]
        P_agg = torch.sum(
            posteriors * safe_weights.unsqueeze(-1),  # [B, L, C] * [B, L, 1]
            dim=1
        )  # [B, C]
        
        # 对无证据的样本返回均匀分布(而非噪声主导的假后验)
        uniform = torch.ones(B, C, device=posteriors.device) / C
        P_agg = torch.where(has_evidence, P_agg, uniform)
        
        return P_agg
    
    def jensen_shannon_divergence(self, P_list):
        """
        三模态 JSD_3: M=(P_T+P_A+P_V)/3, JSD=mean(KL(P_m||M))
        """
        M = torch.stack(P_list).mean(dim=0)
        eps = 1e-8
        js_values = [torch.sum(P * torch.log((P + eps) / (M + eps)), dim=-1) for P in P_list]
        return torch.stack(js_values).mean(dim=0)

    def jensen_shannon_divergence_2(self, P, Q):
        """JSD_2(P,Q): 两分布, 用于 C_m = JSD_2(P_conf_m, P_conf_T)/ln(2)"""
        M = 0.5 * (P + Q)
        eps = 1e-8
        return 0.5 * (torch.sum(P * torch.log((P + eps) / (M + eps)), dim=-1) +
                      torch.sum(Q * torch.log((Q + eps) / (M + eps)), dim=-1))
    
    def kl_divergence_pairwise(self, P_list):
        """
        计算三模态两两 KL 散度的平均, 作为跨模态分歧标量 (与 JS 同形状, 供消融 JS vs KL).
        Args:
            P_list: [P_T, P_A, P_V], 每个 [B, C]
        Returns:
            KL_inter: [B,]
        """
        eps = 1e-8
        # 两两: (T,A), (T,V), (A,V)
        pairs = [(0, 1), (0, 2), (1, 2)]
        kls = []
        for i, j in pairs:
            P, Q = P_list[i], P_list[j]
            # KL(P || Q) = sum(P * log(P/Q))
            kl = torch.sum(P * torch.log((P + eps) / (Q + eps)), dim=-1)  # [B,]
            kls.append(kl)
        return torch.stack(kls).mean(dim=0)  # [B,]
    
    def entropy(self, P):
        """
        计算分布的熵 H(P) = -Σ P*log(P)
        P: [B, C]
        Returns: [B,]
        """
        eps = 1e-8
        return -torch.sum(P * torch.log(P + eps), dim=-1)

    def forward(self, posteriors_dict, conf_masks, confidence_dict=None, divergence_type='js'):
        """
        Args:
            posteriors_dict: {'T'/'A'/'V': [B, L_m, C]}
            conf_masks: {'T'/'A'/'V': [B, L_m]} 冲突证据mask
            confidence_dict: {'T'/'A'/'V': [B, L_m]} 置信度(可选)
            divergence_type: 'js' | 'kl' 冲突度量, 供消融
        Returns:
            inter_div: [B,] 跨模态散度 (JS 或 KL)
            JS_intra: {'T'/'A'/'V': [B,]} 模态内冲突(熵)
            P_conf_dict: {'T'/'A'/'V': [B, C]} 冲突证据聚合后验
        """
        P_conf_list = []
        P_conf_dict = {}
        JS_intra = {}
        
        for modality in ['T', 'A', 'V']:
            posteriors = posteriors_dict[modality]
            mask = conf_masks[modality]
            conf = confidence_dict[modality] if confidence_dict else None
            
            P_conf = self.aggregate_posteriors(posteriors, mask, conf)
            P_conf_list.append(P_conf)
            P_conf_dict[modality] = P_conf
            
            # 模态内冲突: 冲突证据聚合后验的熵(熵越高=模态内部越混乱)
            JS_intra[modality] = self.entropy(P_conf)  # [B,]
        
        if divergence_type == 'kl':
            inter_div = self.kl_divergence_pairwise(P_conf_list)
        else:
            inter_div = self.jensen_shannon_divergence(P_conf_list)
        
        return inter_div, JS_intra, P_conf_dict


class ConflictIntensity(nn.Module):
    """
    冲突强度: C = JSD_3/ln(3); C_m = JSD_2(P_conf_m, P_conf_T)/ln(2)
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer('ln3', torch.log(torch.tensor(3.0)))
        self.register_buffer('ln2', torch.log(torch.tensor(2.0)))
        self.js_calc = EvidenceLevelJS()

    def forward(self, JS_inter, P_conf_dict=None):
        """
        Args:
            JS_inter: [B,] 跨模态 JSD_3
            P_conf_dict: {'T'/'A'/'V': [B, C]} 用于计算 C_m
        Returns:
            C: [B,], C_m: {'T'/'A'/'V': [B,]}
        """
        C = torch.clamp(JS_inter / self.ln3, 0.0, 1.0)
        C_m = {}
        if P_conf_dict is not None:
            P_T = P_conf_dict['T']
            for m in ['T', 'A', 'V']:
                P_m = P_conf_dict[m]
                jsd2 = self.js_calc.jensen_shannon_divergence_2(P_m, P_T)
                C_m[m] = torch.clamp(jsd2 / self.ln2, 0.0, 1.0)
            C_T = 0.5 * (C_m['A'] + C_m['V'])
            C_m['T'] = C_T
        else:
            for m in ['T', 'A', 'V']:
                C_m[m] = C
        return C, C_m


class ConflictJSModule(nn.Module):
    """
    完整的Conflict-JS模块: AlignmentAwareReference + EvidenceSplitter + JS + ConflictIntensity
    """
    def __init__(self, tau_conf=0.3, tau_con=0.1, tau_rel=0.5, num_classes=7, conflict_metric='js',
                 rel_min=0.20, conf_ratio=0.25, con_ratio=0.25, use_alignment_ref=True):
        super().__init__()
        self.use_alignment_ref = use_alignment_ref
        if use_alignment_ref:
            self.align_ref = AlignmentAwareReference(text_dim=1536, align_dim=256, nhead=4)
        self.splitter = EvidenceSplitter(tau_conf, tau_con, tau_rel, conf_ratio=conf_ratio, con_ratio=con_ratio, rel_min=rel_min)
        self.js_calculator = EvidenceLevelJS()
        self.intensity = ConflictIntensity(num_classes)
        self.conflict_metric = conflict_metric

    def forward(self, posteriors_dict, senti_scores_dict, senti_ref_per_token=None,
                hidden_dict=None, mask_dict=None):
        """
        Args:
            posteriors_dict: {'T'/'A'/'V': [B, L_m, C]}
            senti_scores_dict: {'T'/'A'/'V': [B, L_m]}
            senti_ref_per_token: {'T'/'A'/'V': [B, L_m]} 若use_alignment_ref则由此或由align_ref计算
            hidden_dict: {'T'/'A'/'V': hidden} 用于 AlignmentAwareReference
            mask_dict: {'T'/'A'/'V': [B, L_m]} True=valid
        """
        if senti_ref_per_token is None and self.use_alignment_ref and hidden_dict is not None:
            mask_T = ~mask_dict['T'] if mask_dict and mask_dict.get('T') is not None and isinstance(mask_dict['T'], torch.Tensor) else None
            mask_V = ~mask_dict['V'] if mask_dict and mask_dict.get('V') is not None and isinstance(mask_dict['V'], torch.Tensor) else None
            mask_A = ~mask_dict['A'] if mask_dict and mask_dict.get('A') is not None and isinstance(mask_dict['A'], torch.Tensor) else None
            if mask_T is not None and mask_T.numel() == 0:
                mask_T = None
            senti_ref_T, senti_ref_V, senti_ref_A = self.align_ref(
                hidden_dict['T'], hidden_dict['V'], hidden_dict['A'],
                senti_scores_dict['T'], senti_scores_dict['V'], senti_scores_dict['A'],
                mask_T=mask_T, mask_V=mask_V, mask_A=mask_A
            )
            senti_ref_per_token = {'T': senti_ref_T, 'V': senti_ref_V, 'A': senti_ref_A}
        elif senti_ref_per_token is None:
            senti_ref_per_token = {m: senti_scores_dict[m].mean(dim=1, keepdim=True).expand_as(senti_scores_dict[m]) for m in ['T', 'A', 'V']}

        con_masks, conf_masks = self.splitter(posteriors_dict, senti_scores_dict, senti_ref_per_token, mask_dict)

        confidence_dict = {m: _reliability_from_entropy(posteriors_dict[m]) for m in ['T', 'A', 'V']}
        inter_div, JS_intra, P_conf_dict = self.js_calculator(
            posteriors_dict, conf_masks, confidence_dict, divergence_type=self.conflict_metric
        )
        C, C_m = self.intensity(inter_div, P_conf_dict)

        valid_m = {}
        for m in ['T', 'A', 'V']:
            Rel = _reliability_from_entropy(posteriors_dict[m])
            mask = mask_dict[m] if mask_dict and mask_dict.get(m) is not None and isinstance(mask_dict.get(m), torch.Tensor) and mask_dict[m].numel() > 0 else torch.ones_like(Rel, dtype=torch.bool)
            valid_m[m] = mask & (Rel >= self.splitter.rel_min)
        rho_m = {}
        for m in ['T', 'A', 'V']:
            denom = valid_m[m].sum(dim=1).float().clamp(min=1e-6)
            rho_m[m] = (conf_masks[m].sum(dim=1).float() / denom).clamp(0.0, 1.0)
        rho = torch.stack([rho_m[m] for m in ['T', 'A', 'V']]).mean(dim=0)
        return C, C_m, con_masks, conf_masks, P_conf_dict, inter_div, rho, rho_m
