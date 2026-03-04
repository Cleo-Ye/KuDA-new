import torch
import torch.nn.functional as F
from torch import nn
from models.Encoder_KIAdapter import UnimodalEncoder
from models.DyRoutFusion_CLS import DyRoutTrans, SentiCLS
from models.ConflictJS import ConflictJSModule, _reliability_from_entropy
from models.VisionTokenPruner import VisionTokenPruner, TextGuidedVisionPruner
from models.AudioTokenFilter import AudioTokenFilter


class KMSA(nn.Module):
    def __init__(self, opt, dataset, bert_pretrained='bert-base-uncased'):
        super(KMSA, self).__init__()
        self.opt = opt
        
        # Unimodal Encoder & Knowledge Inject Adapter
        self.UniEncKI = UnimodalEncoder(opt, bert_pretrained)

        # Phase 2: Conflict-JS模块(可选启用)
        if opt.use_conflict_js:
            self.conflict_js = ConflictJSModule(
                tau_conf=getattr(opt, 'tau_conf', 0.3),
                tau_con=getattr(opt, 'tau_con', 0.1),
                tau_rel=getattr(opt, 'tau_rel', 0.5),
                num_classes=getattr(opt, 'senti_num_classes', 7),
                conflict_metric=getattr(opt, 'conflict_metric', 'js'),
                rel_min=getattr(opt, 'rel_min', 0.20),
                conf_ratio=getattr(opt, 'conf_ratio', 0.25),
                con_ratio=getattr(opt, 'con_ratio', 0.25),
                use_alignment_ref=getattr(opt, 'use_alignment_ref', True),
                evidence_split_mode=getattr(opt, 'evidence_split_mode', 'topk'),
                min_conf_k=getattr(opt, 'min_conf_k', 4),
                min_con_k=getattr(opt, 'min_con_k', 4)
            )
        else:
            self.conflict_js = None

        # Vision Token Pruner (设计文档第3步: 单模态去冗余, 在ConflictJS之前)
        iec_mode = getattr(opt, 'iec_mode', 'text_guided')
        if getattr(opt, 'use_vision_pruning', False):
            if iec_mode == 'text_guided':
                self.vision_pruner = TextGuidedVisionPruner(
                    vision_keep_ratio=getattr(opt, 'vision_keep_ratio', 0.50),
                    hidden_dim=256,
                    text_dim=768 * 2,
                    nhead=4,
                )
            else:
                self.vision_pruner = VisionTokenPruner(
                    tau_rel=getattr(opt, 'tau_rel', 0.1),
                    target_ratio=getattr(opt, 'vision_target_ratio', 0.3),
                )
        else:
            self.vision_pruner = None

        # Audio 轻量压缩 (Phase 2 论文防守: 对照实验)
        if getattr(opt, 'use_audio_light_pruning', False):
            self.audio_filter = AudioTokenFilter(
                mode=getattr(opt, 'audio_filter_mode', 'top_r'),
                tau_rel=getattr(opt, 'audio_rel_threshold', 0.3),
                target_ratio=getattr(opt, 'audio_retain_ratio', 0.5),
                min_retain=getattr(opt, 'audio_min_retain', 10),
            )
        else:
            self.audio_filter = None

        # P0.4: Projection layers for con consistency loss (align different modality dims)
        con_proj_dim = getattr(opt, 'hidden_size', 256)
        if opt.use_conflict_js:
            self.con_proj_T = nn.Linear(768 * 2, con_proj_dim)
            self.con_proj_A = nn.Linear(256, con_proj_dim)
            self.con_proj_V = nn.Linear(256, con_proj_dim)

        # Multimodal Fusion
        self.DyMultiFus = DyRoutTrans(opt)

        # Output Classification for Sentiment Analysis
        self.CLS = SentiCLS(opt)

        # P2 招1: 边界敏感极性头 (提升 Acc-2/F1, 固定启用 per deterministic plan)
        self.polar_head = nn.Linear(256, 1)

    def forward(self, inputs_data_mask, multi_senti, gt_modal_labels=None):
        device = next(self.parameters()).device
        # Step 1-2: 编码 + 情感分布投影
        uni_fea, uni_senti, posteriors, senti_scores = self.UniEncKI(inputs_data_mask)
        uni_mask = inputs_data_mask['mask']
        self.last_uni_senti_text = uni_senti['T'].detach()
        
        # ============================================================
        # Step 3a: A 模块 —— 提前计算对齐权重（A→B 共享核心）
        # 当 use_conflict_js=True 时，AlignmentAwareReference 作为"A 模块"
        # 产生 attn_vt/attn_at，IEC（B 模块）和 ICR 共享同一套对齐信号
        # ============================================================
        precomputed_attn_vt = None
        precomputed_senti_ref = None

        if self.opt.use_conflict_js and self.conflict_js is not None and self.conflict_js.use_alignment_ref:
            # 计算文本 token 可靠度，用于 Rel_T 再加权
            rel_T = _reliability_from_entropy(posteriors['T'])  # [B, L_t]

            # 构建 padding mask（True=padding，符合 MHA key_padding_mask 约定）
            _mask_T = uni_mask.get('T')
            mask_T_pad = (~_mask_T) if (isinstance(_mask_T, torch.Tensor) and _mask_T.numel() > 0) else None
            _mask_V = uni_mask.get('V')
            mask_V_pad = (~_mask_V) if (isinstance(_mask_V, torch.Tensor) and _mask_V.numel() > 0) else None
            _mask_A = uni_mask.get('A')
            mask_A_pad = (~_mask_A) if (isinstance(_mask_A, torch.Tensor) and _mask_A.numel() > 0) else None

            senti_ref_T, senti_ref_V, senti_ref_A, attn_weights = self.conflict_js.align_ref(
                uni_fea['T'], uni_fea['V'], uni_fea['A'],
                senti_scores['T'], senti_scores['V'], senti_scores['A'],
                mask_T=mask_T_pad, mask_V=mask_V_pad, mask_A=mask_A_pad,
                rel_T=rel_T
            )
            precomputed_attn_vt = attn_weights['attn_vt']   # [B, L_v, L_t]
            precomputed_senti_ref = {'T': senti_ref_T, 'V': senti_ref_V, 'A': senti_ref_A}
            self.last_attn_weights = attn_weights            # 供可视化使用
            self.conflict_js._last_attn_weights = attn_weights

        # ============================================================
        # Step 3b: 单模态去冗余压缩（IEC = B 模块，消费 A 产生的 attn_vt）
        # ============================================================
        self.last_vision_original_len = uni_fea['V'].shape[1]
        if self.vision_pruner is not None:
            if isinstance(self.vision_pruner, TextGuidedVisionPruner):
                mask_v = ~uni_mask['V'] if isinstance(uni_mask.get('V'), torch.Tensor) and uni_mask['V'].numel() > 0 else None
                mask_t = ~uni_mask['T'] if isinstance(uni_mask.get('T'), torch.Tensor) and uni_mask['T'].numel() > 0 else None
                pruned_v, retained_idx, pruning_info = self.vision_pruner(
                    uni_fea['V'], uni_fea['T'], senti_scores['T'],
                    mask_v=mask_v, mask_t=mask_t,
                    precomputed_attn_vt=precomputed_attn_vt.detach() if precomputed_attn_vt is not None else None  # A→B 共享，detach 避免 IEC/ICR 梯度互扰
                )
            else:
                pruned_v, retained_idx, pruning_info = self.vision_pruner(
                    uni_fea['V'], posteriors['V']
                )
            uni_fea['V'] = pruned_v
            posteriors['V'] = torch.gather(
                posteriors['V'], 1,
                retained_idx.unsqueeze(-1).expand(-1, -1, posteriors['V'].shape[-1])
            )
            senti_scores['V'] = torch.gather(senti_scores['V'], 1, retained_idx)
            uni_mask['V'] = pruning_info['pruned_mask']
            self.last_pruning_info = pruning_info
            # 更新 precomputed_senti_ref 的 V 维度以匹配剪枝后长度
            if precomputed_senti_ref is not None:
                precomputed_senti_ref['V'] = torch.gather(
                    precomputed_senti_ref['V'], 1, retained_idx
                )

        # Step 3c: Audio 轻量压缩 (可选)
        if self.audio_filter is not None:
            filtered_a, retained_idx_a, audio_info = self.audio_filter(
                uni_fea['A'], posteriors['A']
            )
            uni_fea['A'] = filtered_a
            B_a, K_a, C_a = posteriors['A'].shape[0], retained_idx_a.shape[1], posteriors['A'].shape[-1]
            posteriors['A'] = torch.gather(
                posteriors['A'], 1,
                retained_idx_a.unsqueeze(-1).expand(-1, -1, C_a)
            )
            senti_scores['A'] = torch.gather(senti_scores['A'], 1, retained_idx_a)
            uni_mask['A'] = audio_info['pruned_mask']
            self.last_audio_pruning_info = audio_info
            # 更新 precomputed_senti_ref 的 A 维度
            if precomputed_senti_ref is not None:
                precomputed_senti_ref['A'] = torch.gather(
                    precomputed_senti_ref['A'], 1, retained_idx_a
                )

        # ============================================================
        # Step 4-5: Conflict-JS 冲突检测（ICR）
        # 若 A→B 共享已计算好 senti_ref，直接传入，跳过内部 align_ref 调用
        # ============================================================
        C = None
        C_m = None
        con_masks = None
        conf_masks = None
        senti_aux_loss = torch.tensor(0.0, device=device)
        js_loss = torch.tensor(0.0, device=device)
        con_loss = torch.tensor(0.0, device=device)
        cal_loss = torch.tensor(0.0, device=device)

        conflict_rho = None
        conflict_rho_m = None
        if self.opt.use_conflict_js and self.conflict_js is not None:
            mask_dict = dict(uni_mask)
            if not (isinstance(mask_dict.get('T'), torch.Tensor) and mask_dict['T'].numel() > 0):
                mask_dict['T'] = torch.ones(posteriors['T'].shape[0], posteriors['T'].shape[1], dtype=torch.bool, device=device)
            C, C_m, con_masks, conf_masks, P_conf_dict, inter_div, conflict_rho, conflict_rho_m = self.conflict_js(
                posteriors, senti_scores,
                senti_ref_per_token=precomputed_senti_ref,  # None 时内部自行计算
                hidden_dict=uni_fea if precomputed_senti_ref is None else None,
                mask_dict=mask_dict
            )
            self.last_JS_inter = inter_div.detach()
            self.last_conflict_intensity = C.detach()
            self.last_conflict_intensity_m = {m: C_m[m].detach() for m in C_m}
            self.last_con_masks = con_masks
            self.last_conf_masks = conf_masks

            # Contrastive divergence calibration: align inter_div with C
            # High-C (conflict) samples should have high JSD; low-C (congruent) low JSD.
            # Using C as a soft target avoids the circular-gradient issue
            # (C itself is derived from inter_div, so we detach it as pseudo-label).
            # This replaces the incorrect "-inter_div.mean()" which maximized divergence
            # unconditionally and directly fought senti_aux_loss.
            _ln3 = torch.log(torch.tensor(3.0, device=inter_div.device))
            inter_div_norm = (inter_div / _ln3).clamp(0.0, 1.0)
            with torch.no_grad():
                C_tgt = C.detach()
            js_loss = F.mse_loss(inter_div_norm, C_tgt)
            con_loss = self._compute_con_consistency_loss(uni_fea, con_masks)
            cal_loss = self._compute_calibration_loss(C, senti_scores, gt_modal_labels)

            if multi_senti is not None:
                senti_aux_loss = self._compute_senti_aux_loss(posteriors, multi_senti)
        
        # Step 6: 动态融合 - 传入冲突强度C/C_m、冲突密度rho和证据masks(双通路)
        multimodal_features, nce_loss = self.DyMultiFus(
            uni_fea, uni_mask,
            conflict_C=C,
            conflict_C_m=C_m,
            conflict_rho=conflict_rho,
            conflict_rho_m=conflict_rho_m,
            con_masks=con_masks,
            conf_masks=conf_masks
        )

        # 暴露门控值供可视化（来自 DyRoutTrans 最后一个 block）
        if hasattr(self.DyMultiFus, 'last_gate_conf_weight'):
            self.last_gate_conf_weight = self.DyMultiFus.last_gate_conf_weight  # [B]
            self.last_gate_alpha = self.DyMultiFus.last_gate_alpha              # {'T','V','A': [B]}

        # Sentiment Classification
        prediction = self.CLS(multimodal_features)

        # P2 招1: 极性头 logits (用于 L_cls = w(y)*BCE(sigmoid(z), I(y>0)))
        polar_logits = None
        if self.polar_head is not None:
            pooled = multimodal_features.mean(dim=1)  # [B, 256]
            polar_logits = self.polar_head(pooled)    # [B, 1]

        return prediction, nce_loss, senti_aux_loss, js_loss, con_loss, cal_loss, polar_logits
    
    def _compute_con_consistency_loss(self, uni_fea, con_masks):
        """
        P0.4: 一致性损失 - 鼓励跨模态congruent证据的表征对齐
        对每个模态用con_mask做masked mean pooling得到h_m_con,
        投影到公共空间后最小化两两之间的cosine距离
        """
        proj = {'T': self.con_proj_T, 'A': self.con_proj_A, 'V': self.con_proj_V}
        con_reps = []
        for m in ['T', 'A', 'V']:
            mask = con_masks[m].float().unsqueeze(-1)  # [B, L_m, 1]
            denom = mask.sum(dim=1).clamp(min=1.0)     # [B, 1]
            h_con = (uni_fea[m] * mask).sum(dim=1) / denom  # [B, D_m]
            h_con = proj[m](h_con)  # [B, hidden_size]
            con_reps.append(F.normalize(h_con, dim=-1))
        
        # Pairwise cosine similarity loss: maximize similarity = minimize (1 - cos_sim)
        loss = torch.tensor(0.0, device=con_reps[0].device)
        pairs = [(0, 1), (0, 2), (1, 2)]  # T-A, T-V, A-V
        for i, j in pairs:
            cos_sim = (con_reps[i] * con_reps[j]).sum(dim=-1)  # [B,]
            loss = loss + (1.0 - cos_sim).mean()
        return loss / len(pairs)
    
    def _compute_calibration_loss(self, C, senti_scores, gt_modal_labels=None):
        """
        P1+P2: 校准损失 - 让冲突强度C与真实模态分歧D对齐
        优先使用GT模态标签(labels_T/A/V)计算D, 回退到predicted senti_scores
        D = mean(|y_T-y_A|, |y_T-y_V|, |y_A-y_V|) 归一化到[0,1]
        L_cal = |C - D_norm|_1
        """
        if gt_modal_labels is not None:
            # P2: 使用GT模态标签 (更强、更稳定的监督)
            # labels_T/A/V 在SIMS中范围约[-1,1]
            y_T = gt_modal_labels['T'].view(-1)  # [B,]
            y_A = gt_modal_labels['A'].view(-1)  # [B,]
            y_V = gt_modal_labels['V'].view(-1)  # [B,]
        else:
            # 回退: 使用predicted senti_scores的均值
            y_T = senti_scores['T'].mean(dim=1)  # [B,]
            y_A = senti_scores['A'].mean(dim=1)  # [B,]
            y_V = senti_scores['V'].mean(dim=1)  # [B,]
        
        # 模态分歧: 两两绝对差的均值 (比std更直观, 对2个模态一致1个不一致也敏感)
        D = (torch.abs(y_T - y_A) + torch.abs(y_T - y_V) + torch.abs(y_A - y_V)) / 3.0
        D_TV = torch.abs(y_T - y_V)
        # 归一化到[0,1]: labels在[-1,1], 最大差值=2, 所以D_max=2
        D_norm = (D / 2.0).clamp(0.0, 1.0)
        D_TV_norm = (D_TV / 2.0).clamp(0.0, 1.0)
        # 可选: 仅对齐到 D_TV（文本-视觉分歧），便于先拉高 r(C, D_TV)
        D_target = D_TV_norm if getattr(self.opt, 'align_to_d_tv_only', False) else D_norm
        # L1 校准损失 + MSE 对齐 (L_align)，提升 r(C,D) 可辩护性
        cal_l1 = (C - D_target).abs().mean()
        align_mse = (C - D_target).pow(2).mean()
        lambda_align = getattr(self.opt, 'lambda_align', 0.1)
        cal_loss = cal_l1 + lambda_align * align_mse
        return cal_loss
    
    def _compute_senti_aux_loss(self, posteriors, labels):
        """
        Priority 2: 计算SentimentProjector的辅助监督损失
        将连续标签离散化为bins, 用CE loss监督各模态的平均后验
        """
        labels = labels.view(-1)  # [B,]
        num_classes = self.opt.senti_num_classes if hasattr(self.opt, 'senti_num_classes') else 7
        
        # 将连续标签[-1, 1]离散化为num_classes个bin
        boundaries = torch.linspace(-1.0, 1.0, num_classes + 1, device=labels.device)[1:-1]
        label_bins = torch.bucketize(labels, boundaries)  # [B,] 值域[0, num_classes-1]
        
        total_loss = torch.tensor(0.0, device=labels.device)
        count = 0
        for m in ['T', 'A', 'V']:
            # 对每个模态的token取平均后验的logits
            avg_posterior = posteriors[m].mean(dim=1)  # [B, C]
            # 用log后验做CE loss (posteriors已经是softmax后的)
            log_posterior = torch.log(avg_posterior + 1e-8)
            loss_m = F.nll_loss(log_posterior, label_bins)
            total_loss = total_loss + loss_m
            count += 1
        
        return total_loss / count

    def _load_pretrain_state(self, checkpoint_path, model):
        """加载预训练权重：只加载当前模型存在且形状一致的参数。
        兼容不同数据集（如 checkpoint 为 709 维视觉、当前为 177 维 SIMS）或不同 BERT 版本。"""
        ckpt = torch.load(checkpoint_path, weights_only=True)
        model_state = model.state_dict()
        ckpt_filtered = {}
        for k, v in ckpt.items():
            if k not in model_state:
                continue
            if v.shape != model_state[k].shape:
                continue
            ckpt_filtered[k] = v
        model.load_state_dict(ckpt_filtered, strict=False)

    def preprocess_model(self, pretrain_path):
        # 加载预训练模型
        self._load_pretrain_state(pretrain_path['T'], self.UniEncKI.enc_t)
        self._load_pretrain_state(pretrain_path['V'], self.UniEncKI.enc_v)
        self._load_pretrain_state(pretrain_path['A'], self.UniEncKI.enc_a)
        # 冻结外部知识注入参数
        for name, parameter in self.UniEncKI.named_parameters():
            if 'adapter' in name or 'decoder' in name:
                parameter.requires_grad = False


def build_model(opt):
    if 'sims' in opt.datasetName:
        l_pretrained = './pretrainedModel/BERT/bert-base-chinese'
    else:
        l_pretrained = './pretrainedModel/BERT/bert-base-uncased'

    model = KMSA(opt, dataset=opt.datasetName, bert_pretrained=l_pretrained)

    return model
