import copy
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.layers = n_layers
        self.activation = getattr(nn, activation)

    def forward(self, x, y):
        x = torch.mean(x, dim=-2)
        y = torch.mean(y, dim=-2)
        x_pred = y

        # normalize to unit sphere (clamp to avoid 0/0 -> NaN)
        eps = 1e-8
        x_pred = x_pred / (x_pred.norm(dim=1, keepdim=True).clamp(min=eps))
        x = x / (x.norm(dim=1, keepdim=True).clamp(min=eps))

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = -(pos - neg).mean()
        return nce


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, key_padding_mask=None):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if key_padding_mask is not None:
            # key_padding_mask: [B, K_len], True = padding (should be masked)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K_len]
            dots = dots.masked_fill(mask, float('-inf'))

        attn = self.attend(dots)
        # When all keys are padding, softmax(-inf,...,-inf) = NaN. Replace with 0.
        attn = attn.nan_to_num(0.0)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, source_x, target_x, key_padding_mask=None):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x, key_padding_mask=key_padding_mask)
        target_x = self.layernorm1(target_x_tmp + target_x)
        target_x = self.layernorm2(self.ffn(target_x) + target_x)
        return target_x


class DyRout_block(nn.Module):
    def __init__(self, opt, dropout):
        super(DyRout_block, self).__init__()
        # Conflict branch: Cross-Attention on conflict evidence tokens
        self.f_t_conf = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_v_conf = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_a_conf = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)

        # Congruent branch: Cross-Attention on congruent evidence tokens
        self.f_t_con = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_v_con = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_a_con = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)

        self.layernorm_conf = nn.LayerNorm(opt.hidden_size)
        self.layernorm_con = nn.LayerNorm(opt.hidden_size)

        # Per-modality gate: alpha_m = sigmoid(k*(C_m*rho_m - tau))
        # 改进: 降低gate_k使门控更线性,避免饱和; 降低gate_tau使更容易触发冲突分支
        gate_k = getattr(opt, 'gate_k', 5.0)   # 从10降到5,更线性
        gate_tau = getattr(opt, 'gate_tau', 0.08)  # 从0.15降到0.08,更容易触发
        self.gate_k = nn.Parameter(torch.tensor(float(gate_k)))
        self.gate_tau = nn.Parameter(torch.tensor(float(gate_tau)))

    def forward(self, source, t_con, t_conf, v_con, v_conf, a_con, a_conf,
                conflict_C=None, conflict_C_m=None, conflict_rho=None, conflict_rho_m=None,
                t_con_pad=None, t_conf_pad=None, v_con_pad=None, v_conf_pad=None,
                a_con_pad=None, a_conf_pad=None):
        if conflict_C is not None:
            # --- Conflict branch: CrossAttn on conf streams ---
            cross_conf_t = self.f_t_conf(target_x=source, source_x=t_conf, key_padding_mask=t_conf_pad)
            cross_conf_v = self.f_v_conf(target_x=source, source_x=v_conf, key_padding_mask=v_conf_pad)
            cross_conf_a = self.f_a_conf(target_x=source, source_x=a_conf, key_padding_mask=a_conf_pad)

            # --- Congruent branch: CrossAttn on con streams ---
            cross_con_t = self.f_t_con(target_x=source, source_x=t_con, key_padding_mask=t_con_pad)
            cross_con_v = self.f_v_con(target_x=source, source_x=v_con, key_padding_mask=v_con_pad)
            cross_con_a = self.f_a_con(target_x=source, source_x=a_con, key_padding_mask=a_con_pad)

            # --- Per-modality gating: alpha_m = sigmoid(k * (C_m - tau)) ---
            if conflict_C_m is not None:
                alpha_t = torch.sigmoid(self.gate_k * (conflict_C_m['T'] - self.gate_tau)).view(-1, 1, 1)
                alpha_v = torch.sigmoid(self.gate_k * (conflict_C_m['V'] - self.gate_tau)).view(-1, 1, 1)
                alpha_a = torch.sigmoid(self.gate_k * (conflict_C_m['A'] - self.gate_tau)).view(-1, 1, 1)
            else:
                alpha_t = torch.sigmoid(self.gate_k * (conflict_C - self.gate_tau)).view(-1, 1, 1)
                alpha_v = alpha_t
                alpha_a = alpha_t

            # Conflict path: weighted sum of per-modality conflict CrossAttn outputs
            h_conf = self.layernorm_conf(
                alpha_t * cross_conf_t + alpha_v * cross_conf_v + alpha_a * cross_conf_a
            )
            # Congruent path: weighted sum of per-modality congruent CrossAttn outputs
            h_con = self.layernorm_con(
                (1 - alpha_t) * cross_con_t + (1 - alpha_v) * cross_con_v + (1 - alpha_a) * cross_con_a
            )
            # Final fusion: C-gated combination of conflict and congruent paths
            conf_weight = torch.sigmoid(self.gate_k * (conflict_C - self.gate_tau)).view(-1, 1, 1)
            output = conf_weight * h_conf + (1 - conf_weight) * h_con

            # 保存门控值供可视化（在推理时通过 model.last_gate_* 访问）
            self._last_conf_weight = conf_weight.detach().squeeze(-1).squeeze(-1)  # [B]
            self._last_alpha = {
                'T': alpha_t.detach().squeeze(-1).squeeze(-1),  # [B]
                'V': alpha_v.detach().squeeze(-1).squeeze(-1),
                'A': alpha_a.detach().squeeze(-1).squeeze(-1),
            }
        else:
            # Fallback: no conflict info, use conf streams as full features
            cross_f_t = self.f_t_conf(target_x=source, source_x=t_conf, key_padding_mask=t_conf_pad)
            cross_f_v = self.f_v_conf(target_x=source, source_x=v_conf, key_padding_mask=v_conf_pad)
            cross_f_a = self.f_a_conf(target_x=source, source_x=a_conf, key_padding_mask=a_conf_pad)
            output = self.layernorm_conf(cross_f_t + cross_f_v + cross_f_a)
        return output


class DyRoutTrans_block(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans_block, self).__init__()
        self.mhatt1 = DyRout_block(opt, dropout=0.3)
        self.mhatt2 = MultiHAtten(opt.hidden_size, dropout=0.)
        self.ffn = FeedForward(opt.hidden_size, opt.ffn_size, dropout=0.)

        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, source, t_con, t_conf, v_con, v_conf, a_con, a_conf,
                conflict_C=None, conflict_C_m=None, conflict_rho=None, conflict_rho_m=None,
                t_con_pad=None, t_conf_pad=None, v_con_pad=None, v_conf_pad=None,
                a_con_pad=None, a_conf_pad=None):
        source = self.norm1(source + self.mhatt1(
            source, t_con, t_conf, v_con, v_conf, a_con, a_conf,
            conflict_C=conflict_C, conflict_C_m=conflict_C_m,
            conflict_rho=conflict_rho, conflict_rho_m=conflict_rho_m,
            t_con_pad=t_con_pad, t_conf_pad=t_conf_pad,
            v_con_pad=v_con_pad, v_conf_pad=v_conf_pad,
            a_con_pad=a_con_pad, a_conf_pad=a_conf_pad
        ))
        source = self.norm2(source + self.mhatt2(q=source, k=source, v=source))
        source = self.norm3(source + self.ffn(source))
        return source


class DyRoutTrans(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans, self).__init__()
        self.opt = opt

        # Length Align
        self.len_t = nn.Linear(opt.seq_lens[0], opt.seq_lens[0])
        self.len_v = nn.Linear(opt.seq_lens[1], opt.seq_lens[0])
        self.len_a = nn.Linear(opt.seq_lens[2], opt.seq_lens[0])

        # Dimension Align
        self.dim_t = nn.Linear(768*2, 256)
        self.dim_v = nn.Linear(256, 256)
        self.dim_a = nn.Linear(256, 256)

        fusion_block = DyRoutTrans_block(opt)
        self.dec_list = self._get_clones(fusion_block, 3)

        self.cpc_ft = CPC(x_size=256, y_size=256)
        self.cpc_fv = CPC(x_size=256, y_size=256)
        self.cpc_fa = CPC(x_size=256, y_size=256)

    def _length_align(self, x, len_layer, target_len):
        """
        Align sequence length. Use learned Linear if input matches expected size,
        otherwise fall back to adaptive_avg_pool1d (handles pruned sequences).
        x: [B, L, D] -> [B, target_len, D]
        """
        x_perm = x.permute(0, 2, 1)  # [B, D, L]
        if x_perm.shape[-1] == len_layer.in_features:
            return len_layer(x_perm).permute(0, 2, 1)
        else:
            return F.adaptive_avg_pool1d(x_perm, target_len).permute(0, 2, 1)

    def _apply_mask_select(self, hidden, mask):
        """
        Apply boolean mask to select tokens, producing a dense sub-sequence.
        hidden: [B, L, D], mask: [B, L] (bool, True=keep)
        Returns:
            out: [B, K, D] where K = max number of True across batch (padded with zeros)
            pad_mask: [B, K] bool, True = padding position (for key_padding_mask in attention)
        """
        if mask is None:
            return hidden, None
        B, L, D = hidden.shape
        counts = mask.sum(dim=1)  # [B,]
        K = max(int(counts.max().item()), 1)
        out = torch.zeros(B, K, D, device=hidden.device, dtype=hidden.dtype)
        pad_mask = torch.ones(B, K, device=hidden.device, dtype=torch.bool)  # True=padding
        for b in range(B):
            idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            n = idx.shape[0]
            if n > 0:
                out[b, :n] = hidden[b, idx]
                pad_mask[b, :n] = False  # real tokens are NOT padding
        return out, pad_mask

    def forward(self, uni_fea, uni_mask, conflict_C=None, conflict_C_m=None,
                conflict_rho=None, conflict_rho_m=None,
                con_masks=None, conf_masks=None):
        target_len = self.opt.seq_lens[0]

        # Dimension align (full features)
        feat_t = self.dim_t(uni_fea['T'])  # [B, L_t, D]
        feat_v = self.dim_v(uni_fea['V'])  # [B, L_v, D]
        feat_a = self.dim_a(uni_fea['A'])  # [B, L_a, D]

        # Padding masks for con/conf streams (None = no masking needed)
        t_con_pad = t_conf_pad = v_con_pad = v_conf_pad = a_con_pad = a_conf_pad = None

        if con_masks is not None and conf_masks is not None:
            # --- Dual-stream: extract dense sub-sequences with padding masks ---
            # Do NOT length-align con/conf streams; pass at natural length K
            # so CrossAttention can use key_padding_mask to ignore zero-padded positions
            t_con, t_con_pad = self._apply_mask_select(feat_t, con_masks['T'])
            t_conf, t_conf_pad = self._apply_mask_select(feat_t, conf_masks['T'])
            v_con, v_con_pad = self._apply_mask_select(feat_v, con_masks['V'])
            v_conf, v_conf_pad = self._apply_mask_select(feat_v, conf_masks['V'])
            a_con, a_con_pad = self._apply_mask_select(feat_a, con_masks['A'])
            a_conf, a_conf_pad = self._apply_mask_select(feat_a, conf_masks['A'])
        else:
            # Fallback: no conflict info, use full length-aligned features as conf, con = zeros
            t_conf = self._length_align(feat_t, self.len_t, target_len)
            v_conf = self._length_align(feat_v, self.len_v, target_len)
            a_conf = self._length_align(feat_a, self.len_a, target_len)
            t_con = torch.zeros_like(t_conf)
            v_con = torch.zeros_like(v_conf)
            a_con = torch.zeros_like(a_conf)

        # Source = sum of all aligned full features (for CPC and initial source)
        hidden_t = self._length_align(feat_t, self.len_t, target_len)
        hidden_v = self._length_align(feat_v, self.len_v, target_len)
        hidden_a = self._length_align(feat_a, self.len_a, target_len)
        source = hidden_t + hidden_v + hidden_a

        for i, dec in enumerate(self.dec_list):
            source = dec(source, t_con, t_conf, v_con, v_conf, a_con, a_conf,
                         conflict_C=conflict_C, conflict_C_m=conflict_C_m,
                         conflict_rho=conflict_rho, conflict_rho_m=conflict_rho_m,
                         t_con_pad=t_con_pad, t_conf_pad=t_conf_pad,
                         v_con_pad=v_con_pad, v_conf_pad=v_conf_pad,
                         a_con_pad=a_con_pad, a_conf_pad=a_conf_pad)

        nce_t = self.cpc_ft(hidden_t, source)
        nce_v = self.cpc_fv(hidden_v, source)
        nce_a = self.cpc_fa(hidden_a, source)
        nce_loss = nce_t + nce_v + nce_a

        # 暴露最后一个 block 的门控值供 OverallModal 透传给可视化
        last_blk = self.dec_list[-1].mhatt1
        if hasattr(last_blk, '_last_conf_weight'):
            self.last_gate_conf_weight = last_blk._last_conf_weight  # [B]
            self.last_gate_alpha = last_blk._last_alpha               # {'T','V','A': [B]}

        return source, nce_loss

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SentiCLS(nn.Module):
    def __init__(self, opt):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, fusion_features):
        fusion_features = torch.mean(fusion_features, dim=-2)
        output = self.cls_layer(fusion_features)
        return output
