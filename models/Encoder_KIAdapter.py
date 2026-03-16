import copy
import torch
from torch import nn, einsum
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from transformers import AutoConfig, AutoModel, AutoTokenizer


def _sanitize_key_padding_mask(mask, device=None):
    """
    防止 key_padding_mask 全 True 导致注意力 softmax(-inf) -> NaN。
    如果某一行全是 True，则强制保留最后一个位置为 False，确保至少有一个可见 token。
    """
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    elif device is not None and mask.device != device:
        mask = mask.to(device)
    mask = mask.bool()
    if mask.dim() >= 2 and mask.numel() > 0:
        all_pad = mask.view(mask.size(0), -1).all(dim=1)
        if all_pad.any():
            mask = mask.clone()
            mask[all_pad, -1] = False
    return mask


class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(Classifier, self).__init__()
        ModuleList = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                ModuleList.append(nn.Linear(input_size, h))
                ModuleList.append(nn.GELU())
            else:
                ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                ModuleList.append(nn.GELU())
        ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = torch.mean(x, dim=-2)
        x = self.dropout(x)
        output = self.MLP(x)
        return output


class PositionEncoding(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super(PositionEncoding, self).__init__()
        # self.cls_token = nn.parameter.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.parameter.Parameter(torch.zeros(1, num_patches, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        # batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # 支持可变序列长度：当实际 seq_len < num_patches 时，只取前 seq_len 个位置编码
        seq_len = embeddings.size(1)
        pos_emb = self.position_embeddings[:, :seq_len, :].to(embeddings.device)
        embeddings = embeddings + pos_emb
        embeddings = self.dropout(embeddings)
        return embeddings



class TfEncoder(nn.Module):
    def __init__(self, fea_size, num_patches, nhead, dim_feedforward, num_layers, pos_dropout=0., tf_dropout=0.2):
        super(TfEncoder, self).__init__()
        self.pos_encoder = PositionEncoding(
            num_patches=num_patches,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=pos_dropout
        )

        tfencoder_layer = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward // 2, dropout=tf_dropout, activation='gelu', batch_first=True)
        self.tfencoder = TransformerEncoder(tfencoder_layer, num_layers)

    def forward(self, src, src_key_padding_mask):
        # 手动逐层运行以保留每层隐藏状态（torch.nn.TransformerEncoder 默认只返回最终输出）
        src_key_padding_mask = _sanitize_key_padding_mask(src_key_padding_mask, device=src.device)
        src = self.pos_encoder(src)

        output = src
        hidden_state_list = [output]  # 与你截图一致：包含初始输入 + 每层输出
        for layer in self.tfencoder.layers:
            output = layer(output, src_mask=None, src_key_padding_mask=src_key_padding_mask)
            hidden_state_list.append(output)

        if self.tfencoder.norm is not None:
            output = self.tfencoder.norm(output)

        return output, hidden_state_list


class UniEncoder(nn.Module):
    def __init__(self, m, pretrained, num_patches, fea_size, nhead, dim_feedforward, num_layers, hf_cache_dir=None):
        super(UniEncoder, self).__init__()
        self.m = m
        self.hf_cache_dir = hf_cache_dir

        if m in "VA":
            self.tfencoder = TfEncoder(
                fea_size=fea_size,
                num_patches=num_patches,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers
            )
            self.layernorm = nn.LayerNorm(dim_feedforward)
        else:
            # 使用 AutoModel/AutoTokenizer 支持 BERT、RoBERTa、chinese-roberta-wwm-ext 等
            # 需与 pkl 生成时的 tokenizer 一致
            self.model_config = AutoConfig.from_pretrained(
                pretrained,
                output_hidden_states=True,
                cache_dir=self.hf_cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained,
                cache_dir=self.hf_cache_dir
            )
            self.model = AutoModel.from_pretrained(
                pretrained,
                config=self.model_config,
                cache_dir=self.hf_cache_dir
            )

    def forward(self, inputs, key_padding_mask):
        if self.m in "VA":
            key_padding_mask = _sanitize_key_padding_mask(key_padding_mask, device=inputs.device)
            tf_last_hidden_state, _ = self.tfencoder(inputs, src_key_padding_mask=key_padding_mask)
            output = self.layernorm(tf_last_hidden_state)  # [B, L, dim_feedforward]
            return output
        else:
            input_ids = inputs[:, 0, :].long()
            input_mask = inputs[:, 1, :].float()
            segment_ids = inputs[:, 2, :].long()
            hidden_states = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids
            )
            output = hidden_states['last_hidden_state']  # [B, L, 768]

            # 更新 BERT mask 矩阵（供后续 attention 使用）
            key_padding_mask = []
            for sen in input_mask:
                mask = [not bool(item) for item in sen]
                key_padding_mask.append(mask)
            key_padding_mask = torch.tensor(key_padding_mask, device=inputs.device)
            key_padding_mask = _sanitize_key_padding_mask(key_padding_mask, device=inputs.device)

            return output, key_padding_mask

    def _get_tokenizer(self):
        return self.tokenizer

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class UniPretrain(nn.Module):
    def __init__(self, modality, num_patches, pretrained='./pretrainedModel/BERT/bert-base-uncased', fea_size=709, proj_fea_dim=128, drop_out=0., hf_cache_dir=None):
        super(UniPretrain, self).__init__()
        self.m = modality
        if modality == "T":
            proj_fea_dim = 768

        self.encoder = UniEncoder(
            m=modality,
            pretrained=pretrained,
            fea_size=fea_size,
            num_patches=num_patches,
            nhead=8,
            dim_feedforward=proj_fea_dim,
            num_layers=4,
            hf_cache_dir=hf_cache_dir,
        )
        self.decoder = Classifier(
            input_size=proj_fea_dim,
            hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4), int(proj_fea_dim / 8)],
            output_size=1,
            drop_out=drop_out
        )

    def forward(self, inputs):
        uni_fea, key_padding_mask = inputs[self.m], inputs["mask"][self.m]

        if self.m == 'T':        # [B, L, H]
            uni_hidden, inputs["mask"][self.m] = self.encoder(uni_fea, key_padding_mask)
        else:
            uni_hidden = self.encoder(uni_fea, key_padding_mask)

        uni_pred = self.decoder(uni_hidden)     # [B, 1]
        return uni_hidden, uni_pred


class UnimodalEncoder(nn.Module):
    def __init__(self, opt, bert_pretrained='./pretrainedModel/BERT/bert-base-uncased'):
        super(UnimodalEncoder, self).__init__()
        hf_cache_dir = getattr(opt, "hf_cache_dir", None)
        fea_dims = getattr(opt, 'fea_dims', [768, 177, 25])  # T, V, A; sims: [768,177,25], mosi/mosei: [768,709,33]
        self.enc_t = UniPretrain(modality="T", pretrained=bert_pretrained, num_patches=opt.seq_lens[0], proj_fea_dim=768, hf_cache_dir=hf_cache_dir)
        self.enc_v = UniPretrain(modality="V", num_patches=opt.seq_lens[1], fea_size=fea_dims[1])
        self.enc_a = UniPretrain(modality="A", num_patches=opt.seq_lens[2], fea_size=fea_dims[2])

        # 情感投影头 (T: 768, V/A: 128，移除 KI 后不再 *2)
        from models.SentimentProjector import SentimentProjector
        num_classes = getattr(opt, 'senti_num_classes', 7)
        self.senti_proj_t = SentimentProjector(768, num_classes)
        self.senti_proj_v = SentimentProjector(128, num_classes)
        self.senti_proj_a = SentimentProjector(128, num_classes)

    def forward(self, inputs_data_mask):
        # Encoder Part
        hidden_t, uni_T_pre = self.enc_t(inputs_data_mask)
        hidden_v, uni_V_pre = self.enc_v(inputs_data_mask)
        hidden_a, uni_A_pre = self.enc_a(inputs_data_mask)
        
        # Phase 2: 生成情感后验分布
        posteriors_t, senti_t = self.senti_proj_t(hidden_t)  # [B, L_t, C], [B, L_t]
        posteriors_v, senti_v = self.senti_proj_v(hidden_v)  # [B, L_v, C], [B, L_v]
        posteriors_a, senti_a = self.senti_proj_a(hidden_a)  # [B, L_a, C], [B, L_a]
        
        posteriors = {'T': posteriors_t, 'V': posteriors_v, 'A': posteriors_a}
        senti_scores = {'T': senti_t, 'V': senti_v, 'A': senti_a}

        return {'T': hidden_t, 'V': hidden_v, 'A': hidden_a}, \
               {'T': uni_T_pre, 'V': uni_V_pre, 'A': uni_A_pre}, \
               posteriors, senti_scores
