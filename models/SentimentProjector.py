"""
Phase 2: 情感投影头模块
将token表征投影到情感后验分布空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentimentProjector(nn.Module):
    """
    将任意维度的token表征投影到情感后验分布
    """
    def __init__(self, input_dim, num_classes=7, hidden_dim=128, dropout=0.1):
        """
        Args:
            input_dim: 输入token维度(如256*2=512 for KuDA)
            num_classes: 情感类别数(SIMS: 7个bins, 或回归任务设为离散bins)
            hidden_dim: 隐层维度
            dropout: dropout率
        """
        super().__init__()
        self.num_classes = num_classes
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 为回归任务定义情感bins的中心值(用于计算情感分数)
        # 固定 v = [-3,-2,-1,0,1,2,3]，后验期望比分段argmax更平滑，适合d_i连续差异
        centers = torch.tensor([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
        self.register_buffer('class_centers', centers[:num_classes] if num_classes <= 7 else torch.linspace(-3.0, 3.0, num_classes))
    
    def forward(self, hidden, return_logits=False):
        """
        Args:
            hidden: [B, L, D] token表征
            return_logits: 是否返回logits(用于loss计算)
        Returns:
            posteriors: [B, L, C] 情感后验分布(softmax)
            senti_scores: [B, L] 期望情感分数(加权和)
            logits: [B, L, C] (可选)
        """
        logits = self.projector(hidden)  # [B, L, num_classes]
        posteriors = F.softmax(logits, dim=-1)  # [B, L, num_classes]
        
        # 计算期望情感分数: Σ p_i * center_i
        senti_scores = torch.sum(
            posteriors * self.class_centers.view(1, 1, -1),
            dim=-1
        )  # [B, L]
        
        if return_logits:
            return posteriors, senti_scores, logits
        return posteriors, senti_scores
    
    def get_confidence(self, posteriors):
        """
        计算置信度: Rel_i = max(p_i)
        Args:
            posteriors: [B, L, C]
        Returns:
            confidence: [B, L]
        """
        return torch.max(posteriors, dim=-1)[0]
