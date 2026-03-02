"""
温度缩放校准模块
训练后校准: 使用验证集拟合最优温度T, 使 softmax(logits/T) 的概率更准确
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """
    温度缩放: 训练后在验证集上拟合单一温度参数T
    """
    def __init__(self, initial_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits):
        """
        Args:
            logits: [B, L, C] or [B, C]
        Returns:
            calibrated_posteriors: [B, L, C] or [B, C]
        """
        return F.softmax(logits / self.temperature, dim=-1)
    
    def fit(self, logits_list, labels_list, lr=0.01, max_iter=50):
        """
        在验证集上拟合最优温度
        Args:
            logits_list: List of [B, C] logits tensors
            labels_list: List of [B,] label indices
            lr: 学习率
            max_iter: 最大迭代次数
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits = torch.cat(logits_list, dim=0)  # [N, C]
        labels = torch.cat(labels_list, dim=0)  # [N,]
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        return self.temperature.item()


def compute_ece(posteriors, labels, n_bins=15):
    """
    计算Expected Calibration Error
    Args:
        posteriors: [N, C] 概率分布
        labels: [N,] 真实标签索引
        n_bins: 分桶数
    Returns:
        ece: float
        bin_acc: [n_bins] 每个桶的准确率
        bin_conf: [n_bins] 每个桶的平均置信度
        bin_count: [n_bins] 每个桶的样本数
    """
    confidences, predictions = torch.max(posteriors, dim=-1)
    accuracies = (predictions == labels).float()
    
    # 将置信度分为n_bins个桶
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_acc = []
    bin_conf = []
    bin_count = []
    
    for i in range(n_bins):
        # 找到落在当前桶的样本
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()
        
        if bin_size > 0:
            # 当前桶的平均准确率和置信度
            avg_acc = accuracies[in_bin].mean().item()
            avg_conf = confidences[in_bin].mean().item()
            ece += abs(avg_acc - avg_conf) * bin_size
            bin_acc.append(avg_acc)
            bin_conf.append(avg_conf)
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
        bin_count.append(bin_size)
    
    ece /= len(confidences)
    return ece, bin_acc, bin_conf, bin_count


def calibrate_and_evaluate(model, valid_loader, num_classes=7):
    """
    对模型进行温度缩放校准并评估ECE
    Args:
        model: 训练好的模型
        valid_loader: 验证集dataloader
        num_classes: 情感类别数
    Returns:
        temperature: 拟合的温度
        ece_before: 校准前的ECE
        ece_after: 校准后的ECE
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 收集验证集的logits和labels (使用情感分类任务)
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for data in valid_loader:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device).view(-1)
            
            # 获取情感后验分布 (通过UniEncKI的SentimentProjector)
            uni_fea, uni_senti, posteriors, senti_scores = model.UniEncKI(inputs)
            
            # 使用text模态的平均后验作为样本级情感分布
            avg_post = posteriors['T'].mean(dim=1)  # [B, C]
            # 将连续标签离散化为bins
            boundaries = torch.linspace(-1.0, 1.0, num_classes + 1, device=device)[1:-1]
            label_bins = torch.bucketize(label, boundaries)
            
            logits = torch.log(avg_post + 1e-8)  # 反推logits
            logits_list.append(logits.cpu())
            labels_list.append(label_bins.cpu())
    
    # 计算校准前的ECE
    posteriors_before = torch.cat([F.softmax(l, dim=-1) for l in logits_list], dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    ece_before, _, _, _ = compute_ece(posteriors_before, labels_all)
    
    # 拟合温度
    ts = TemperatureScaling()
    temperature = ts.fit(logits_list, labels_list)
    
    # 计算校准后的ECE
    with torch.no_grad():
        posteriors_after = torch.cat([ts(l) for l in logits_list], dim=0)
    ece_after, _, _, _ = compute_ece(posteriors_after, labels_all)
    
    return temperature, ece_before, ece_after
