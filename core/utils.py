import os
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate
from scipy import stats


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def ConfigLogging(file_path):
    # 创建一个 logger
    logger = logging.getLogger("save_option_results")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename=file_path, encoding='utf8')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_model(save_path, result, modality, model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_path = os.path.join(
        save_path,
        'DatasetName_{}_MAE-{}_Corr-{}.pth'.format(
            modality,
            result["MAE"],
            result["Corr"]
        )
    )
    torch.save(model.state_dict(), save_file_path)


def save_print_results(opt, logger, train_re, valid_re, test_re):
    if opt.datasetName in ['mosi', 'mosei']:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_7"], train_re["Has0_acc_2"], train_re["Non0_acc_2"], train_re["Has0_F1_score"], train_re["Non0_F1_score"]],
            ["Valid", valid_re["MAE"], valid_re["Corr"], valid_re["Mult_acc_7"], valid_re["Has0_acc_2"], valid_re["Non0_acc_2"], valid_re["Has0_F1_score"], valid_re["Non0_F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_7"], test_re["Has0_acc_2"], test_re["Non0_acc_2"], test_re["Has0_F1_score"], test_re["Non0_F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-7", "Acc-2", "Acc-2-N0", "F1", "F1-N0"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        logger.info(table.replace('\n', '\n\n'))
    else:
        results = [
            ["Train", train_re["MAE"], train_re["Corr"], train_re["Mult_acc_2"], train_re["Mult_acc_3"], train_re["Mult_acc_5"], train_re["F1_score"]],
            ["Valid", valid_re["MAE"], valid_re["Corr"], valid_re["Mult_acc_2"], valid_re["Mult_acc_3"], valid_re["Mult_acc_5"], valid_re["F1_score"]],
            ["Test", test_re["MAE"], test_re["Corr"], test_re["Mult_acc_2"], test_re["Mult_acc_3"], test_re["Mult_acc_5"], test_re["F1_score"]]
        ]
        headers = ["Phase", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]

        table = '\n' + tabulate(results, headers, tablefmt="grid") + '\n'
        if logger is not None:
            logger.info(table.replace('\n', '\n\n'))
        else:
            print(table)


def calculate_ratio_senti(uni_senti, multi_senti, k=2.):
    ratio = {}
    for m in ['T', 'V', 'A']:
        uni_senti[m] = torch.exp(-1 * k * torch.pow(torch.abs(uni_senti[m] - multi_senti), 2))

    # 进行归一化
    for m in ['T', 'V', 'A']:
        ratio[m] = uni_senti[m] / (uni_senti['T'] + uni_senti['V'] + uni_senti['A'])
        ratio[m] = ratio[m].unsqueeze(-1)

    return ratio


def calculate_u_test(pred, label):
    pred, label = pred.squeeze().numpy(), label.squeeze().numpy()
    label_mean = np.mean(label)
    alpha = 0.05

    pred_mean = np.mean(pred)
    pred_std = np.std(pred, ddof=1)
    label_std = np.std(label, ddof=1)
    # standard_error = pred_std / np.sqrt(len(pred))
    standard_error = np.sqrt(label_std / len(label) + pred_std / len(pred))

    Z = (label_mean - pred_mean) / standard_error
    critical_value = stats.norm.ppf(1 - alpha)
    if Z >= critical_value:
        print("拒绝原假设，接受备择假设")
    else:
        print("无法拒绝原假设")


def get_inconsistency_subset(predictions, labels, senti_text=None, threshold=0.5):
    """
    Phase 2: 构建不一致子集
    筛选"文本单模态情感与真实标签差异大"的样本
    这代表“仅看文本会判断错误，需要跨模态信息才能纠正”的样本
    
    Args:
        predictions: 模型预测 [N,] (仅用于计算子集上的指标)
        labels: 真实标签 [N,]
        senti_text: 文本单模态情感分数 [N,]
                    如果提供，用 |senti_text - label| 定义不一致
                    如果未提供，回退到用标签符号与幅度的启发式方法
        threshold: 不一致阈值
    Returns:
        inconsistent_indices: 不一致样本的索引
        consistent_indices: 一致样本的索引
    """
    if senti_text is not None:
        # 用文本单模态情感与真实标签的差异来定义不一致
        differences = torch.abs(senti_text - labels)
    else:
        # Fallback: 用标签绝对值较小但非零的样本作为“模糊”样本
        # 这些样本更可能存在跨模态不一致
        differences = torch.abs(labels)
    
    inconsistent_mask = differences > threshold
    consistent_mask = differences <= threshold
    
    inconsistent_indices = torch.nonzero(inconsistent_mask).squeeze(-1)
    consistent_indices = torch.nonzero(consistent_mask).squeeze(-1)
    
    # 确保返回至少1维tensor
    if inconsistent_indices.dim() == 0:
        inconsistent_indices = inconsistent_indices.unsqueeze(0)
    if consistent_indices.dim() == 0:
        consistent_indices = consistent_indices.unsqueeze(0)
    
    return inconsistent_indices, consistent_indices


def get_crossmodal_inconsistency_subset(senti_text, senti_audio, senti_vision, threshold=0.3):
    """
    基于跨模态情感差异定义不一致子集
    当文本与音频/视频的情感分数差异超过阈值时，认为存在跨模态不一致
    
    Args:
        senti_text: [N,] 文本情感分数
        senti_audio: [N,] 音频情感分数
        senti_vision: [N,] 视频情感分数
        threshold: 跨模态差异阈值
    Returns:
        inconsistent_indices, consistent_indices
    """
    diff_ta = torch.abs(senti_text - senti_audio)
    diff_tv = torch.abs(senti_text - senti_vision)
    max_diff = torch.max(diff_ta, diff_tv)  # [N,]
    
    inconsistent_mask = max_diff > threshold
    consistent_mask = ~inconsistent_mask
    
    inconsistent_indices = torch.nonzero(inconsistent_mask).squeeze(-1)
    consistent_indices = torch.nonzero(consistent_mask).squeeze(-1)
    
    if inconsistent_indices.dim() == 0:
        inconsistent_indices = inconsistent_indices.unsqueeze(0)
    if consistent_indices.dim() == 0:
        consistent_indices = consistent_indices.unsqueeze(0)
    
    return inconsistent_indices, consistent_indices


def compute_metrics_by_subset(predictions, labels, indices):
    """
    Phase 2: 计算子集上的指标
    
    Args:
        predictions: [N,]
        labels: [N,]
        indices: 子集索引
    Returns:
        metrics: dict with MAE, Corr
    """
    if len(indices) == 0:
        return {'MAE': float('nan'), 'Corr': float('nan'), 'Count': 0}
    
    pred_subset = predictions[indices]
    label_subset = labels[indices]
    
    mae = torch.mean(torch.abs(pred_subset - label_subset)).item()
    
    # Pearson Correlation
    pred_mean = torch.mean(pred_subset)
    label_mean = torch.mean(label_subset)
    numerator = torch.sum((pred_subset - pred_mean) * (label_subset - label_mean))
    denominator = torch.sqrt(
        torch.sum((pred_subset - pred_mean) ** 2) * 
        torch.sum((label_subset - label_mean) ** 2)
    )
    corr = (numerator / (denominator + 1e-8)).item()
    
    return {'MAE': mae, 'Corr': corr, 'Count': len(indices)}
