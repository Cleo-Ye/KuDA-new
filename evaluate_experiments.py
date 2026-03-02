"""
Phase 2-3: 实验评估脚本
包含消融实验、不一致子集评估、可视化生成
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.utils import compute_metrics_by_subset, get_inconsistency_subset


def run_ablation_experiments(model, dataloader, device, opt, logger):
    """
    运行消融实验
    
    测试不同组件组合的效果:
    1. Baseline (去KI后)
    2. +证据拆分
    3. +Evidence-JS
    4. +路由
    5. +Token筛选
    """
    results = {}
    
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        all_conflict_C = []
        all_senti_text = []  # 收集文本单模态情感分数
        
        for data in dataloader:
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
            label = data['labels']['M'].to(device).view(-1, 1)
            
            output, _, _, _, _, _, _ = model(inputs, None)
            
            all_preds.append(output.cpu())
            all_labels.append(label.cpu())
            
            # 收集冲突强度
            if hasattr(model, 'last_conflict_intensity'):
                all_conflict_C.append(model.last_conflict_intensity.cpu())
            
            # 收集文本单模态情感预测(用于定义不一致子集)
            if hasattr(model, 'last_uni_senti_text'):
                all_senti_text.append(model.last_uni_senti_text.cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        # 整体指标
        mae = torch.mean(torch.abs(all_preds - all_labels)).item()
        pred_mean = torch.mean(all_preds)
        label_mean = torch.mean(all_labels)
        numerator = torch.sum((all_preds - pred_mean) * (all_labels - label_mean))
        denominator = torch.sqrt(
            torch.sum((all_preds - pred_mean) ** 2) * 
            torch.sum((all_labels - label_mean) ** 2)
        )
        corr = (numerator / (denominator + 1e-8)).item()
        
        results['overall'] = {'MAE': mae, 'Corr': corr, 'Count': len(all_preds)}
        
        # Priority 4: 用文本单模态情感定义不一致子集
        senti_text = None
        if len(all_senti_text) > 0:
            senti_text = torch.cat(all_senti_text).squeeze()
        
        inconsistent_indices, consistent_indices = get_inconsistency_subset(
            all_preds.squeeze(), 
            all_labels.squeeze(),
            senti_text=senti_text,
            threshold=0.5
        )
        
        results['inconsistent'] = compute_metrics_by_subset(
            all_preds.squeeze(), 
            all_labels.squeeze(), 
            inconsistent_indices
        )
        results['consistent'] = compute_metrics_by_subset(
            all_preds.squeeze(), 
            all_labels.squeeze(), 
            consistent_indices
        )
        
        # 冲突强度统计
        if len(all_conflict_C) > 0:
            all_conflict_C = torch.cat(all_conflict_C)
            results['conflict_stats'] = {
                'mean': all_conflict_C.mean().item(),
                'std': all_conflict_C.std().item(),
                'min': all_conflict_C.min().item(),
                'max': all_conflict_C.max().item()
            }
    
    return results


def visualize_conflict_intensity_distribution(model, dataloader, device, save_path):
    """
    可视化冲突强度分布
    对比一致样本vs不一致样本的冲突强度C
    """
    model.eval()
    conflict_C_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for data in dataloader:
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
            label = data['labels']['M'].to(device)
            
            output, _, _, _, _, _, _ = model(inputs, None)
            
            if hasattr(model, 'last_conflict_intensity'):
                conflict_C_list.append(model.last_conflict_intensity.cpu())
            
            labels_list.append(label.cpu())
            preds_list.append(output.cpu())
    
    if len(conflict_C_list) == 0:
        print("Warning: No conflict intensity recorded")
        return
    
    conflict_C = torch.cat(conflict_C_list).squeeze().numpy()
    labels = torch.cat(labels_list).squeeze().numpy()
    preds = torch.cat(preds_list).squeeze().numpy()
    
    # 区分一致vs不一致样本
    text_label_diff = np.abs(preds - labels)
    inconsistent_mask = text_label_diff > 0.5
    
    C_inconsistent = conflict_C[inconsistent_mask]
    C_consistent = conflict_C[~inconsistent_mask]
    
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(C_consistent, bins=30, alpha=0.6, label=f'Consistent (n={len(C_consistent)})', color='blue')
    plt.hist(C_inconsistent, bins=30, alpha=0.6, label=f'Inconsistent (n={len(C_inconsistent)})', color='red')
    plt.xlabel('Conflict Intensity C')
    plt.ylabel('Frequency')
    plt.title('Conflict Intensity Distribution: Consistent vs Inconsistent Samples')
    plt.legend()
    plt.grid(alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Conflict intensity distribution saved to {save_path}")
    print(f"Consistent samples - C mean: {C_consistent.mean():.4f}, std: {C_consistent.std():.4f}")
    print(f"Inconsistent samples - C mean: {C_inconsistent.mean():.4f}, std: {C_inconsistent.std():.4f}")


def print_ablation_table(results_dict, logger=None):
    """
    打印消融实验表格
    
    Args:
        results_dict: {
            'baseline': {'overall': {...}, 'inconsistent': {...}},
            '+evidence_split': {...},
            '+evidence_js': {...},
            'full': {...},
            '+token_pruning': {...}
        }
    """
    from tabulate import tabulate
    
    table_data = []
    for exp_name, results in results_dict.items():
        row = [
            exp_name,
            results['overall']['MAE'],
            results['overall']['Corr'],
            results.get('inconsistent', {}).get('MAE', '-'),
            results.get('inconsistent', {}).get('Corr', '-'),
            results.get('inconsistent', {}).get('Count', 0)
        ]
        table_data.append(row)
    
    headers = ["Experiment", "Overall MAE↓", "Overall Corr↑", 
               "Inconsist MAE↓", "Inconsist Corr↑", "Inconsist Count"]
    
    table = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
    
    if logger:
        logger.info("\n" + table)
    else:
        print("\n" + table)
    
    return table
