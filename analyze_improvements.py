"""
改进前后对比分析工具
读取checkpoint并对比改进前后的关键指标
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_checkpoint_info(ckpt_path):
    """加载checkpoint的训练信息"""
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, weights_only=False)
    return {
        'epoch': ckpt.get('epoch', 0),
        'valid_mae': ckpt.get('valid_mae', 0),
        'valid_corr': ckpt.get('valid_corr', 0),
        'opt': ckpt.get('opt', {})
    }


def compare_experiments():
    """对比不同实验配置的结果"""
    base_dir = './checkpoints'
    
    experiments = {
        'Baseline': 'baseline',
        '+IEC only': 'iec_only',
        '+ICR only': 'icr_only',
        'IEC+ICR full': 'iec_icr_full'
    }
    
    results = {}
    for name, folder in experiments.items():
        ckpt_path = os.path.join(base_dir, f'{folder}_seed0', 'best.pth')
        info = load_checkpoint_info(ckpt_path)
        if info:
            results[name] = info
            print(f"{name:20s} | MAE: {info['valid_mae']:.4f} | Corr: {info['valid_corr']:.4f}")
        else:
            print(f"{name:20s} | 未找到checkpoint")
    
    return results


def analyze_improvements(results):
    """分析改进效果"""
    if 'Baseline' not in results:
        print("未找到baseline结果，无法对比")
        return
    
    baseline = results['Baseline']
    print("\n" + "="*60)
    print("改进效果分析")
    print("="*60)
    
    for name, result in results.items():
        if name == 'Baseline':
            continue
        
        mae_improve = baseline['valid_mae'] - result['valid_mae']
        corr_improve = result['valid_corr'] - baseline['valid_corr']
        mae_percent = (mae_improve / baseline['valid_mae']) * 100
        corr_percent = (corr_improve / baseline['valid_corr']) * 100
        
        print(f"\n{name}:")
        print(f"  MAE:  {baseline['valid_mae']:.4f} → {result['valid_mae']:.4f} (Δ={mae_improve:+.4f}, {mae_percent:+.2f}%)")
        print(f"  Corr: {baseline['valid_corr']:.4f} → {result['valid_corr']:.4f} (Δ={corr_improve:+.4f}, {corr_percent:+.2f}%)")


def plot_comparison(results, save_path='./figures/improvement_comparison.png'):
    """绘制对比图"""
    if not results:
        print("无结果可绘制")
        return
    
    os.makedirs('./figures', exist_ok=True)
    
    names = list(results.keys())
    maes = [results[n]['valid_mae'] for n in names]
    corrs = [results[n]['valid_corr'] for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MAE对比
    axes[0].bar(names, maes, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[0].set_ylabel('MAE (↓)', fontsize=12)
    axes[0].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, (n, v) in enumerate(zip(names, maes)):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10)
    
    # Corr对比
    axes[1].bar(names, corrs, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[1].set_ylabel('Correlation (↑)', fontsize=12)
    axes[1].set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, (n, v) in enumerate(zip(names, corrs)):
        axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存至: {save_path}")


def check_key_parameters(results):
    """检查关键超参数设置"""
    print("\n" + "="*60)
    print("关键超参数检查")
    print("="*60)
    
    key_params = [
        'gate_k', 'gate_tau', 'conf_ratio', 'con_ratio', 'rel_min',
        'lambda_js', 'lambda_con', 'lambda_cal', 'vision_keep_ratio'
    ]
    
    for name, result in results.items():
        print(f"\n{name}:")
        opt = result.get('opt', {})
        for param in key_params:
            value = opt.get(param, 'N/A')
            print(f"  {param:20s}: {value}")


def main():
    print("KuDA 改进前后对比分析")
    print("="*60)
    
    # 对比实验结果
    results = compare_experiments()
    
    # 分析改进效果
    analyze_improvements(results)
    
    # 检查超参数
    check_key_parameters(results)
    
    # 绘制对比图
    if results:
        plot_comparison(results)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()
