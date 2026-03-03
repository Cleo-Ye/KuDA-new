"""
保留率扫描脚本: 生成 pruning curve 主图（IEC 效率-性能 Pareto 曲线）

在一个已训练好的 IEC+ICR 模型上，通过 override vision_keep_ratio，
评估不同压缩率下的性能（MAE/Corr/Acc-2/F1）与计算量（平均 vision token 数），
生成 retention_ratio_vs_performance.png 作为论文核心分析图之一。

用法:
    python run_pruning_sweep.py --checkpoint_path ./checkpoints/full_IEC+ICR_full/best.pth
    python run_pruning_sweep.py --checkpoint_path ./checkpoints/full_IEC+ICR_full/best.pth \\
        --ratios 0.2,0.3,0.5,0.7,1.0 --save_dir ./results/pruning_sweep
"""
import os
import sys
import copy
import argparse
import torch
import numpy as np
from opts import parse_opts
from core.dataset import MMDataLoader
from core.metric import MetricsTop
from models.OverallModal import build_model
from visualize_results import plot_retention_ratio_curves


def evaluate_at_ratio(model, test_loader, device, metrics, keep_ratio):
    """
    在给定保留率下评估模型，返回性能指标 + 平均 vision token 数.
    通过临时 override vision_pruner.vision_keep_ratio 实现，不重新训练.
    """
    # 临时设置 keep_ratio
    pruner = getattr(model, 'vision_pruner', None)
    old_ratio = None
    if pruner is not None and hasattr(pruner, 'vision_keep_ratio'):
        old_ratio = pruner.vision_keep_ratio
        pruner.vision_keep_ratio = keep_ratio

    model.eval()
    y_pred, y_true = [], []
    vision_token_counts = []

    with torch.no_grad():
        for data in test_loader:
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
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            # 记录压缩后的 vision token 数
            if hasattr(model, 'last_pruning_info'):
                vision_token_counts.append(model.last_pruning_info['pruned_length'])
            else:
                vision_token_counts.append(inputs['V'].shape[1])

    # 恢复原 keep_ratio
    if pruner is not None and old_ratio is not None:
        pruner.vision_keep_ratio = old_ratio

    pred = torch.cat(y_pred)
    true = torch.cat(y_true)
    results = metrics(pred, true)
    results['vision_tokens_mean'] = float(np.mean(vision_token_counts))
    results['ratio'] = keep_ratio
    return results


def main():
    # ---- 参数解析 ----
    sweep_parser = argparse.ArgumentParser(add_help=False)
    sweep_parser.add_argument('--checkpoint_path', type=str, default='',
                              help='Path to trained IEC+ICR checkpoint')
    sweep_parser.add_argument('--ratios', type=str,
                              default='0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0',
                              help='Comma-separated list of vision_keep_ratio values')
    sweep_parser.add_argument('--save_dir', type=str,
                              default='./results/pruning_sweep',
                              help='Directory to save pruning curve plots')
    sweep_args, remaining = sweep_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    opt = parse_opts()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 加载模型 ----
    ckpt_file = sweep_args.checkpoint_path
    if not ckpt_file:
        # 自动寻找默认路径
        ckpt_file = os.path.join('./checkpoints', f'full_IEC+ICR_full', 'best.pth')
        if not os.path.exists(ckpt_file):
            ckpt_file = os.path.join('./checkpoints', opt.datasetName.upper(), 'best.pth')
    if not os.path.exists(ckpt_file):
        print(f"Error: checkpoint not found at {ckpt_file}")
        print("Please specify --checkpoint_path")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_file}")
    ckpt = torch.load(ckpt_file, weights_only=False)

    # 从 checkpoint 恢复关键 opt
    if 'opt' in ckpt:
        for key in ['use_conflict_js', 'use_vision_pruning', 'iec_mode',
                    'vision_keep_ratio', 'use_routing', 'lambda_con', 'lambda_cal',
                    'rel_min', 'conf_ratio', 'con_ratio']:
            if key in ckpt['opt']:
                setattr(opt, key, ckpt['opt'][key])

    # 确保使用 IEC（即使 checkpoint 没有记录）
    opt.use_vision_pruning = True
    opt.iec_mode = getattr(opt, 'iec_mode', 'text_guided')

    model = build_model(opt).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in ckpt['model_state_dict'].items()
                if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    print(f"  Loaded {len(filtered)}/{len(model_state)} params")

    # ---- 数据 & 评估指标 ----
    dataLoader = MMDataLoader(opt)
    metrics = MetricsTop().getMetics(opt.datasetName)

    # ---- 扫描各保留率 ----
    keep_ratios = [float(r.strip()) for r in sweep_args.ratios.split(',')]
    keep_ratios = sorted(set(keep_ratios))

    print(f"\nSweeping vision_keep_ratio: {keep_ratios}")
    print(f"{'Ratio':>8} {'MAE':>8} {'Corr':>8} {'Acc-2':>8} {'F1':>8} {'Tokens':>8}")
    print('-' * 52)

    ratio_results = []
    for r in keep_ratios:
        res = evaluate_at_ratio(model, dataLoader['test'], device, metrics, r)
        ratio_results.append(res)
        print(f"{r:>8.2f} {res.get('MAE', 0):>8.4f} {res.get('Corr', 0):>8.4f} "
              f"{res.get('Mult_acc_2', 0):>8.4f} {res.get('F1_score', 0):>8.4f} "
              f"{res['vision_tokens_mean']:>8.1f}")

    # ---- 生成曲线图 ----
    os.makedirs(sweep_args.save_dir, exist_ok=True)
    print(f"\nGenerating pruning curve plots -> {sweep_args.save_dir}/")
    plot_retention_ratio_curves(ratio_results, sweep_args.save_dir)

    # 打印最优 sweet-spot
    if ratio_results:
        best_mae = min(ratio_results, key=lambda x: x.get('MAE', float('inf')))
        print(f"\nBest MAE at r={best_mae['ratio']:.2f}: "
              f"MAE={best_mae.get('MAE', 0):.4f}, "
              f"Tokens={best_mae['vision_tokens_mean']:.1f}")

    print("Done.")


if __name__ == '__main__':
    main()
