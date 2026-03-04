#!/usr/bin/env python3
"""
按 D_TV（文本-视觉真实分歧）分桶，比较「当前模型」与 baseline 的 MAE/Acc-2，
证明冲突感知在高分歧样本上增益更大。用于论文分组分析。

用法:
  cd /path/to/KuDA
  python scripts/analyze_gain_by_disagreement.py \
    --checkpoint_ours ./checkpoints/full_IEC+ICR_full/best.pth \
    --checkpoint_baseline ./checkpoints/full_baseline/best.pth \
    --datasetName sims --use_cmvn True \
    [--save_dir ./results/gain_by_disagreement] [--n_bins 5]
"""
import os
import sys
import argparse
import numpy as np
import torch

# 项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from opts import parse_opts
from core.dataset import MMDataLoader
from core.metric import MetricsTop
from models.OverallModal import build_model


def _load_model_and_opt(ckpt_path, base_opt):
    """从 checkpoint 加载模型，用 ckpt['opt'] 恢复配置后 build_model。"""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    opt = argparse.Namespace(**vars(base_opt))
    if 'opt' in ckpt:
        for k, v in ckpt['opt'].items():
            setattr(opt, k, v)
    model = build_model(opt)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model, opt


def _compute_mae_acc2(y_pred, y_true):
    """y_pred, y_true: numpy [N]. Return MAE, Acc-2 (SIMS convention)."""
    y_pred = np.clip(y_pred, -1.0, 1.0)
    y_true = np.clip(y_true, -1.0, 1.0)
    mae = np.mean(np.abs(y_pred - y_true))
    # Acc-2: two classes [-1,0], (0,1]
    ms_2 = [-1.01, 0.0, 1.01]
    p2 = np.zeros_like(y_pred)
    t2 = np.zeros_like(y_true)
    for i in range(2):
        p2[np.logical_and(y_pred > ms_2[i], y_pred <= ms_2[i + 1])] = i
        t2[np.logical_and(y_true > ms_2[i], y_true <= ms_2[i + 1])] = i
    acc2 = np.mean(p2 == t2)
    return float(mae), float(acc2)


def run():
    # 从 argv 中移除本脚本专用参数，再调用 parse_opts，否则 opts 会报 unknown args
    script_args = ['--checkpoint_ours', '--checkpoint_baseline', '--save_dir', '--n_bins']
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] in script_args:
            i += 2  # skip value
            continue
        filtered_argv.append(sys.argv[i])
        i += 1
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + filtered_argv
    base_opt = parse_opts()
    sys.argv = old_argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_ours', type=str, required=True,
                        help='Path to our model (e.g. IEC+ICR full)')
    parser.add_argument('--checkpoint_baseline', type=str, required=True,
                        help='Path to baseline checkpoint')
    parser.add_argument('--save_dir', type=str, default='./results/gain_by_disagreement',
                        help='Where to save table and plot')
    parser.add_argument('--n_bins', type=int, default=5, help='Number of D_TV bins')
    args, _ = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据（与 base_opt 一致）
    dataLoader = MMDataLoader(base_opt)
    test_loader = dataLoader['test']

    # 收集每样本的 label_M, D_TV（以及可选 D_TA, D_AV）
    labels_M = []
    D_TV_list = []
    has_modal = True
    for data in test_loader:
        labels_M.append(data['labels']['M'].numpy().flatten())
        if 'T' in data['labels'] and 'V' in data['labels']:
            y_t = data['labels']['T'].numpy().flatten()
            y_v = data['labels']['V'].numpy().flatten()
            D_TV_list.append(np.abs(y_t - y_v))
        else:
            has_modal = False
            break
    if not has_modal or len(D_TV_list) == 0:
        print("Error: Dataset does not provide per-modality labels (T/V). Need SIMS.")
        return
    labels_M = np.concatenate(labels_M)
    D_TV = np.concatenate(D_TV_list)
    n_samples = len(labels_M)

    # 分桶
    n_bins = args.n_bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(D_TV, percentiles)
    bin_edges[-1] = bin_edges[-1] + 1e-6
    bin_idx = np.digitize(D_TV, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # 加载两个模型
    model_ours, _ = _load_model_and_opt(args.checkpoint_ours, base_opt)
    model_ours = model_ours.to(device)
    model_ours.eval()
    model_base, _ = _load_model_and_opt(args.checkpoint_baseline, base_opt)
    model_base = model_base.to(device)
    model_base.eval()

    pred_ours = []
    pred_baseline = []
    with torch.no_grad():
        for data in test_loader:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1] + 1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1] + 1].to(device),
                    'T': []
                }
            }
            out_ours, *_ = model_ours(inputs, None)
            out_base, *_ = model_base(inputs, None)
            pred_ours.append(out_ours.cpu().numpy().flatten())
            pred_baseline.append(out_base.cpu().numpy().flatten())
    pred_ours = np.concatenate(pred_ours)
    pred_baseline = np.concatenate(pred_baseline)
    if len(pred_ours) != n_samples or len(pred_baseline) != n_samples:
        print("Warning: prediction length != label length; trimming to min.")
        n_samples = min(len(pred_ours), len(pred_baseline), len(labels_M), len(D_TV))
        pred_ours = pred_ours[:n_samples]
        pred_baseline = pred_baseline[:n_samples]
        labels_M = labels_M[:n_samples]
        D_TV = D_TV[:n_samples]
        bin_idx = bin_idx[:n_samples]

    # 按桶统计
    results = []
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() == 0:
            results.append({
                'bin': b, 'n': 0, 'D_TV_lo': bin_edges[b], 'D_TV_hi': bin_edges[b + 1],
                'MAE_ours': np.nan, 'MAE_base': np.nan, 'delta_MAE': np.nan,
                'Acc2_ours': np.nan, 'Acc2_base': np.nan, 'delta_Acc2': np.nan,
            })
            continue
        pred_o = pred_ours[mask]
        pred_b = pred_baseline[mask]
        lab = labels_M[mask]
        mae_o, acc2_o = _compute_mae_acc2(pred_o, lab)
        mae_b, acc2_b = _compute_mae_acc2(pred_b, lab)
        results.append({
            'bin': b, 'n': int(mask.sum()),
            'D_TV_lo': bin_edges[b], 'D_TV_hi': bin_edges[b + 1],
            'D_TV_mean': float(D_TV[mask].mean()),
            'MAE_ours': mae_o, 'MAE_base': mae_b, 'delta_MAE': mae_b - mae_o,
            'Acc2_ours': acc2_o, 'Acc2_base': acc2_b, 'delta_Acc2': acc2_o - acc2_b,
        })

    # 打印并保存表格
    sep = "-" * 90
    header = f"{'Bin':<4} {'n':>6} {'D_TV_mean':>10} {'MAE_ours':>10} {'MAE_base':>10} {'ΔMAE':>8} {'Acc2_ours':>10} {'Acc2_base':>10} {'ΔAcc2':>8}"
    lines = [
        "Gain by D_TV (Text-Vision label disagreement)",
        "Ours: " + args.checkpoint_ours,
        "Baseline: " + args.checkpoint_baseline,
        sep,
        header,
        sep,
    ]
    for r in results:
        if r['n'] == 0:
            lines.append(f"{r['bin']:<4} {r['n']:>6} {'-':>10} {'-':>10} {'-':>10} {'-':>8} {'-':>10} {'-':>10} {'-':>8}")
        else:
            lines.append(
                f"{r['bin']:<4} {r['n']:>6} {r['D_TV_mean']:>10.4f} {r['MAE_ours']:>10.4f} {r['MAE_base']:>10.4f} "
                f"{r['delta_MAE']:>+8.4f} {r['Acc2_ours']:>10.4f} {r['Acc2_base']:>10.4f} {r['delta_Acc2']:>+8.4f}"
            )
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    out_txt = os.path.join(args.save_dir, 'gain_by_D_TV_bins.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Table saved to {out_txt}")

    # 简单柱状图：ΔMAE 和 ΔAcc2 随 bin
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        bins_ok = [r['bin'] for r in results if r['n'] > 0]
        delta_mae = [r['delta_MAE'] for r in results if r['n'] > 0]
        delta_acc2 = [r['delta_Acc2'] for r in results if r['n'] > 0]
        x = np.arange(len(bins_ok))
        axes[0].bar(x, delta_mae, color='#4A90D9', alpha=0.8)
        axes[0].axhline(0, color='gray', linestyle='--')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"Bin{b}\n(n={results[b]['n']})" for b in bins_ok])
        axes[0].set_ylabel(r'$\Delta$ MAE (baseline - ours)')
        axes[0].set_title('MAE gain by D_TV bin (positive = ours better)')
        axes[1].bar(x, delta_acc2, color='#4A90D9', alpha=0.8)
        axes[1].axhline(0, color='gray', linestyle='--')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"Bin{b}\n(n={results[b]['n']})" for b in bins_ok])
        axes[1].set_ylabel(r'$\Delta$ Acc-2 (ours - baseline)')
        axes[1].set_title('Acc-2 gain by D_TV bin (positive = ours better)')
        plt.tight_layout()
        out_png = os.path.join(args.save_dir, 'gain_by_D_TV_bins.png')
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == '__main__':
    run()
