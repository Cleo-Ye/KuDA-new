#!/usr/bin/env python3
"""
可视化 R（冗余）与 S（协同）的 batch 级 vs sample 级：
- r_global/s_global：batch 级先验（同一 batch 内所有样本相同）
- alpha_r/alpha_s：sample 级路由权重（每个样本不同）

用法（从项目根目录）:
  python scripts/visualize_alpha_rs.py --checkpoint_dir ./checkpoints/pid_prior_full
  python scripts/visualize_alpha_rs.py --checkpoint_dir ./checkpoints/pid_prior_full --max_batches 30 --split train
"""
import os
import sys
import argparse

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# 必须在 import torch 之前设置 GPU
def _set_gpu():
    for i, arg in enumerate(sys.argv):
        if arg in ('--gpu', '-g') and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            break
        if arg.startswith('--gpu='):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split('=', 1)[1]
            break
_set_gpu()

import numpy as np
import torch
from experiment_configs import DATASET_CONFIGS
from core.dataset import MMDataLoader
from models.OverallModal import build_model


def apply_dataset_config(opt):
    key = str(getattr(opt, "datasetName", "")).lower()
    cfg = DATASET_CONFIGS.get(key)
    if cfg is None:
        return opt
    if "dataPath" in cfg:
        opt.dataPath = cfg["dataPath"]
    if "seq_lens" in cfg:
        opt.seq_lens = list(cfg["seq_lens"])
    if "fea_dims" in cfg:
        opt.fea_dims = list(cfg["fea_dims"])
    return opt


def get_dims_from_pkl(opt):
    data_path = getattr(opt, "dataPath", None)
    if not data_path or not os.path.isfile(data_path):
        return opt
    try:
        import pickle
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        d = data.get("train", data)
        v, a = d.get("vision"), d.get("audio")
        if v is not None and a is not None:
            opt.fea_dims[1] = int(v.shape[-1])
            opt.fea_dims[2] = int(a.shape[-1])
    except Exception:
        pass
    return opt


def main():
    parser = argparse.ArgumentParser(description='Visualize R/S: batch-level prior vs sample-level alpha')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='e.g. ./checkpoints/pid_prior_full')
    parser.add_argument('--checkpoint_path', type=str, default='', help='default: checkpoint_dir/best.pth')
    parser.add_argument('--max_batches', type=int, default=20, help='number of batches to collect')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--out', type=str, default='', help='output image path; default: checkpoint_dir/alpha_rs_visualization.png')
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    ckpt_path = args.checkpoint_path or os.path.join(ckpt_dir, 'best.pth')
    if not os.path.isfile(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        return 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'opt' not in ckpt:
        print('Checkpoint has no "opt"; need a training checkpoint.')
        return 1
    opt = argparse.Namespace(**ckpt['opt'])
    opt = apply_dataset_config(opt)
    opt = get_dims_from_pkl(opt)

    from core.utils import setup_seed
    setup_seed(getattr(opt, 'seed', 1111))
    model = build_model(opt).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_loader = MMDataLoader(opt)
    loader = data_loader[args.split]
    if loader is None:
        print(f'Split "{args.split}" not found.')
        return 1

    # 收集多个 batch 的 alpha_r, alpha_s, r_global, s_global
    list_alpha_r, list_alpha_s = [], []
    list_r_global, list_s_global = [], []
    first_batch_size = None

    with torch.no_grad():
        for bi, data in enumerate(loader):
            if bi >= args.max_batches:
                break
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1] + 1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1] + 1].to(device),
                    'T': [],
                },
            }
            label = data['labels']['M'].to(device).view(-1, 1)
            out = model(inputs, label.clone())
            if not isinstance(out, dict) or 'alpha_r' not in out:
                print('Model is not Route B (pid_dualpath) or checkpoint mismatch.')
                return 1
            ar = out['alpha_r'].cpu().numpy().flatten()
            as_ = out['alpha_s'].cpu().numpy().flatten()
            rg = out['r_global'].cpu().numpy().flatten()
            sg = out['s_global'].cpu().numpy().flatten()
            list_alpha_r.append(ar)
            list_alpha_s.append(as_)
            list_r_global.append(rg)
            list_s_global.append(sg)
            if first_batch_size is None:
                first_batch_size = len(ar)

    alpha_r_all = np.concatenate(list_alpha_r)
    alpha_s_all = np.concatenate(list_alpha_s)
    r_global_all = np.concatenate(list_r_global)
    s_global_all = np.concatenate(list_s_global)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping plots. Install with: pip install matplotlib')
        print('Collected stats:')
        print(f'  alpha_r: mean={alpha_r_all.mean():.4f}, std={alpha_r_all.std():.4f}, min={alpha_r_all.min():.4f}, max={alpha_r_all.max():.4f}')
        print(f'  alpha_s: mean={alpha_s_all.mean():.4f}, std={alpha_s_all.std():.4f}, min={alpha_s_all.min():.4f}, max={alpha_s_all.max():.4f}')
        return 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # 1) 第一个 batch：样本索引 vs 值 —— batch 级先验（水平线）vs sample 级 alpha（点）
    ax = axes[0, 0]
    n = first_batch_size
    rg_b0 = list_r_global[0]
    sg_b0 = list_s_global[0]
    ar_b0 = list_alpha_r[0]
    as_b0 = list_alpha_s[0]
    x = np.arange(n)
    ax.axhline(rg_b0[0], color='C0', linestyle='--', label=f'r_global (batch)={rg_b0[0]:.3f}')
    ax.scatter(x, ar_b0, c='C0', s=12, alpha=0.8, label='alpha_r (sample)')
    ax.axhline(sg_b0[0], color='C1', linestyle='--', label=f's_global (batch)={sg_b0[0]:.3f}')
    ax.scatter(x, as_b0, c='C1', s=12, alpha=0.8, label='alpha_s (sample)')
    ax.set_xlabel('Sample index in batch')
    ax.set_ylabel('Value')
    ax.set_title('One batch: batch-level prior (dashed) vs sample-level alpha (points)')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 2) 所有样本的 alpha_r / alpha_s 直方图
    ax = axes[0, 1]
    ax.hist(alpha_r_all, bins=30, alpha=0.6, label='alpha_r', color='C0', density=True)
    ax.hist(alpha_s_all, bins=30, alpha=0.6, label='alpha_s', color='C1', density=True)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Sample-level distribution (all batches)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) alpha_r vs alpha_s 散点图（展示样本级多样性）
    ax = axes[1, 0]
    ax.scatter(alpha_r_all, alpha_s_all, s=5, alpha=0.4, c='green')
    ax.plot([0.5, 0.5], [0, 1], 'k--', alpha=0.5)
    ax.plot([0, 1], [0.5, 0.5], 'k--', alpha=0.5)
    ax.set_xlabel('alpha_r (Redundancy)')
    ax.set_ylabel('alpha_s (Synergy)')
    ax.set_title('Sample-level: alpha_r vs alpha_s (deviation from 0.5)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 4) 每个 batch 的 alpha_r/alpha_s 均值±标准差（证明 batch 间也有变化）
    ax = axes[1, 1]
    n_batches = len(list_alpha_r)
    batch_means_r = [arr.mean() for arr in list_alpha_r]
    batch_means_s = [arr.mean() for arr in list_alpha_s]
    batch_std_r = [arr.std() for arr in list_alpha_r]
    batch_std_s = [arr.std() for arr in list_alpha_s]
    x = np.arange(n_batches)
    ax.errorbar(x, batch_means_r, yerr=batch_std_r, fmt='o-', capsize=2, label='alpha_r mean±std', color='C0')
    ax.errorbar(x, batch_means_s, yerr=batch_std_s, fmt='s-', capsize=2, label='alpha_s mean±std', color='C1')
    ax.set_xlabel('Batch index')
    ax.set_ylabel('Mean ± Std')
    ax.set_title('Per-batch: sample-level variance (std > 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('R/S: Batch-level prior vs Sample-level routing (pid_dualpath)', fontsize=11)
    plt.tight_layout()
    out_path = args.out or os.path.join(ckpt_dir, 'alpha_rs_visualization.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    print(f'  alpha_r: mean={alpha_r_all.mean():.4f}, std={alpha_r_all.std():.4f}')
    print(f'  alpha_s: mean={alpha_s_all.mean():.4f}, std={alpha_s_all.std():.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
