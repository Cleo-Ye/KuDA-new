"""
重构架构 (PID + 双分支) 专用可视化：协同度 S 与解耦特征 F_cons / F_conf。
生成：S 分布直方图、F_cons/F_conf 的 t-SNE、S vs 预测误差、按 S 分桶的 MAE/Corr。
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def collect_synergy_and_features(model, dataloader, device, max_samples=None):
    """对 test 集做一次 forward，收集 S, pred, label, F_cons, F_conf。"""
    model.eval()
    S_list, pred_list, label_list = [], [], []
    F_cons_list, F_conf_list = [], []

    with torch.no_grad():
        n = 0
        for data in dataloader:
            if max_samples is not None and n >= max_samples:
                break
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
            label = data['labels']['M'].to(device)
            out = model(inputs, None)
            # 兼容新旧接口：
            # - 旧版本: model 返回 (pred, ..., F_cons, F_conf, S, ...)
            # - 新版本: model 返回 dict，键包含 'pred', 'F_R', 'F_S', 'alpha_s' 等
            if isinstance(out, dict):
                pred = out['pred']
                # 约定: F_S 视为 Synergy / consistent 分支, F_R 视为 redundant / conflict 分支
                F_cons = out.get('F_S', out.get('F_R'))
                F_conf = out.get('F_R', out.get('F_S'))
                alpha_s = out.get('alpha_s')
                if alpha_s is None:
                    raise ValueError("Model output dict missing 'alpha_s' for Synergy visualization.")
                # alpha_s: [B, L] 或 [B, 1] -> 统一聚合为 [B]
                if alpha_s.dim() > 1:
                    S = alpha_s.mean(dim=1)
                else:
                    S = alpha_s.view(-1)
            else:
                # 向后兼容旧 tuple 接口
                pred, _, _, F_cons, F_conf, S, _ = out

            b = pred.size(0)
            S_list.append(S.cpu().numpy())
            pred_list.append(pred.cpu().numpy().squeeze())
            label_list.append(label.cpu().numpy().squeeze())
            F_cons_list.append(F_cons.cpu().numpy())
            F_conf_list.append(F_conf.cpu().numpy())
            n += b

    S_arr = np.concatenate(S_list, axis=0)
    pred_arr = np.concatenate(pred_list, axis=0)
    label_arr = np.concatenate(label_list, axis=0)
    F_cons_arr = np.concatenate(F_cons_list, axis=0)
    F_conf_arr = np.concatenate(F_conf_list, axis=0)
    if pred_arr.ndim == 0:
        pred_arr = np.atleast_1d(pred_arr)
    if label_arr.ndim == 0:
        label_arr = np.atleast_1d(label_arr)

    return {
        'S': S_arr,
        'pred': pred_arr,
        'label': label_arr,
        'F_cons': F_cons_arr,
        'F_conf': F_conf_arr,
    }


def _fig_path(save_dir, base_name, model_name):
    """输出路径：若提供 model_name 则文件名为 {model_name}_{base_name}。"""
    if model_name:
        base_name = f"{model_name}_{base_name}"
    return os.path.join(save_dir, base_name)


def plot_s_distribution(S, save_dir, model_name=''):
    """协同度 S 的分布直方图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(S, bins=50, color='#4A90D9', alpha=0.7, edgecolor='white', density=True)
    ax.axvline(S.mean(), color='#D94A4A', linestyle='--', linewidth=2, label=f'Mean = {S.mean():.3f}')
    ax.set_xlabel('Synergy S', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Synergy S (Test Set)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = _fig_path(save_dir, 'synergy_distribution.png', model_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_tsne_decouple(F_cons, F_conf, save_dir, perplexity=30, max_points=1000, model_name=''):
    """F_cons 与 F_conf 合并做 t-SNE，用颜色区分两类特征，证明解耦。"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  Skip t-SNE (sklearn not installed).")
        return

    n = F_cons.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        F_cons = F_cons[idx]
        F_conf = F_conf[idx]
        n = max_points

    # 合并：前 n 个为 F_cons，后 n 个为 F_conf；标签 0=cons, 1=conf
    X = np.vstack([F_cons, F_conf])
    labels = np.array([0] * n + [1] * n)

    # sklearn >= 1.0 使用 max_iter，旧版为 n_iter
    try:
        tsne = TSNE(n_components=2, perplexity=min(perplexity, n - 1), random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=min(perplexity, n - 1), random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_2d[labels == 0, 0], X_2d[labels == 0, 1], c='#4A90D9', alpha=0.6, s=20, label='F_cons')
    ax.scatter(X_2d[labels == 1, 0], X_2d[labels == 1, 1], c='#D94A4A', alpha=0.6, s=20, label='F_conf')
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE of Decoupled Features (F_cons vs F_conf)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = _fig_path(save_dir, 'tsne_decouple.png', model_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_s_vs_error(S, pred, label, save_dir, model_name=''):
    """S vs 预测绝对误差散点图。"""
    err = np.abs(pred - label)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(S, err, alpha=0.4, s=15, c='#2ECC71')
    ax.set_xlabel('Synergy S', fontsize=12)
    ax.set_ylabel('|Pred - Label|', fontsize=12)
    ax.set_title('Synergy S vs. Prediction Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    path = _fig_path(save_dir, 'synergy_vs_error.png', model_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_binned_performance(S, pred, label, save_dir, n_bins=4, model_name=''):
    """按 S 分桶，每桶计算 MAE 和 Corr，条形图。"""
    bins = np.percentile(S, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 1e-6
    bin_idx = np.digitize(S, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mae_per_bin = []
    corr_per_bin = []
    bin_labels = []
    for k in range(n_bins):
        mask = bin_idx == k
        if mask.sum() < 5:
            mae_per_bin.append(np.nan)
            corr_per_bin.append(np.nan)
            bin_labels.append(f'Q{k+1}\n(n={mask.sum()})')
            continue
        p, l = pred[mask], label[mask]
        mae_per_bin.append(np.abs(p - l).mean())
        if p.std() > 1e-8 and l.std() > 1e-8:
            corr_per_bin.append(np.corrcoef(p, l)[0, 1])
        else:
            corr_per_bin.append(np.nan)
        bin_labels.append(f'Q{k+1}\n(n={mask.sum()})')

    x = np.arange(n_bins)
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x - width / 2, mae_per_bin, width, label='MAE', color='#4A90D9', alpha=0.8)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_xlabel('Synergy S bucket (low → high)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, corr_per_bin, width, label='Corr', color='#D94A4A', alpha=0.8)
    ax2.set_ylabel('Correlation', fontsize=12)
    ax2.legend(loc='upper right')

    ax1.set_title('MAE & Corr by Synergy S Bucket (Test Set)', fontsize=14)
    fig.tight_layout()
    path = _fig_path(save_dir, 'binned_mae_corr_by_s.png', model_name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize synergy S and decoupled features (PID refactor)')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Path to checkpoint (e.g. ./checkpoints/sweep_f1/f1_baseline/best.pth)')
    parser.add_argument('--model_name', type=str, default='',
                        help='Model name for output filenames (e.g. f1_baseline, r2_c1c5). If empty, derived from checkpoint path.')
    parser.add_argument('--save_dir', type=str, default='./results/visualizations_synergy',
                        help='Directory to save figures')
    parser.add_argument('--max_tsne', type=int, default=1000, help='Max points for t-SNE (subsample if larger)')
    args, _ = parser.parse_known_args()

    # 从 sys.argv 移除本脚本参数，避免 parse_opts 报错
    filtered = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] in ['--checkpoint_path', '--model_name', '--save_dir', '--max_tsne']:
            i += 2
            continue
        filtered.append(sys.argv[i])
        i += 1
    sys.argv = filtered

    from opts import parse_opts
    from core.dataset import MMDataLoader
    from models.OverallModal import build_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 若未指定 checkpoint，用默认路径并依赖 parse_opts
    if args.checkpoint_path:
        ckpt_path = os.path.abspath(args.checkpoint_path)
    else:
        opt_default = parse_opts()
        ckpt_path = os.path.join('./checkpoints', opt_default.datasetName.upper(), 'best.pth')

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 模型名：用于输出文件名；未指定时从 checkpoint 路径推断（如 .../f1_baseline/best.pth -> f1_baseline）
    model_name = (args.model_name or '').strip()
    if not model_name and ckpt_path:
        parent = os.path.basename(os.path.dirname(ckpt_path))
        if parent and parent != 'checkpoints':
            model_name = parent
    # 文件名中只保留安全字符
    if model_name:
        model_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in model_name)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'opt' in ckpt:
        opt = argparse.Namespace(**ckpt['opt']) if isinstance(ckpt['opt'], dict) else ckpt['opt']
    else:
        opt = parse_opts()
    model = build_model(opt).to(device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    dataLoader = MMDataLoader(opt)
    test_loader = dataLoader['test']

    print("Collecting S, pred, label, F_cons, F_conf on test set...")
    stats = collect_synergy_and_features(model, test_loader, device)

    S = stats['S']
    pred = stats['pred']
    label = stats['label']
    F_cons = stats['F_cons']
    F_conf = stats['F_conf']

    print(f"  N = {len(S)}, S mean = {S.mean():.4f}, S std = {S.std():.4f}")

    print("\n1. Synergy S distribution...")
    plot_s_distribution(S, save_dir, model_name)

    print("\n2. t-SNE of F_cons vs F_conf...")
    plot_tsne_decouple(F_cons, F_conf, save_dir, max_points=args.max_tsne, model_name=model_name)

    print("\n3. S vs prediction error...")
    plot_s_vs_error(S, pred, label, save_dir, model_name)

    print("\n4. Binned MAE & Corr by S...")
    plot_binned_performance(S, pred, label, save_dir, n_bins=4, model_name=model_name)

    print(f"\nAll synergy/decouple visualizations saved to: {save_dir}" + (f" (prefix: {model_name})" if model_name else ""))


if __name__ == '__main__':
    main()
