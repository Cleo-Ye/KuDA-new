"""
Phase 2-3: 可视化工具
生成冲突强度分布图、证据拆分统计、Case Study、
C vs 预测误差散点图、Pruning前后对比、C_m分布、证据比例汇总
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ============================================================
# Helper: 统一的forward收集器, 避免重复forward
# ============================================================
def _collect_all_stats(model, dataloader, device):
    """
    对整个dataloader做一次forward, 收集所有可视化需要的统计量.
    Returns:
        stats: dict with keys:
            'C': [N,] conflict intensity
            'C_m': {'T'/'A'/'V': [N,]} per-modality C
            'preds': [N,] predictions
            'labels': [N,] ground truth labels
            'con_counts': {'T'/'A'/'V': [N,]} congruent token counts
            'conf_counts': {'T'/'A'/'V': [N,]} conflict token counts
            'seq_lens': {'T'/'A'/'V': int} sequence lengths per modality
            'vision_orig_len': int, original vision length before pruning
            'vision_pruned_len': int, vision length after pruning
            'samples': list of per-sample dicts (for case study)
    """
    model.eval()
    stats = {
        'C': [], 'C_m': {'T': [], 'A': [], 'V': []},
        'preds': [], 'labels': [],
        'con_counts': {'T': [], 'A': [], 'V': []},
        'conf_counts': {'T': [], 'A': [], 'V': []},
        'seq_lens': {},
        'vision_orig_len': 0, 'vision_pruned_len': 0,
        'samples': [],
    }

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
            pred, _, _, _, _, _, _ = model(inputs, None)

            stats['preds'].append(pred.cpu().squeeze())
            stats['labels'].append(label.cpu().squeeze())

            if hasattr(model, 'last_conflict_intensity'):
                stats['C'].append(model.last_conflict_intensity.cpu())
            if hasattr(model, 'last_conflict_intensity_m'):
                for m in ['T', 'A', 'V']:
                    stats['C_m'][m].append(model.last_conflict_intensity_m[m].cpu())
            if hasattr(model, 'last_con_masks') and hasattr(model, 'last_conf_masks'):
                for m in ['T', 'A', 'V']:
                    stats['con_counts'][m].append(model.last_con_masks[m].sum(dim=1).cpu())
                    stats['conf_counts'][m].append(model.last_conf_masks[m].sum(dim=1).cpu())
                    stats['seq_lens'][m] = model.last_con_masks[m].shape[1]
            if hasattr(model, 'last_vision_original_len'):
                stats['vision_orig_len'] = model.last_vision_original_len
            if hasattr(model, 'last_pruning_info'):
                stats['vision_pruned_len'] = model.last_pruning_info['pruned_length']
            else:
                # 若未启用剪枝模块(或未记录pruning_info), 视为保留原长度
                # 用当前forward的vision长度作为"after"长度，避免出现 55 -> 0 的误导统计
                stats['vision_pruned_len'] = inputs['V'].shape[1]

            # 收集per-sample info for case study
            if hasattr(model, 'last_conflict_intensity'):
                B = inputs['V'].shape[0]
                for b in range(B):
                    stats['samples'].append({
                        'vision': data['vision'][b],
                        'audio': data['audio'][b],
                        'text': data['text'][b],
                        'vision_padding_mask': data['vision_padding_mask'][b, 1:data['vision'].shape[1]+1],
                        'audio_padding_mask': data['audio_padding_mask'][b, 1:data['audio'].shape[1]+1],
                        'labels': {'M': data['labels']['M'][b]},
                        'C': model.last_conflict_intensity[b].item(),
                        'pred': pred[b].item(),
                    })

    # Concatenate
    if stats['C']:
        stats['C'] = torch.cat(stats['C']).numpy()
    else:
        stats['C'] = np.array([])
    stats['preds'] = torch.cat(stats['preds']).numpy()
    stats['labels'] = torch.cat(stats['labels']).numpy()
    for m in ['T', 'A', 'V']:
        if stats['C_m'][m]:
            stats['C_m'][m] = torch.cat(stats['C_m'][m]).numpy()
        else:
            stats['C_m'][m] = np.array([])
        if stats['con_counts'][m]:
            stats['con_counts'][m] = torch.cat(stats['con_counts'][m]).numpy()
            stats['conf_counts'][m] = torch.cat(stats['conf_counts'][m]).numpy()
        else:
            stats['con_counts'][m] = np.array([])
            stats['conf_counts'][m] = np.array([])

    return stats


# ============================================================
# 1. Conflict Intensity Distribution
# ============================================================
def visualize_conflict_distribution(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    C = stats['C']
    if len(C) == 0:
        print("Warning: No conflict intensity recorded. Make sure use_conflict_js=True")
        return

    threshold_C = np.median(C)
    C_low = C[C <= threshold_C]
    C_high = C[C > threshold_C]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    axes[0].hist(C_low, bins=30, alpha=0.6, label=f'Low (n={len(C_low)})', color='#4A90D9')
    axes[0].hist(C_high, bins=30, alpha=0.6, label=f'High (n={len(C_high)})', color='#D94A4A')
    axes[0].set_xlabel('Conflict Intensity C')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Conflict Intensity Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot([C_low, C_high], tick_labels=['Low Conflict', 'High Conflict'])
    axes[1].set_ylabel('Conflict Intensity C')
    axes[1].set_title('Conflict Intensity Box Plot')
    axes[1].grid(axis='y', alpha=0.3)

    # KDE density
    if len(C) > 10:
        sns.kdeplot(C_low, ax=axes[2], label='Low Conflict', color='#4A90D9', fill=True, alpha=0.3)
        sns.kdeplot(C_high, ax=axes[2], label='High Conflict', color='#D94A4A', fill=True, alpha=0.3)
        axes[2].set_xlabel('Conflict Intensity C')
        axes[2].set_ylabel('Density')
        axes[2].set_title('KDE Density')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'conflict_intensity_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nConflict intensity distribution saved to {save_path}")
    print(f"  C range: [{C.min():.4f}, {C.max():.4f}], dynamic range: {C.max()-C.min():.4f}")
    print(f"  Low conflict - mean: {C_low.mean():.4f}, std: {C_low.std():.4f}")
    print(f"  High conflict - mean: {C_high.mean():.4f}, std: {C_high.std():.4f}")
    print(f"  Separation (high_mean - low_mean): {C_high.mean() - C_low.mean():.4f}")


# ============================================================
# 2. Evidence Split Statistics (improved: bar chart with %)
# ============================================================
def visualize_evidence_split_stats(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if len(stats['con_counts']['T']) == 0:
        print("Warning: No evidence masks recorded")
        return

    modalities = ['T', 'A', 'V']
    modality_names = ['Text', 'Audio', 'Vision']

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for idx, (m, name) in enumerate(zip(modalities, modality_names)):
        con = stats['con_counts'][m]
        conf = stats['conf_counts'][m]
        total = stats['seq_lens'].get(m, 1)

        # Top row: scatter plot
        x = np.arange(len(con))
        axes[0, idx].scatter(x, con, alpha=0.4, s=8, label='Congruent', color='#4A90D9')
        axes[0, idx].scatter(x, conf, alpha=0.4, s=8, label='Conflict', color='#D94A4A')
        axes[0, idx].set_xlabel('Sample Index')
        axes[0, idx].set_ylabel('Token Count')
        axes[0, idx].set_title(f'{name} Evidence Counts (L={total})')
        axes[0, idx].legend()
        axes[0, idx].grid(alpha=0.3)

        # Bottom row: ratio bar chart (mean %)
        con_ratio = con.mean() / total * 100
        conf_ratio = conf.mean() / total * 100
        neutral_ratio = max(100 - con_ratio - conf_ratio, 0)

        bars = axes[1, idx].bar(
            ['Congruent', 'Conflict', 'Neutral'],
            [con_ratio, conf_ratio, neutral_ratio],
            color=['#4A90D9', '#D94A4A', '#AAAAAA'], alpha=0.8
        )
        for bar, val in zip(bars, [con_ratio, conf_ratio, neutral_ratio]):
            axes[1, idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{val:.1f}%', ha='center', fontsize=10)
        axes[1, idx].set_ylabel('Percentage (%)')
        axes[1, idx].set_title(f'{name} Evidence Ratio')
        axes[1, idx].set_ylim([0, 100])
        axes[1, idx].grid(axis='y', alpha=0.3)

        print(f"  {name}: Congruent={con.mean():.1f} ({con_ratio:.1f}%), "
              f"Conflict={conf.mean():.1f} ({conf_ratio:.1f}%), "
              f"Neutral={neutral_ratio:.1f}%")

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'evidence_split_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evidence split statistics saved to {save_path}")


# ============================================================
# 3. Case Study (improved)
# ============================================================
def visualize_case_study(model, sample, device, save_dir, case_id=0):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        inputs = {
            'V': sample['vision'].unsqueeze(0).to(device),
            'A': sample['audio'].unsqueeze(0).to(device),
            'T': sample['text'].unsqueeze(0).to(device),
            'mask': {
                'V': sample['vision_padding_mask'].unsqueeze(0).to(device),
                'A': sample['audio_padding_mask'].unsqueeze(0).to(device),
                'T': []
            }
        }
        output, _, _, _, _, _, _ = model(inputs, None)

        if not (hasattr(model, 'last_con_masks') and hasattr(model, 'last_conf_masks')):
            print("Warning: No evidence masks recorded")
            return

        con_masks = model.last_con_masks
        conf_masks = model.last_conf_masks
        C = model.last_conflict_intensity.item()
        C_m = {m: model.last_conflict_intensity_m[m].item() for m in ['T', 'A', 'V']}
        pred = output.item()
        label = sample['labels']['M'].item()

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1, 1, 0.4]})
        modalities = ['T', 'A', 'V']
        modality_names = ['Text', 'Audio', 'Vision']

        for idx, (m, name) in enumerate(zip(modalities, modality_names)):
            con_mask = con_masks[m][0].cpu().numpy()
            conf_mask = conf_masks[m][0].cpu().numpy()
            seq_len = len(con_mask)
            x = np.arange(seq_len)

            axes[idx].bar(x[con_mask], np.ones(con_mask.sum()),
                         alpha=0.7, label='Congruent', color='#4A90D9', width=0.8)
            axes[idx].bar(x[conf_mask], np.ones(conf_mask.sum()),
                         alpha=0.7, label='Conflict', color='#D94A4A', width=0.8)

            n_con = con_mask.sum()
            n_conf = conf_mask.sum()
            axes[idx].set_ylabel(f'{name}\n(C_m={C_m[m]:.3f})', fontsize=10)
            axes[idx].set_ylim([0, 1.5])
            axes[idx].legend(loc='upper right', fontsize=8)
            axes[idx].set_title(
                f'{name}: {n_conf} conflict + {n_con} congruent / {seq_len} total',
                fontsize=10, loc='left'
            )
            axes[idx].grid(axis='x', alpha=0.2)

        # Bottom panel: summary text
        axes[3].axis('off')
        summary = (
            f"Conflict Intensity C = {C:.4f}  |  "
            f"Prediction = {pred:.3f}  |  Label = {label:.3f}  |  "
            f"Error = {abs(pred - label):.3f}\n"
            f"C_T = {C_m['T']:.4f}  |  C_A = {C_m['A']:.4f}  |  C_V = {C_m['V']:.4f}"
        )
        axes[3].text(0.5, 0.5, summary, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(f'Case Study #{case_id} (Conflict Intensity C={C:.4f})', fontsize=14, y=1.01)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'case_study_{case_id}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Case study saved to {save_path}")
        print(f"    Pred={pred:.3f}, Label={label:.3f}, C={C:.4f}")


# ============================================================
# 4. NEW: C vs Prediction Error Scatter
# ============================================================
def visualize_c_vs_error(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    C = stats['C']
    if len(C) == 0:
        print("Warning: No conflict intensity for C-vs-error plot")
        return

    errors = np.abs(stats['preds'] - stats['labels'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter
    sc = axes[0].scatter(C, errors, c=C, cmap='RdYlBu_r', alpha=0.5, s=15, edgecolors='none')
    plt.colorbar(sc, ax=axes[0], label='C')
    axes[0].set_xlabel('Conflict Intensity C')
    axes[0].set_ylabel('|Prediction - Label|')
    axes[0].set_title('Conflict Intensity vs Prediction Error')
    axes[0].grid(alpha=0.3)

    # Correlation
    corr = np.corrcoef(C, errors)[0, 1]
    axes[0].text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=axes[0].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Binned bar chart: split C into 5 bins
    n_bins = 5
    bin_edges = np.linspace(C.min(), C.max() + 1e-8, n_bins + 1)
    bin_labels = [f'{bin_edges[i]:.3f}-\n{bin_edges[i+1]:.3f}' for i in range(n_bins)]
    bin_means = []
    bin_stds = []
    for i in range(n_bins):
        mask = (C >= bin_edges[i]) & (C < bin_edges[i+1])
        if mask.sum() > 0:
            bin_means.append(errors[mask].mean())
            bin_stds.append(errors[mask].std())
        else:
            bin_means.append(0)
            bin_stds.append(0)

    bars = axes[1].bar(range(n_bins), bin_means, yerr=bin_stds,
                       color='#D94A4A', alpha=0.7, capsize=4)
    axes[1].set_xticks(range(n_bins))
    axes[1].set_xticklabels(bin_labels, fontsize=8)
    axes[1].set_xlabel('Conflict Intensity C (binned)')
    axes[1].set_ylabel('Mean |Error|')
    axes[1].set_title('Mean Error by Conflict Intensity Bin')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'c_vs_prediction_error.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nC vs prediction error saved to {save_path}")
    print(f"  Pearson correlation(C, |error|) = {corr:.4f}")


# ============================================================
# 5. NEW: Pruning Before/After Comparison
# ============================================================
def visualize_pruning_comparison(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    orig = stats['vision_orig_len']
    pruned = stats['vision_pruned_len']
    if orig == 0:
        print("Warning: No pruning info recorded")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: before vs after
    categories = ['Original', 'After Pruning']
    values = [orig, pruned]
    colors = ['#AAAAAA', '#4A90D9']
    bars = axes[0].bar(categories, values, color=colors, alpha=0.8, width=0.5)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(val), ha='center', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Number of Vision Tokens')
    axes[0].set_title('Vision Token Pruning')
    axes[0].grid(axis='y', alpha=0.3)

    # Pie chart: retained vs removed
    retained_pct = pruned / orig * 100
    removed_pct = 100 - retained_pct
    axes[1].pie([retained_pct, removed_pct],
               labels=[f'Retained ({pruned})', f'Removed ({orig - pruned})'],
               colors=['#4A90D9', '#D94A4A'],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    axes[1].set_title(f'Compression Ratio: {removed_pct:.1f}% removed')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pruning_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPruning comparison saved to {save_path}")
    print(f"  Vision tokens: {orig} -> {pruned} ({removed_pct:.1f}% removed)")


# ============================================================
# 6. NEW: Per-modality C_m Distribution
# ============================================================
def visualize_cm_distribution(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if len(stats['C_m']['T']) == 0:
        print("Warning: No per-modality C_m recorded")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    modalities = ['T', 'A', 'V']
    modality_names = ['Text', 'Audio', 'Vision']
    colors = ['#2ECC71', '#3498DB', '#9B59B6']

    for idx, (m, name, color) in enumerate(zip(modalities, modality_names, colors)):
        cm = stats['C_m'][m]
        axes[idx].hist(cm, bins=30, alpha=0.7, color=color, edgecolor='white')
        axes[idx].axvline(cm.mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean={cm.mean():.4f}')
        axes[idx].set_xlabel(f'C_{m} ({name})')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{name} Conflict Intensity C_{m}')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        axes[idx].set_xlim([0, 1])

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cm_per_modality_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPer-modality C_m distribution saved to {save_path}")
    for m, name in zip(modalities, modality_names):
        cm = stats['C_m'][m]
        print(f"  C_{m} ({name}): mean={cm.mean():.4f}, std={cm.std():.4f}, "
              f"range=[{cm.min():.4f}, {cm.max():.4f}]")


# ============================================================
# 6b. Conflict strength bucket vs performance (ICR evidence)
# ============================================================
def visualize_conflict_bucket_performance(stats, save_dir):
    """
    Phase 1 必须: 冲突强度分桶 vs MAE/Acc-2, 证明 ICR 有效.
    """
    os.makedirs(save_dir, exist_ok=True)
    C = stats['C']
    if len(C) == 0:
        print("Warning: No conflict intensity for bucket performance plot")
        return

    preds = stats['preds']
    labels = stats['labels']
    errors = np.abs(preds - labels)
    # Acc-2: (pred>=0) == (label>=0)
    acc2 = (np.sign(preds) == np.sign(labels)) | (np.abs(labels) < 1e-6)
    acc2 = acc2.astype(np.float64)

    n_bins = 5
    bin_edges = np.linspace(C.min(), C.max() + 1e-8, n_bins + 1)
    bin_labels = [f'[{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f})' for i in range(n_bins)]
    mae_per_bin = []
    acc2_per_bin = []
    count_per_bin = []
    for i in range(n_bins):
        mask = (C >= bin_edges[i]) & (C < bin_edges[i + 1])
        n = mask.sum()
        count_per_bin.append(n)
        if n > 0:
            mae_per_bin.append(errors[mask].mean())
            acc2_per_bin.append(acc2[mask].mean())
        else:
            mae_per_bin.append(np.nan)
            acc2_per_bin.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_bins)
    width = 0.35
    axes[0].bar(x - width/2, mae_per_bin, width, label='MAE', color='#D94A4A', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bin_labels, fontsize=8, rotation=15)
    axes[0].set_ylabel('MAE')
    axes[0].set_xlabel('Conflict Intensity C (bucket)')
    axes[0].set_title('Mean MAE by Conflict Bucket')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x - width/2, acc2_per_bin, width, label='Acc-2', color='#4A90D9', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bin_labels, fontsize=8, rotation=15)
    axes[1].set_ylabel('Acc-2')
    axes[1].set_xlabel('Conflict Intensity C (bucket)')
    axes[1].set_title('Acc-2 by Conflict Bucket')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'conflict_bucket_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConflict bucket vs performance saved to {save_path}")
    for i in range(n_bins):
        print(f"  Bin {i} (n={count_per_bin[i]}): MAE={mae_per_bin[i]:.4f}, Acc-2={acc2_per_bin[i]:.4f}")


# ============================================================
# 6c. Retention ratio r vs performance and compute (IEC evidence)
# ============================================================
def plot_retention_ratio_curves(ratio_results, save_dir):
    """
    Phase 1 必须: 保留率 r vs 性能(MAE/Corr/Acc-2/F1) 与 计算量(vision_tokens_mean).
    ratio_results: list of dict, e.g. [{'ratio': 0.2, 'MAE': 0.35, 'Corr': 0.72,
       'Mult_acc_2': 0.78, 'F1_score': 0.76, 'vision_tokens_mean': 11.2}, ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    if not ratio_results:
        print("Warning: No ratio_results for retention curve")
        return

    ratios = np.array([r['ratio'] for r in ratio_results])
    sort_idx = np.argsort(ratios)
    ratios = ratios[sort_idx]

    metrics = ['MAE', 'Corr', 'Mult_acc_2', 'F1_score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, m in enumerate(metrics):
        if m in ratio_results[0]:
            vals = np.array([r[m] for r in ratio_results])[sort_idx]
            axes[idx].plot(ratios, vals, 'o-', linewidth=2, markersize=8)
            axes[idx].set_xlabel('Vision retention ratio r')
            axes[idx].set_ylabel(m)
            axes[idx].set_title(f'{m} vs r')
            axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'retention_ratio_vs_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Retention ratio vs performance saved to {save_path}")

    if 'vision_tokens_mean' in ratio_results[0]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        tokens = np.array([r['vision_tokens_mean'] for r in ratio_results])[sort_idx]
        ax2.plot(ratios, tokens, 's-', color='#2ECC71', linewidth=2, markersize=8)
        ax2.set_xlabel('Vision retention ratio r')
        ax2.set_ylabel('Mean vision tokens per sample')
        ax2.set_title('Compute (vision tokens) vs r')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        save_path2 = os.path.join(save_dir, 'retention_ratio_vs_compute.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Retention ratio vs compute saved to {save_path2}")


# ============================================================
# 7. NEW: Evidence Ratio Summary Table (saved as image)
# ============================================================
def visualize_evidence_summary_table(stats, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    if len(stats['con_counts']['T']) == 0:
        print("Warning: No evidence data for summary table")
        return

    modalities = ['T', 'A', 'V']
    modality_names = ['Text', 'Audio', 'Vision']

    rows = []
    for m, name in zip(modalities, modality_names):
        con = stats['con_counts'][m]
        conf = stats['conf_counts'][m]
        total = stats['seq_lens'].get(m, 1)
        rows.append([
            name, total,
            f'{conf.mean():.1f} ({conf.mean()/total*100:.1f}%)',
            f'{con.mean():.1f} ({con.mean()/total*100:.1f}%)',
            f'{max(total - conf.mean() - con.mean(), 0):.1f} '
            f'({max(100 - conf.mean()/total*100 - con.mean()/total*100, 0):.1f}%)',
        ])

    # Add C stats
    C = stats['C']
    if len(C) > 0:
        rows.append(['---', '---', '---', '---', '---'])
        rows.append(['C (overall)', '', f'mean={C.mean():.4f}', f'std={C.std():.4f}',
                     f'range=[{C.min():.4f},{C.max():.4f}]'])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    col_labels = ['Modality', 'Seq Len', 'Conflict Evidence', 'Congruent Evidence', 'Neutral']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                    cellLoc='center', colColours=['#E8E8E8']*5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    plt.title('Evidence Split Summary', fontsize=14, pad=20)
    save_path = os.path.join(save_dir, 'evidence_summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nEvidence summary table saved to {save_path}")


# ============================================================
# Main: Generate All Visualizations
# ============================================================
def generate_all_visualizations(model, test_loader, device, save_dir='./results/visualizations'):
    print("\nGenerating visualizations...")
    print("Collecting statistics from test set (single forward pass)...")

    stats = _collect_all_stats(model, test_loader, device)

    # 1. Conflict intensity distribution
    print("\n1. Conflict intensity distribution...")
    visualize_conflict_distribution(stats, save_dir)

    # 2. Evidence split statistics
    print("\n2. Evidence split statistics...")
    visualize_evidence_split_stats(stats, save_dir)

    # 3. Case studies
    print("\n3. Case studies...")
    samples = stats['samples']
    if len(samples) > 0:
        all_C = np.array([s['C'] for s in samples])
        high_idx = np.argmax(all_C)
        low_idx = np.argmin(all_C)
        mid_idx = np.argsort(all_C)[len(all_C)//2]

        for case_id, idx, desc in [(0, low_idx, 'low'), (1, mid_idx, 'medium'), (2, high_idx, 'high')]:
            print(f"  Case {case_id} ({desc} conflict)...")
            visualize_case_study(model, samples[idx], device, save_dir, case_id)

    # 4. C vs prediction error
    print("\n4. C vs prediction error...")
    visualize_c_vs_error(stats, save_dir)

    # 5. Pruning comparison
    print("\n5. Pruning before/after comparison...")
    visualize_pruning_comparison(stats, save_dir)

    # 6. Per-modality C_m distribution
    print("\n6. Per-modality C_m distribution...")
    visualize_cm_distribution(stats, save_dir)

    # 7. Evidence summary table
    print("\n7. Evidence summary table...")
    visualize_evidence_summary_table(stats, save_dir)

    # 8. Conflict bucket vs performance (ICR evidence)
    print("\n8. Conflict bucket vs performance...")
    visualize_conflict_bucket_performance(stats, save_dir)

    print(f"\n{'='*60}")
    print(f"All 8 visualizations saved to {save_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    import sys
    from opts import parse_opts
    from core.dataset import MMDataLoader
    from models.OverallModal import build_model

    # 额外的可视化专用参数 (在parse_opts之后手动解析)
    viz_parser = argparse.ArgumentParser(add_help=False)
    viz_parser.add_argument('--checkpoint_path', type=str, default='',
                           help='Path to checkpoint file (default: ./checkpoints/{DATASET}/best.pth)')
    viz_parser.add_argument('--save_dir', type=str, default='',
                           help='Directory to save visualizations (default: ./results/visualizations)')
    viz_args, _ = viz_parser.parse_known_args()

    # opts.parse_opts() 使用 argparse.parse_args(), 遇到未知参数会报错。
    # 因此先把可视化专用参数从 sys.argv 里移除, 再调用 parse_opts().
    filtered_argv = []
    skip_next = False
    for a in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if a in ['--checkpoint_path', '--save_dir']:
            skip_next = True
            continue
        filtered_argv.append(a)
    sys.argv = filtered_argv

    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定checkpoint路径
    if viz_args.checkpoint_path:
        ckpt_file = viz_args.checkpoint_path
    elif getattr(opt, 'checkpoint_dir', '') and opt.checkpoint_dir:
        ckpt_file = os.path.join(opt.checkpoint_dir, 'best.pth')
    else:
        ckpt_file = f'./checkpoints/{opt.datasetName.upper()}/best.pth'

    # 确定可视化保存目录
    if viz_args.save_dir:
        save_dir = viz_args.save_dir
    else:
        save_dir = './results/visualizations'

    print(f"Loading checkpoint: {ckpt_file}")
    print(f"Saving visualizations to: {save_dir}")

    ckpt = torch.load(ckpt_file, weights_only=False)
    ckpt_state = ckpt['model_state_dict']

    # 从checkpoint恢复训练时的opt参数(如vision_target_ratio等)
    if 'opt' in ckpt:
        saved_opt = ckpt['opt']
        for key in ['vision_target_ratio', 'tau_rel', 'tau_conf', 'tau_con',
                     'use_vision_pruning', 'use_conflict_js', 'use_routing',
                     'lambda_con', 'lambda_cal']:
            if key in saved_opt:
                setattr(opt, key, saved_opt[key])
                print(f"  Restored opt.{key} = {saved_opt[key]}")

    # 根据checkpoint自动推断可视化需要开启的开关
    has_conflict_js = any(k.startswith('conflict_js.') for k in ckpt_state.keys())
    has_vision_pruner = any(k.startswith('vision_pruner.') for k in ckpt_state.keys())
    has_routing = any(k.startswith('DyMultiFus.') for k in ckpt_state.keys())

    if has_conflict_js:
        opt.use_conflict_js = True
    if has_routing:
        opt.use_routing = True
    # 注意: VisionTokenPruner通常没有可训练参数, state_dict里不会出现vision_pruner.* keys。
    # 因此不能用has_vision_pruner==False来强制关闭剪枝开关。
    # 这里优先使用checkpoint中保存的opt.use_vision_pruning(上面已restore)，仅在明确检测到时强制打开。
    if has_vision_pruner:
        opt.use_vision_pruning = True

    # 加载模型
    model = build_model(opt).to(device)
    model_state = model.state_dict()
    filtered_state = {}
    skipped_shape = 0
    skipped_missing = 0
    for k, v in ckpt_state.items():
        if k not in model_state:
            skipped_missing += 1
            continue
        if v.shape != model_state[k].shape:
            skipped_shape += 1
            continue
        filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if skipped_shape:
        print(f"[WARN] Skipped keys due to shape mismatch: {skipped_shape}")
    if skipped_missing:
        print(f"[WARN] Skipped keys not present in current model: {skipped_missing}")
    if missing:
        print(f"[WARN] Missing keys (not loaded, will use random init): {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys (ignored): {len(unexpected)}")

    dataLoader = MMDataLoader(opt)

    generate_all_visualizations(model, dataLoader['test'], device, save_dir=save_dir)
