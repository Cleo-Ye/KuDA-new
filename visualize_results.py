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
            'gate_weights': [N,] conf_weight (门控 α vs C 可视化)
            'gate_alpha': {'T'/'A'/'V': [N,]} per-modality gate alpha
            'samples': list of per-sample dicts (for case study + alignment heatmap)
    """
    model.eval()
    stats = {
        'C': [], 'C_m': {'T': [], 'A': [], 'V': []},
        'preds': [], 'labels': [],
        'modal_labels': {'T': [], 'A': [], 'V': []},   # 模态级 GT 标签，用于 C-vs-disagreement
        'con_counts': {'T': [], 'A': [], 'V': []},
        'conf_counts': {'T': [], 'A': [], 'V': []},
        'seq_lens': {},
        'vision_orig_len': 0, 'vision_pruned_len': 0,
        'gate_weights': [],
        'gate_alpha': {'T': [], 'A': [], 'V': []},
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

            # 收集模态级 GT 标签（CH-SIMS 等数据集含 T/A/V 分标签）
            for m in ['T', 'A', 'V']:
                if m in data['labels']:
                    stats['modal_labels'][m].append(data['labels'][m].cpu().squeeze())

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
                stats['vision_pruned_len'] = inputs['V'].shape[1]

            # 收集门控值（gate_conf_weight / gate_alpha）
            if hasattr(model, 'last_gate_conf_weight'):
                stats['gate_weights'].append(model.last_gate_conf_weight.cpu())
            if hasattr(model, 'last_gate_alpha'):
                for m in ['T', 'A', 'V']:
                    stats['gate_alpha'][m].append(model.last_gate_alpha[m].cpu())

            # 收集per-sample info for case study + alignment heatmap
            if hasattr(model, 'last_conflict_intensity'):
                B_bs = inputs['V'].shape[0]
                # 取 attn_vt (第一个样本) for case study
                attn_vt_batch = None
                if hasattr(model, 'last_attn_weights') and model.last_attn_weights is not None:
                    attn_vt_batch = model.last_attn_weights.get('attn_vt', None)
                for b in range(B_bs):
                    sample_dict = {
                        'vision': data['vision'][b],
                        'audio': data['audio'][b],
                        'text': data['text'][b],
                        'vision_padding_mask': data['vision_padding_mask'][b, 1:data['vision'].shape[1]+1],
                        'audio_padding_mask': data['audio_padding_mask'][b, 1:data['audio'].shape[1]+1],
                        'labels': {'M': data['labels']['M'][b]},
                        'C': model.last_conflict_intensity[b].item(),
                        'pred': pred[b].item(),
                    }
                    # 保存对齐权重切片（[L_v, L_t]）用于热力图
                    if attn_vt_batch is not None:
                        sample_dict['attn_vt'] = attn_vt_batch[b].cpu()
                    stats['samples'].append(sample_dict)

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
    if stats['gate_weights']:
        stats['gate_weights'] = torch.cat(stats['gate_weights']).numpy()
    else:
        stats['gate_weights'] = np.array([])
    for m in ['T', 'A', 'V']:
        if stats['gate_alpha'][m]:
            stats['gate_alpha'][m] = torch.cat(stats['gate_alpha'][m]).numpy()
        else:
            stats['gate_alpha'][m] = np.array([])

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
# 9. Gate Behavior: α vs C 散点 + 分箱均值曲线
# ============================================================
def visualize_gate_behavior(stats, save_dir):
    """
    门控行为可视化: conf_weight (α) vs 冲突强度 C.
    回答审稿人质疑: "路由真的发生了吗？门控是否跟随冲突强度变化？"

    图表包含:
    - 左: C vs conf_weight 散点图 + 分箱均值折线（带 std 误差棒）
    - 右: per-modality alpha (α_T/V/A) 随 C 的分箱均值
    """
    os.makedirs(save_dir, exist_ok=True)
    C = stats['C']
    gate_w = stats['gate_weights']
    if len(C) == 0 or len(gate_w) == 0:
        print("Warning: No gate weights recorded. Make sure use_conflict_js=True and model ran a forward pass.")
        return

    n_bins = 5
    bin_edges = np.linspace(C.min(), C.max() + 1e-8, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 分箱统计 conf_weight
    bin_means, bin_stds, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (C >= bin_edges[i]) & (C < bin_edges[i + 1])
        vals = gate_w[mask]
        bin_counts.append(mask.sum())
        bin_means.append(vals.mean() if len(vals) > 0 else np.nan)
        bin_stds.append(vals.std() if len(vals) > 1 else 0.0)

    has_alpha = any(len(stats['gate_alpha'][m]) > 0 for m in ['T', 'A', 'V'])
    ncols = 2 if has_alpha else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    # 左图：conf_weight vs C
    ax = axes[0]
    sc = ax.scatter(C, gate_w, c=C, cmap='RdYlBu_r', alpha=0.35, s=10, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='C')
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-',
                color='black', linewidth=2, markersize=7, capsize=5, label='Bin mean ± std')
    ax.set_xlabel('Conflict Intensity C', fontsize=12)
    ax.set_ylabel('Gate Weight (conf_weight α)', fontsize=12)
    ax.set_title('Gate Behavior: α vs Conflict Intensity C', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    corr = np.corrcoef(C, gate_w)[0, 1] if len(C) > 1 else 0.0
    ax.text(0.05, 0.95, f'Pearson r = {corr:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 右图（如有）：per-modality alpha
    if has_alpha:
        ax2 = axes[1]
        colors = {'T': '#2ECC71', 'V': '#9B59B6', 'A': '#3498DB'}
        for m in ['T', 'V', 'A']:
            alpha_vals = stats['gate_alpha'][m]
            if len(alpha_vals) == 0:
                continue
            m_means, m_stds = [], []
            for i in range(n_bins):
                mask = (C >= bin_edges[i]) & (C < bin_edges[i + 1])
                vals = alpha_vals[mask]
                m_means.append(vals.mean() if len(vals) > 0 else np.nan)
                m_stds.append(vals.std() if len(vals) > 1 else 0.0)
            ax2.errorbar(bin_centers, m_means, yerr=m_stds, fmt='o-',
                         color=colors[m], linewidth=2, markersize=7, capsize=4, label=f'α_{m}')
        ax2.set_xlabel('Conflict Intensity C (binned)', fontsize=12)
        ax2.set_ylabel('Per-modality Gate α', fontsize=12)
        ax2.set_title('Per-modality Gate α by Conflict Bin', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gate_behavior.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGate behavior visualization saved to {save_path}")
    print(f"  conf_weight: mean={gate_w.mean():.4f}, std={gate_w.std():.4f}")
    print(f"  Pearson r(C, conf_weight) = {corr:.4f}")
    for i in range(n_bins):
        print(f"  Bin {i} (n={bin_counts[i]}): C≈{bin_centers[i]:.3f}, α_mean={bin_means[i]:.4f}±{bin_stds[i]:.4f}")


# ============================================================
# 10. Alignment Heatmap Case Study (V→T attn + conflict tokens)
# ============================================================
def visualize_c_vs_label_disagreement(stats, save_dir):
    """
    分析模型的冲突强度 C 与 CH-SIMS 数据集模态级标签分歧度的关系。
    回答：C 是否捕捉到了数据集中真实存在的跨模态情感分歧（T/A/V label inconsistency）？

    绘制：
      左上: scatter C vs D_TA (|y_T - y_A|), Pearson r
      右上: scatter C vs D_TV (|y_T - y_V|), Pearson r
      左下: scatter C vs D_AV (|y_A - y_V|), Pearson r
      右下: scatter C vs D_all (mean of three pairwise), + 分桶均值折线
    """
    # 检查是否有模态标签
    has_T = len(stats['modal_labels']['T']) > 0
    has_A = len(stats['modal_labels']['A']) > 0
    has_V = len(stats['modal_labels']['V']) > 0
    if not (has_T and has_A and has_V):
        print("  Skipped: modal labels (T/A/V) not available in this dataset.")
        return

    def _to_1d(x):
        arr = x.numpy() if hasattr(x, 'numpy') else np.array(x)
        return np.atleast_1d(arr.flatten())

    C   = np.concatenate([_to_1d(c) for c in stats['C']])
    y_T = np.concatenate([_to_1d(x) for x in stats['modal_labels']['T']])
    y_A = np.concatenate([_to_1d(x) for x in stats['modal_labels']['A']])
    y_V = np.concatenate([_to_1d(x) for x in stats['modal_labels']['V']])
    C = C.flatten(); y_T = y_T.flatten(); y_A = y_A.flatten(); y_V = y_V.flatten()

    # 对齐长度（以最短为准）
    n = min(len(C), len(y_T), len(y_A), len(y_V))
    C, y_T, y_A, y_V = C[:n], y_T[:n], y_A[:n], y_V[:n]

    D_TA = np.abs(y_T - y_A)
    D_TV = np.abs(y_T - y_V)
    D_AV = np.abs(y_A - y_V)
    D_all = (D_TA + D_TV + D_AV) / 3.0

    from scipy.stats import pearsonr

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Conflict Intensity C vs. Modality Label Disagreement\n(Validating that C captures real cross-modal conflict in CH-SIMS)',
                 fontsize=13, fontweight='bold')

    pairs = [
        (axes[0, 0], D_TA, 'D_TA = |y_T - y_A|', '#e74c3c', 'Text vs Audio'),
        (axes[0, 1], D_TV, 'D_TV = |y_T - y_V|', '#3498db', 'Text vs Vision'),
        (axes[1, 0], D_AV, 'D_AV = |y_A - y_V|', '#2ecc71', 'Audio vs Vision'),
        (axes[1, 1], D_all, 'D_all = mean(D_TA, D_TV, D_AV)', '#9b59b6', 'Overall Disagreement'),
    ]

    for ax, D, xlabel, color, title in pairs:
        r, pval = pearsonr(C, D)
        # 散点（透明以免遮挡）
        ax.scatter(C, D, alpha=0.25, s=8, c=color, label=f'n={n}')
        # 分桶均值折线
        n_bins = 5
        bin_edges = np.percentile(C, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) > 2:
            bin_means_x, bin_means_y, bin_stds_y = [], [], []
            for i in range(len(bin_edges) - 1):
                mask = (C >= bin_edges[i]) & (C < bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_means_x.append(C[mask].mean())
                    bin_means_y.append(D[mask].mean())
                    bin_stds_y.append(D[mask].std())
            if bin_means_x:
                ax.errorbar(bin_means_x, bin_means_y, yerr=bin_stds_y,
                            fmt='o-', color='black', linewidth=2, markersize=6,
                            capsize=4, label='Bin mean±std', zorder=5)
        ax.set_xlabel('Conflict Intensity C', fontsize=11)
        ax.set_ylabel(xlabel, fontsize=11)
        ax.set_title(f'{title}\nPearson r = {r:.4f}  (p = {pval:.2e})', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, 'c_vs_label_disagreement.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"C vs label disagreement saved to {out_path}")
    r_TA, _ = pearsonr(C, D_TA)
    r_TV, _ = pearsonr(C, D_TV)
    r_AV, _ = pearsonr(C, D_AV)
    r_all, _ = pearsonr(C, D_all)
    print(f"  Pearson r(C, D_TA) = {r_TA:.4f}  |  r(C, D_TV) = {r_TV:.4f}")
    print(f"  Pearson r(C, D_AV) = {r_AV:.4f}  |  r(C, D_all) = {r_all:.4f}")
    # 分桶均值表格
    n_bins = 5
    bin_edges = np.percentile(C, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    print(f"\n  {'Bin':>6}  {'n':>5}  {'C_mean':>8}  {'D_TA':>8}  {'D_TV':>8}  {'D_AV':>8}  {'D_all':>8}")
    for i in range(len(bin_edges) - 1):
        mask = (C >= bin_edges[i]) & (C < bin_edges[i + 1])
        if mask.sum() > 0:
            print(f"  {i:>6}  {mask.sum():>5}  "
                  f"{C[mask].mean():>8.4f}  "
                  f"{D_TA[mask].mean():>8.4f}  "
                  f"{D_TV[mask].mean():>8.4f}  "
                  f"{D_AV[mask].mean():>8.4f}  "
                  f"{D_all[mask].mean():>8.4f}")


def visualize_alignment_heatmap(model, sample, device, save_dir, case_id=0):
    """
    对齐热力图 + 冲突 token 标注 Case Study.
    回答审稿人质疑: "你的 alignment-aware conflict 是否真的捕捉到语义矛盾？"

    图表包含:
    - 主图: V→T 注意力权重热力图 (L_v × L_t), 冲突 token 行用红色标注
    - 右侧: senti_T(j) 柱状图（文本 token 情感强度）
    - 底部: senti_V(i) 与 senti_ref_V(i) 对比（蓝=V情感, 橙=对齐参照）
    - 标注: 被判为 conflict/congruent 的帧
    """
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

    # 检查必要属性
    if not hasattr(model, 'last_attn_weights') or model.last_attn_weights is None:
        print(f"Warning: No alignment weights. Make sure use_conflict_js=True.")
        return
    if 'attn_vt' not in model.last_attn_weights:
        print("Warning: attn_vt not found in last_attn_weights.")
        return

    attn_vt = model.last_attn_weights['attn_vt'][0].cpu().numpy()  # [L_v, L_t]
    L_v, L_t = attn_vt.shape

    C_val = model.last_conflict_intensity[0].item() if hasattr(model, 'last_conflict_intensity') else 0.0
    pred_val = output[0].item()
    label_val = sample['labels']['M'].item()

    # 获取冲突/一致 token mask
    conf_mask_v = np.zeros(L_v, dtype=bool)
    con_mask_v = np.zeros(L_v, dtype=bool)
    if hasattr(model, 'last_conf_masks') and model.last_conf_masks is not None:
        if 'V' in model.last_conf_masks:
            # conf_masks 对应压缩后的 vision，长度可能 ≠ L_v；对齐到 attn_vt
            cm = model.last_conf_masks['V'][0].cpu().numpy()
            n = min(len(cm), L_v)
            conf_mask_v[:n] = cm[:n]
    if hasattr(model, 'last_con_masks') and model.last_con_masks is not None:
        if 'V' in model.last_con_masks:
            cm = model.last_con_masks['V'][0].cpu().numpy()
            n = min(len(cm), L_v)
            con_mask_v[:n] = cm[:n]

    # ---- 绘图 ----
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.8, 4, 1.5], width_ratios=[4, 0.5, 0.5],
                          hspace=0.05, wspace=0.05)

    ax_heat = fig.add_subplot(gs[1, 0])
    ax_senti_t = fig.add_subplot(gs[1, 1], sharey=None)
    ax_senti_v = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[0, :])
    ax_info.axis('off')

    # 标题信息
    ax_info.text(0.5, 0.5,
        f"Case Study #{case_id} | C = {C_val:.4f} | Pred = {pred_val:.3f} | Label = {label_val:.3f} | Error = {abs(pred_val-label_val):.3f}\n"
        f"Red rows = conflict tokens, Blue rows = congruent tokens (Vision modality)",
        ha='center', va='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    # 热力图
    im = ax_heat.imshow(attn_vt, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.01, label='Attn weight')

    # 标注冲突/一致行
    for vi in range(L_v):
        if conf_mask_v[vi]:
            ax_heat.axhspan(vi - 0.5, vi + 0.5, color='#D94A4A', alpha=0.25, lw=0)
        elif con_mask_v[vi]:
            ax_heat.axhspan(vi - 0.5, vi + 0.5, color='#4A90D9', alpha=0.20, lw=0)

    ax_heat.set_xlabel('Text token index j', fontsize=10)
    ax_heat.set_ylabel('Vision frame index i', fontsize=10)
    ax_heat.set_title('V→T Alignment Weights (attn_vt)', fontsize=11)

    # 文本 token 重要性（列均值，代理 senti_T）
    col_importance = attn_vt.mean(axis=0)  # [L_t]
    colors_t = ['#D94A4A' if v > col_importance.mean() else '#AAAAAA' for v in col_importance]
    ax_senti_t.barh(range(L_t), col_importance, color=colors_t, alpha=0.8)
    ax_senti_t.set_xlabel('Attn col-mean', fontsize=8)
    ax_senti_t.set_yticks([])
    ax_senti_t.set_title('Text\nimportance', fontsize=9)
    ax_senti_t.grid(axis='x', alpha=0.3)

    # 视觉帧行均值（代理 senti_V 对参照）
    row_attn_sum = attn_vt.sum(axis=1)  # [L_v]
    bar_colors = []
    for vi in range(L_v):
        if conf_mask_v[vi]:
            bar_colors.append('#D94A4A')
        elif con_mask_v[vi]:
            bar_colors.append('#4A90D9')
        else:
            bar_colors.append('#AAAAAA')
    ax_senti_v.bar(range(L_v), row_attn_sum, color=bar_colors, alpha=0.8)
    ax_senti_v.set_xlabel('Vision frame index i', fontsize=10)
    ax_senti_v.set_ylabel('Attn row-sum\n(alignment strength)', fontsize=9)
    ax_senti_v.set_xlim(-0.5, L_v - 0.5)
    ax_senti_v.grid(axis='y', alpha=0.3)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D94A4A', alpha=0.5, label='Conflict token'),
        Patch(facecolor='#4A90D9', alpha=0.5, label='Congruent token'),
        Patch(facecolor='#AAAAAA', alpha=0.5, label='Neutral'),
    ]
    ax_senti_v.legend(handles=legend_elements, loc='upper right', fontsize=8)

    save_path = os.path.join(save_dir, f'alignment_heatmap_case{case_id}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Alignment heatmap (case {case_id}) saved to {save_path}")
    print(f"    C={C_val:.4f}, Pred={pred_val:.3f}, Label={label_val:.3f}")


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

    # 9. Gate behavior (α vs C) — 回答"路由真的发生了吗？"
    print("\n9. Gate behavior (α vs C)...")
    visualize_gate_behavior(stats, save_dir)

    # 10. Alignment heatmap case studies — 回答"对齐感知冲突是否捕捉到语义矛盾？"
    print("\n10. Alignment heatmap case studies...")
    samples = stats['samples']
    if len(samples) > 0:
        all_C = np.array([s['C'] for s in samples])
        high_idx = int(np.argmax(all_C))
        low_idx = int(np.argmin(all_C))
        for case_id, idx in [(0, high_idx), (1, low_idx)]:
            desc = 'high' if case_id == 0 else 'low'
            print(f"  Case {case_id} ({desc} conflict, C={all_C[idx]:.4f})...")
            visualize_alignment_heatmap(model, samples[idx], device, save_dir, case_id)

    # 11. C vs 模态标签分歧度（直接验证 C 捕捉到了数据集中真实的跨模态冲突）
    print("\n11. C vs modality label disagreement...")
    visualize_c_vs_label_disagreement(stats, save_dir)

    print(f"\n{'='*60}")
    print(f"All 11 visualizations saved to {save_dir}/")
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
