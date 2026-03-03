"""
评估脚本: 加载训练好的模型进行温度缩放校准和ECE评估, 并生成 reliability diagram.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from opts import parse_opts
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from models.TemperatureScaling import calibrate_and_evaluate, compute_ece
from core.metric import MetricsTop


def plot_reliability_diagram(bin_acc_before, bin_conf_before, bin_count_before,
                             bin_acc_after, bin_conf_after, bin_count_after,
                             save_path, n_bins=15):
    """
    绘制 reliability diagram (校准曲线).
    回答审稿人质疑: "你门控用的置信信号是否可靠？Rel 可信吗？"

    显示温度缩放 before/after 对比:
    - 完美校准对角线 (黑色虚线)
    - before (红色虚线): 温度缩放前的置信度 vs 准确率
    - after (蓝色实线): 温度缩放后

    Args:
        bin_acc_before/after: [n_bins] 每桶的准确率
        bin_conf_before/after: [n_bins] 每桶的平均置信度
        bin_count_before/after: [n_bins] 每桶的样本数
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- 左图: reliability diagram ----
    ax = axes[0]
    # 对角参考线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration', zorder=3)

    # before
    valid_b = [i for i in range(n_bins) if bin_count_before[i] > 0]
    if valid_b:
        x_b = [bin_conf_before[i] for i in valid_b]
        y_b = [bin_acc_before[i] for i in valid_b]
        ax.plot(x_b, y_b, 'o--', color='#D94A4A', linewidth=2, markersize=7,
                label='Before calibration', alpha=0.9)

    # after
    valid_a = [i for i in range(n_bins) if bin_count_after[i] > 0]
    if valid_a:
        x_a = [bin_conf_after[i] for i in valid_a]
        y_a = [bin_acc_after[i] for i in valid_a]
        ax.plot(x_a, y_a, 'o-', color='#4A90D9', linewidth=2, markersize=7,
                label='After calibration (Temp. Scaling)', alpha=0.9)

    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Fraction Correct (Accuracy)', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # ---- 右图: 每桶样本量直方图 ----
    ax2 = axes[1]
    bin_centers = np.linspace(0, 1, n_bins, endpoint=False) + 0.5 / n_bins
    width = 0.8 / n_bins
    bars = ax2.bar(bin_centers, bin_count_after, width=width,
                   color='#4A90D9', alpha=0.7, label='Sample count (after)')
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Confidence Histogram', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved to {save_path}")


def evaluate_with_calibration():
    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = build_model(opt).to(device)
    
    # 加载checkpoint
    ckpt_path = os.path.join('./checkpoints', opt.datasetName.upper(), 'best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}, Valid MAE: {ckpt['valid_mae']:.4f}, Valid Corr: {ckpt['valid_corr']:.4f}")
    
    # 加载数据
    dataLoader = MMDataLoader(opt)
    
    # 温度缩放校准和ECE评估
    print("\nPerforming temperature scaling calibration...")
    temperature, ece_before, ece_after = calibrate_and_evaluate(
        model, dataLoader['valid'], num_classes=opt.senti_num_classes
    )

    print(f"\n{'='*60}")
    print(f"Calibration Results:")
    print(f"  Optimal Temperature: {temperature:.4f}")
    print(f"  ECE Before Calibration: {ece_before:.4f}")
    print(f"  ECE After Calibration: {ece_after:.4f}")
    if ece_before > 1e-8:
        print(f"  ECE Improvement: {(ece_before - ece_after):.4f} ({(ece_before - ece_after)/ece_before*100:.1f}%)")
    print(f"{'='*60}\n")

    # 生成 reliability diagram
    print("Generating reliability diagram...")
    from models.TemperatureScaling import TemperatureScaling
    import torch.nn.functional as F

    # 重新收集 logits（简化：仅用 text modality 平均后验）
    model.eval()
    logits_list, labels_list = [], []
    device_cal = next(model.parameters()).device
    n_bins_cal = 15
    boundaries_cal = torch.linspace(-1.0, 1.0, opt.senti_num_classes + 1, device=device_cal)[1:-1]

    with torch.no_grad():
        for data in dataLoader['valid']:
            inputs = {
                'V': data['vision'].to(device_cal),
                'A': data['audio'].to(device_cal),
                'T': data['text'].to(device_cal),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device_cal),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device_cal),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device_cal).view(-1)
            uni_fea, uni_senti, posteriors, senti_scores = model.UniEncKI(inputs)
            avg_post = posteriors['T'].mean(dim=1)
            label_bins = torch.bucketize(label, boundaries_cal)
            logits_list.append(torch.log(avg_post + 1e-8).cpu())
            labels_list.append(label_bins.cpu())

    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    post_before = F.softmax(logits_all, dim=-1)
    _, bin_acc_b, bin_conf_b, bin_count_b = compute_ece(post_before, labels_all, n_bins=n_bins_cal)

    ts = TemperatureScaling()
    ts.fit(logits_list, labels_list)
    with torch.no_grad():
        post_after = ts(logits_all)
    _, bin_acc_a, bin_conf_a, bin_count_a = compute_ece(post_after, labels_all, n_bins=n_bins_cal)

    os.makedirs('./results', exist_ok=True)
    plot_reliability_diagram(
        bin_acc_b, bin_conf_b, bin_count_b,
        bin_acc_a, bin_conf_a, bin_count_a,
        save_path='./results/reliability_diagram.png',
        n_bins=n_bins_cal
    )
    
    # 在测试集上评估
    metrics = MetricsTop().getMetics(opt.datasetName)
    model.eval()
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for data in dataLoader['test']:
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
    
    pred = torch.cat(y_pred)
    true = torch.cat(y_true)
    test_results = metrics(pred, true)
    
    print("Test Results:")
    print(f"  MAE:   {test_results['MAE']:.4f}")
    print(f"  Corr:  {test_results['Corr']:.4f}")
    print(f"  Acc-2: {test_results.get('Mult_acc_2', 0):.4f}")
    print(f"  F1:    {test_results.get('F1_score', 0):.4f}")


if __name__ == '__main__':
    evaluate_with_calibration()
