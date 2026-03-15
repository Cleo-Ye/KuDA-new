"""
在高冲突子集上比较 baseline、IEC+ICR(r=0.5)、IEC+ICR(r=0.3)，用于强化「IEC+ICR 必要性」的数值证据。
使用测试集 GT 模态标签 (T/V/A) 定义不一致子集：max(|y_T-y_V|, |y_T-y_A|) > threshold。

用法:
  python evaluate_high_conflict_subset.py --ckpt_root ./checkpoints --threshold 0.4
"""
import os
import sys
import argparse
import torch
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from core.metric import MetricsTop


def make_test_loader(opt):
    """构造固定顺序（shuffle=False）的测试集 DataLoader，避免每次顺序不同导致 pred 与 label 错位。"""
    from torch.utils.data import DataLoader
    from core.dataset import MMDataset
    dataset = MMDataset(opt, mode='test')
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=0,       # 0 避免多进程随机性
        shuffle=False,       # 评估必须关闭 shuffle
    )


def load_model_and_predict(ckpt_path, test_loader, device):
    """加载 checkpoint，在固定顺序测试集上跑一遍，返回 pred, label_M, label_T, label_V, label_A。"""
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt["opt"], dict):
        ckpt_opt = argparse.Namespace(**ckpt["opt"])
    else:
        ckpt_opt = ckpt["opt"]
    model = build_model(ckpt_opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    all_pred, all_m, all_t, all_v, all_a = [], [], [], [], []
    has_modal = (
        "T" in test_loader.dataset.labels
        and "V" in test_loader.dataset.labels
        and "A" in test_loader.dataset.labels
    )
    with torch.no_grad():
        for data in test_loader:
            inputs = {
                "V": data["vision"].to(device),
                "A": data["audio"].to(device),
                "T": data["text"].to(device),
                "mask": {
                    "V": data["vision_padding_mask"][:, 1 : data["vision"].shape[1] + 1].to(device),
                    "A": data["audio_padding_mask"][:, 1 : data["audio"].shape[1] + 1].to(device),
                    "T": [],
                },
            }
            out, _, _, _, _, _, _ = model(inputs, None)
            all_pred.append(out.cpu().squeeze(-1))
            all_m.append(data["labels"]["M"].cpu())
            if has_modal:
                all_t.append(data["labels"]["T"].cpu())
                all_v.append(data["labels"]["V"].cpu())
                all_a.append(data["labels"]["A"].cpu())

    pred = torch.cat(all_pred).reshape(-1)
    label_m = torch.cat(all_m).reshape(-1)
    label_t = torch.cat(all_t).reshape(-1) if has_modal else None
    label_v = torch.cat(all_v).reshape(-1) if has_modal else None
    label_a = torch.cat(all_a).reshape(-1) if has_modal else None
    return pred, label_m, label_t, label_v, label_a, has_modal


def high_conflict_mask(label_t, label_v, label_a, threshold):
    """True = 高冲突样本：max(|T-V|, |T-A|) > threshold"""
    diff_tv = torch.abs(label_t - label_v)
    diff_ta = torch.abs(label_t - label_a)
    max_diff = torch.max(diff_tv, diff_ta)
    return max_diff > threshold


def main():
    # 先解析本脚本专用参数，并从 sys.argv 中移除，避免 parse_opts() 报 unrecognized arguments
    script_parser = argparse.ArgumentParser(description="Compare baseline vs IEC+ICR_full on high-conflict subset")
    script_parser.add_argument("--ckpt_root", type=str, default="./checkpoints")
    script_parser.add_argument("--threshold", type=float, default=0.4,
                              help="Modal disagreement threshold for high-conflict subset (max(|T-V|,|T-A|) > th)")
    args, unknown = script_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from opts import parse_opts
    opt = parse_opts()

    # 构造一次固定顺序的测试 DataLoader，两个模型共用，保证 pred 与 label 顺序完全对齐
    test_loader = make_test_loader(opt)

    ckpt_baseline = os.path.join(args.ckpt_root, "full_baseline", "best.pth")
    ckpt_full = os.path.join(args.ckpt_root, "full_IEC+ICR_full", "best.pth")
    ckpt_r03 = os.path.join(args.ckpt_root, "full_IEC+ICR_r03", "best.pth")

    print(f"Loading baseline:       {ckpt_baseline}")
    print(f"Loading IEC+ICR(r=0.5): {ckpt_full}")
    print(f"Loading IEC+ICR(r=0.3): {ckpt_r03}")
    res_b = load_model_and_predict(ckpt_baseline, test_loader, device)
    res_f = load_model_and_predict(ckpt_full,    test_loader, device)
    res_r03 = load_model_and_predict(ckpt_r03,   test_loader, device)

    if res_b is None:
        print("Missing checkpoint: full_baseline/best.pth under --ckpt_root")
        return
    pred_b, label_m, label_t, label_v, label_a, has_modal = res_b
    pred_f = res_f[0] if res_f is not None else None
    pred_r03 = res_r03[0] if res_r03 is not None else None

    if not has_modal or label_t is None:
        print("Dataset has no T/V/A modal labels; reporting overall MAE only.")
        idx_high = None
        n_high = 0
        n_all = label_m.shape[0]
    else:
        mask_high = high_conflict_mask(label_t, label_v, label_a, args.threshold)
        idx_high = torch.nonzero(mask_high).squeeze(-1)
        if idx_high.dim() == 0:
            idx_high = idx_high.unsqueeze(0)
        n_high = mask_high.sum().item()
        n_all = label_m.shape[0]

    # 已在 load_model_and_predict 内 reshape(-1)，此处无需再展平

    def mae(pred, label, indices=None):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        if indices is not None and len(indices) > 0:
            return torch.abs(pred[indices] - label[indices]).mean().item()
        return torch.abs(pred - label).mean().item()

    mae_b_all = mae(pred_b, label_m)
    mae_f_all = mae(pred_f, label_m) if pred_f is not None else float("nan")
    mae_r03_all = mae(pred_r03, label_m) if pred_r03 is not None else float("nan")
    if n_high > 0 and idx_high is not None:
        mae_b_high = mae(pred_b, label_m, idx_high)
        mae_f_high = mae(pred_f, label_m, idx_high) if pred_f is not None else float("nan")
        mae_r03_high = mae(pred_r03, label_m, idx_high) if pred_r03 is not None else float("nan")
    else:
        mae_b_high = mae_f_high = mae_r03_high = float("nan")

    def fmt(x):
        return "{:.4f}".format(x) if isinstance(x, float) and not (x != x) else "N/A"

    print("")
    print("=" * 92)
    print("Baseline vs IEC+ICR(r=0.5) vs IEC+ICR(r=0.3) (test set)")
    print("  -> 对应 Full Experiments Summary 表: baseline / IEC+ICR(r=0.5) / IEC+ICR(r=0.3) 行")
    print("High-conflict subset: max(|y_T-y_V|, |y_T-y_A|) > {:.2f}".format(args.threshold))
    print("=" * 92)
    print("{:32} {:>14} {:>16} {:>16}".format("", "Baseline", "IEC+ICR(r=0.5)", "IEC+ICR(r=0.3)"))
    print("-" * 92)
    print("{:32} {:>14} {:>16} {:>16}".format(
        "MAE (all, n={})".format(n_all), fmt(mae_b_all), fmt(mae_f_all), fmt(mae_r03_all)))
    if n_high > 0:
        print("{:32} {:>14} {:>16} {:>16}".format(
            "MAE (high-conflict, n={})".format(n_high), fmt(mae_b_high), fmt(mae_f_high), fmt(mae_r03_high)))
        if mae_b_high > 1e-6 and pred_f is not None:
            pct_f = (mae_f_high - mae_b_high) / mae_b_high * 100
            print("  -> vs baseline on high-conflict: IEC+ICR(r=0.5) {:+.1f}%".format(pct_f))
        if mae_b_high > 1e-6 and pred_r03 is not None:
            pct_r03 = (mae_r03_high - mae_b_high) / mae_b_high * 100
            print("  -> vs baseline on high-conflict: IEC+ICR(r=0.3) {:+.1f}% (negative = better)".format(pct_r03))
    print("=" * 92)
    print("")
    print("If |Δ| on high-conflict subset is larger than on full set, the module is more necessary on conflicting samples.")
    print("")
    print("Note: MAE(all) should be close to the table (baseline ~0.32, IEC+ICR(r=0.5) ~0.30, IEC+ICR(r=0.3) ~0.31).")
    print("      Missing column = checkpoint not found under --ckpt_root.")
    print("")


if __name__ == "__main__":
    main()
