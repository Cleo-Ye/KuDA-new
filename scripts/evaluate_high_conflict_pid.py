"""
在 pid_dualpath 模型的高冲突子集上评估 MAE。
高冲突定义：max(|y_T-y_V|, |y_T-y_A|) > threshold（模态标签分歧大）。

用法:
  python scripts/evaluate_high_conflict_pid.py --ckpt_path ./checkpoints/sweep_unified/best_run/best.pth
  python scripts/evaluate_high_conflict_pid.py --ckpt_dir ./checkpoints/sweep_unified/lr2e-5_bl1e-5_bs32_sd2024_pl2_... --threshold 0.4

注意：sweep 默认 --no_save_model 不保存 checkpoint。需先用最佳配置重训并加 --save_model 得到 best.pth。
"""
import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import MMDataset
from models.OverallModal import build_model


def get_dims_from_pkl(opt):
    p = getattr(opt, "dataPath", None)
    if not p or not os.path.isfile(p):
        return opt
    try:
        import pickle
        with open(p, "rb") as f:
            data = pickle.load(f)
        d = data.get("train", data)
        v, a = d.get("vision"), d.get("audio")
        if v is not None and a is not None:
            opt.fea_dims[1] = int(v.shape[-1])
            opt.fea_dims[2] = int(a.shape[-1])
    except Exception:
        pass
    return opt


def high_conflict_mask(label_t, label_v, label_a, threshold):
    """True = 高冲突样本：max(|T-V|, |T-A|) > threshold"""
    diff_tv = torch.abs(label_t - label_v)
    diff_ta = torch.abs(label_t - label_a)
    max_diff = torch.max(diff_tv, diff_ta)
    return max_diff > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", type=str, default="", help="best.pth 路径")
    ap.add_argument("--ckpt_dir", type=str, default="", help="checkpoint 目录，自动找 best.pth")
    ap.add_argument("--threshold", type=float, default=0.4, help="高冲突阈值 max(|T-V|,|T-A|)>th")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = args.ckpt_path
    if not ckpt_path and args.ckpt_dir:
        ckpt_path = os.path.join(args.ckpt_dir, "best.pth")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Sweep 默认不保存模型。请用最佳配置重训并加 --save_model 得到 best.pth")
        return 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    opt = get_dims_from_pkl(opt)

    from experiment_configs import DATASET_CONFIGS
    cfg = DATASET_CONFIGS.get(str(getattr(opt, "datasetName", "")).lower(), {})
    if cfg.get("dataPath"):
        opt.dataPath = cfg["dataPath"]

    model = build_model(opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_dataset = MMDataset(opt, mode="test")
    has_modal = "T" in test_dataset.labels and "V" in test_dataset.labels and "A" in test_dataset.labels
    if not has_modal:
        print("Dataset 无 T/V/A 模态标签，无法计算高冲突子集")
        return 1

    loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    preds, labels_m, labels_t, labels_v, labels_a = [], [], [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc="eval"):
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
            out = model(inputs, None)
            pred = out["pred"] if isinstance(out, dict) else out[0]
            preds.append(pred.cpu().squeeze(-1))
            labels_m.append(data["labels"]["M"])
            labels_t.append(data["labels"]["T"])
            labels_v.append(data["labels"]["V"])
            labels_a.append(data["labels"]["A"])

    pred = torch.cat(preds).reshape(-1)
    label_m = torch.cat(labels_m).reshape(-1)
    label_t = torch.cat(labels_t).reshape(-1)
    label_v = torch.cat(labels_v).reshape(-1)
    label_a = torch.cat(labels_a).reshape(-1)

    mask_high = high_conflict_mask(label_t, label_v, label_a, args.threshold)
    idx_high = torch.nonzero(mask_high).squeeze(-1)
    if idx_high.dim() == 0:
        idx_high = idx_high.unsqueeze(0)
    n_high = mask_high.sum().item()
    n_all = label_m.shape[0]

    mae_all = torch.abs(pred - label_m).mean().item()
    mae_high = torch.abs(pred[idx_high] - label_m[idx_high]).mean().item() if n_high > 0 else float("nan")

    print("")
    print("=" * 60)
    print(f"High-Conflict Evaluation (pid_dualpath)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Threshold: max(|T-V|, |T-A|) > {args.threshold}")
    print("=" * 60)
    print(f"  MAE (all, n={n_all}):        {mae_all:.4f}")
    print(f"  MAE (high-conflict, n={n_high}): {mae_high:.4f}")
    if n_high > 0 and not (mae_high != mae_high):
        ratio = mae_high / mae_all if mae_all > 1e-6 else 0
        print(f"  Ratio (high/all):           {ratio:.2f}x")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
