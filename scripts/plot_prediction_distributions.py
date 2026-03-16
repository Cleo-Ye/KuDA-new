"""
plot_prediction_distributions.py
=================================
画三分布对比图：ground truth、单模型预测、集成预测。
用于诊断是否存在「预测向 0 收缩」「输出方差明显小于真值」等现象。

用法：
    # 单模型 + 集成对比
    python scripts/plot_prediction_distributions.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --gpu 0 -o dist.png

    # 指定单模型
    python scripts/plot_prediction_distributions.py --ckpt ./checkpoints/multiseed_best/seed_1111/best.pth --gpu 0 -o dist.png
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import MMDataset
from models.OverallModal import build_model


def apply_dataset_config(opt):
    from experiment_configs import DATASET_CONFIGS
    key = str(getattr(opt, "datasetName", "")).lower()
    cfg = DATASET_CONFIGS.get(key)
    if cfg:
        if "dataPath" in cfg:
            opt.dataPath = cfg["dataPath"]
        if "seq_lens" in cfg:
            opt.seq_lens = list(cfg["seq_lens"])
        if "fea_dims" in cfg:
            opt.fea_dims = list(cfg["fea_dims"])
    return opt


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


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    opt = apply_dataset_config(opt)
    opt = get_dims_from_pkl(opt)
    model = build_model(opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, opt


def collect_preds(model, loader, device):
    preds, labels = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc="pred", leave=False):
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
            preds.append(pred.cpu().numpy().ravel())
            labels.append(data["labels"]["M"].numpy().ravel())
    return np.concatenate(preds), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="")
    ap.add_argument("--ckpt_dir", type=str, default="")
    ap.add_argument("--tune_weights", action="store_true")
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    ap.add_argument("-o", "--output", type=str, default="prediction_distributions.png")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_list = []
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_list = [args.ckpt]
    elif args.ckpt_dir:
        for name in sorted(os.listdir(args.ckpt_dir)):
            p = os.path.join(args.ckpt_dir, name, "best.pth")
            if os.path.isfile(p):
                ckpt_list.append(p)
    if not ckpt_list:
        print("未找到 checkpoint")
        return

    model, opt = load_model(ckpt_list[0], device)
    dataset = MMDataset(opt, mode=args.split)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    preds_list = []
    label = None
    for i, p in enumerate(ckpt_list):
        model, _ = load_model(p, device)
        if i == 0:
            pred, label = collect_preds(model, loader, device)
        else:
            pred, _ = collect_preds(model, loader, device)
        preds_list.append(pred)

    pred_single = preds_list[0]
    weights = np.ones(len(preds_list)) / len(preds_list)
    if args.tune_weights and len(ckpt_list) > 1:
        from scipy.optimize import minimize
        valid_ds = MMDataset(opt, mode="valid")
        valid_ld = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        v_list, y_valid = [], None
        for i, p in enumerate(ckpt_list):
            m, _ = load_model(p, device)
            if i == 0:
                pv, y_valid = collect_preds(m, valid_ld, device)
            else:
                pv, _ = collect_preds(m, valid_ld, device)
            v_list.append(pv)
        V = np.stack(v_list, axis=0)
        def valid_mae(w):
            return np.abs((V * w[:, np.newaxis]).sum(axis=0) - y_valid).mean()
        res = minimize(valid_mae, x0=weights, method="SLSQP",
                      bounds=[(0, 1)] * len(weights),
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        if res.success:
            weights = res.x

    pred_ensemble = (np.stack(preds_list, axis=0) * weights[:, np.newaxis]).sum(axis=0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(-1.05, 1.05, 43)
    ax.hist(label, bins=bins, alpha=0.5, label=f"Ground Truth (var={label.var():.3f})", color="green", density=True)
    ax.hist(pred_single, bins=bins, alpha=0.5, label=f"单模型 (var={pred_single.var():.3f})", color="blue", density=True)
    ax.hist(pred_ensemble, bins=bins, alpha=0.5, label=f"集成 (var={pred_ensemble.var():.3f})", color="red", density=True)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"预测分布对比 ({args.split}集)")
    ax.legend()
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(_ROOT, out_path)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"已保存: {out_path}")
    print(f"真值: mean={label.mean():.4f} std={label.std():.4f} var={label.var():.4f}")
    print(f"单模型: mean={pred_single.mean():.4f} std={pred_single.std():.4f} var={pred_single.var():.4f}")
    print(f"集成: mean={pred_ensemble.mean():.4f} std={pred_ensemble.std():.4f} var={pred_ensemble.var():.4f}")
    print(f"方差比(集成/真值): {pred_ensemble.var() / (label.var() + 1e-9):.4f}")


if __name__ == "__main__":
    main()
