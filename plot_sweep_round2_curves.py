#!/usr/bin/env python3
"""
绘制 sweep 各配置的验证曲线（Valid MAE / Valid Corr vs epoch）。
用法:
  python plot_sweep_round2_curves.py                          # round2
  python plot_sweep_round2_curves.py --ckpt_root ./checkpoints/sweep_f1  # F1 优化
"""
import os
import csv
import argparse
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROUND2_CONFIGS = [
    ("R2-0 Baseline", "r2_baseline"),
    ("R2-1 C6", "r2_c6"),
    ("R2-2 C1", "r2_c1"),
    ("R2-3 C1+C6", "r2_c1c6"),
    ("R2-4 C5+C6", "r2_c5c6"),
    ("R2-5 C1+C5", "r2_c1c5"),
]
F1_OPT_CONFIGS = [
    ("F1-0 Baseline (C1+C5)", "f1_baseline"),
    ("F1-1 λ_cls=0.35", "f1_l035"),
    ("F1-2 λ_cls=0.4", "f1_l04"),
    ("F1-3 λ_cls=0.35+pw", "f1_l035_pw"),
]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_curve(path):
    if not os.path.isfile(path):
        return None, None, None
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f, skipinitialspace=True)
        rows = list(r)
    if not rows:
        return None, None, None
    epoch, valid_mae, valid_corr = [], [], []
    for row in rows:
        try:
            ep = int(row.get("epoch", row.get("\ufeffepoch", "")).strip())
            epoch.append(ep)
            valid_mae.append(float(row.get("valid_mae", 0)))
            valid_corr.append(float(row.get("valid_corr", 0)))
        except (ValueError, KeyError):
            continue
    if not epoch:
        return None, None, None
    return epoch, valid_mae, valid_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints/sweep_round2")
    args = parser.parse_args()
    ckpt_root = args.ckpt_root.rstrip("/")
    out_dir = ckpt_root
    if "sweep_f1" in ckpt_root:
        configs = F1_OPT_CONFIGS
        prefix = "f1"
    else:
        configs = ROUND2_CONFIGS
        prefix = "round2"
    out_mae = os.path.join(out_dir, f"{prefix}_valid_mae.png")
    out_corr = os.path.join(out_dir, f"{prefix}_valid_corr.png")
    out_combo = os.path.join(out_dir, f"{prefix}_valid_curves.png")

    os.makedirs(out_dir, exist_ok=True)
    fig, (ax_mae, ax_corr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for (label, subdir), color in zip(configs, COLORS[: len(configs)]):
        path = os.path.join(ckpt_root, subdir, "valid_curve.csv")
        epoch, valid_mae, valid_corr = load_curve(path)
        if epoch is None:
            print(f"Skip (no file): {path}")
            continue
        ax_mae.plot(epoch, valid_mae, label=label, color=color, alpha=0.9)
        ax_corr.plot(epoch, valid_corr, label=label, color=color, alpha=0.9)

    ax_mae.set_ylabel("Valid MAE (lower better)")
    ax_mae.set_title(f"{prefix.upper()}: Valid MAE vs Epoch")
    ax_mae.legend(loc="upper right", fontsize=8)
    ax_mae.grid(True, alpha=0.3)
    ax_mae.set_ylim(bottom=0)

    ax_corr.set_ylabel("Valid Corr (higher better)")
    ax_corr.set_xlabel("Epoch")
    ax_corr.set_title(f"{prefix.upper()}: Valid Corr vs Epoch")
    ax_corr.legend(loc="lower right", fontsize=8)
    ax_corr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_combo, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_combo}")

    # 单图：MAE
    fig, ax = plt.subplots(figsize=(10, 4))
    for (label, subdir), color in zip(configs, COLORS[: len(configs)]):
        path = os.path.join(ckpt_root, subdir, "valid_curve.csv")
        epoch, valid_mae, _ = load_curve(path)
        if epoch is None:
            continue
        ax.plot(epoch, valid_mae, label=label, color=color, alpha=0.9)
    ax.set_ylabel("Valid MAE")
    ax.set_xlabel("Epoch")
    ax.set_title(f"{prefix.upper()}: Valid MAE vs Epoch")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_mae, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_mae}")

    # 单图：Corr
    fig, ax = plt.subplots(figsize=(10, 4))
    for (label, subdir), color in zip(configs, COLORS[: len(configs)]):
        path = os.path.join(ckpt_root, subdir, "valid_curve.csv")
        epoch, _, valid_corr = load_curve(path)
        if epoch is None:
            continue
        ax.plot(epoch, valid_corr, label=label, color=color, alpha=0.9)
    ax.set_ylabel("Valid Corr")
    ax.set_xlabel("Epoch")
    ax.set_title(f"{prefix.upper()}: Valid Corr vs Epoch")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_corr, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_corr}")


if __name__ == "__main__":
    main()
