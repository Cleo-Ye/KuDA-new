"""
快速多配置对比脚本：少量 epoch 跑多种配置，汇总结果并可视化分布。
用法: python quick_compare_configs.py [--epochs 2] [--out_dir ./results/quick_compare]
"""
import os
import sys
import json
import copy
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import setup_seed
from models.OverallModal import build_model
from core.metric import MetricsTop


# 要对比的配置列表（名称 -> 覆盖的 opt 字段）
QUICK_CONFIGS = {
    "baseline": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": False,
        "use_vision_pruning": False,
    },
    "+IEC(r=0.5)": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": False,
        "use_vision_pruning": True,
        "vision_target_ratio": 0.5,
    },
    "+ICR": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": True,
        "use_vision_pruning": False,
    },
    "IEC+ICR(r=0.5)": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": True,
        "use_vision_pruning": True,
        "vision_target_ratio": 0.5,
    },
    "IEC+ICR(r=0.3)": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": True,
        "use_vision_pruning": True,
        "vision_target_ratio": 0.3,
    },
    "IEC+ICR(metric=KL)": {
        "use_ki": False,
        "use_cmvn": True,
        "use_conflict_js": True,
        "use_vision_pruning": True,
        "vision_target_ratio": 0.5,
        "conflict_metric": "kl",
    },
}


def train_few_epochs(model, data_loader, opt, device, n_epochs):
    """训练 n_epochs 个 epoch，返回最后一轮验证 MAE（用于早停对比）。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    loss_fn = torch.nn.MSELoss()
    scheduler = get_scheduler(optimizer, n_epochs)
    lambda_nce = getattr(opt, "lambda_nce", 0.1)
    lambda_senti = getattr(opt, "lambda_senti", 0.05)
    lambda_js = getattr(opt, "lambda_js", 0.1)
    lambda_con = getattr(opt, "lambda_con", 0.1)
    lambda_cal = getattr(opt, "lambda_cal", 0.1)

    for epoch in range(1, n_epochs + 1):
        model.train()
        for data in data_loader["train"]:
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
            label = data["labels"]["M"].to(device).view(-1, 1)
            copy_label = label.clone().detach()
            gt_modal = None
            if "T" in data["labels"] and "A" in data["labels"] and "V" in data["labels"]:
                gt_modal = {
                    "T": data["labels"]["T"].to(device).float(),
                    "A": data["labels"]["A"].to(device).float(),
                    "V": data["labels"]["V"].to(device).float(),
                }
            output, nce_loss, senti_aux, js_loss, con_loss, cal_loss, _ = model(
                inputs, copy_label, gt_modal_labels=gt_modal
            )
            loss = (
                loss_fn(output, label)
                + lambda_nce * nce_loss
                + lambda_senti * senti_aux
                + lambda_js * js_loss
                + lambda_con * con_loss
                + lambda_cal * cal_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return None


def evaluate_test(model, data_loader, opt, device):
    """在测试集上评估，返回 MAE, Corr, Mult_acc_2, F1_score 等。"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
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
            label = data["labels"]["M"].to(device).view(-1, 1)
            output, _, _, _, _, _, _ = model(inputs, None)
            all_preds.append(output.cpu())
            all_labels.append(label.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    metrics_fn = MetricsTop().getMetics(opt.datasetName)
    results = metrics_fn(preds, labels)
    return results


def run_one_config(config_name, config_overrides, base_opt, device, n_epochs):
    """跑单配置：覆盖 base_opt → 建模型 → 短训 → 测试集评估。"""
    opt = copy.deepcopy(base_opt)
    for k, v in config_overrides.items():
        setattr(opt, k, v)
    opt.n_epochs = n_epochs
    setup_seed(opt.seed)
    model = build_model(opt).to(device)
    data_loader = MMDataLoader(opt)
    train_few_epochs(model, data_loader, opt, device, n_epochs)
    test_results = evaluate_test(model, data_loader["test"], opt, device)
    return {
        "config": config_name,
        **test_results,
    }


def main():
    # 用 parse_known_args 先取本脚本参数，剩余参数给 parse_opts()
    parser = argparse.ArgumentParser(description="Quick multi-config comparison (few epochs)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per config (default 2)")
    parser.add_argument("--out_dir", type=str, default="./results/quick_compare", help="Output directory")
    parser.add_argument("--configs", type=str, nargs="*", default=None,
                        help="Config names to run (default: all in QUICK_CONFIGS)")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    configs_to_run = args.configs or list(QUICK_CONFIGS.keys())
    results_list = []
    for name in tqdm(configs_to_run, desc="Configs"):
        if name not in QUICK_CONFIGS:
            print(f"Unknown config: {name}, skip.")
            continue
        try:
            row = run_one_config(
                name,
                QUICK_CONFIGS[name],
                base_opt,
                device,
                args.epochs,
            )
            results_list.append(row)
            acc2 = row.get("Mult_acc_2", row.get("Has0_acc_2", 0))
            f1 = row.get("F1_score", row.get("Has0_F1_score", 0))
            print(f"  {name}: MAE={row['MAE']:.4f}, Corr={row['Corr']:.4f}, Acc_2={acc2:.4f}, F1={f1:.4f}")
        except Exception as e:
            import traceback
            print(f"  {name} FAILED: {e}")
            traceback.print_exc()
            results_list.append({"config": name, "error": str(e)})

    # 保存 JSON（将 numpy 标量转为 Python 原生类型）
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    out_json = os.path.join(args.out_dir, "quick_compare_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(to_serializable(results_list), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_json}")

    # 保存 CSV（仅有效行）
    valid_for_csv = [r for r in results_list if "error" not in r]
    if valid_for_csv:
        import csv
        out_csv = os.path.join(args.out_dir, "quick_compare_results.csv")
        keys = ["config"] + [k for k in valid_for_csv[0].keys() if k != "config"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(valid_for_csv)
        print(f"CSV saved to {out_csv}")

    # 可视化：只画有效结果
    valid = [r for r in results_list if "error" not in r]
    if not valid:
        print("No valid results to plot.")
        return

    # 指标列：SIMS 用 Mult_acc_2/F1_score，MOSI/MOSEI 用 Has0_acc_2/Has0_F1_score
    candidate_metrics = ["MAE", "Corr", "Mult_acc_2", "F1_score", "Has0_acc_2", "Has0_F1_score"]
    metric_keys = [k for k in candidate_metrics if k in valid[0]]
    if not metric_keys:
        metric_keys = ["MAE", "Corr"]

    names = [r["config"] for r in valid]
    x = np.arange(len(names))
    width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(4 + len(names) * 1.2, 5))
    for i, mk in enumerate(metric_keys):
        vals = [r.get(mk, np.nan) for r in valid]
        off = (i - len(metric_keys) / 2 + 0.5) * width
        bars = ax.bar(x + off, vals, width, label=mk)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Quick Compare ({args.epochs} epochs each)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(args.out_dir, "quick_compare_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    # 简单表格打印
    print("\n" + "=" * 70)
    print("Quick Compare Summary (test set)")
    print("=" * 70)
    for r in valid:
        parts = [f"{k}={r.get(k, 'N/A')}" for k in metric_keys]
        print(f"  {r['config']}: " + ", ".join(parts))
    print("=" * 70)


if __name__ == "__main__":
    main()
