import argparse
import json
import os
from typing import Dict, Any, List

import torch

from core.dataset import MMDataLoader
from core.metric import MetricsTop
from models.OverallModal import build_model


# SIMS metrics keys (与 core/metric.py __eval_sims_regression 一致)
SIMS_METRIC_KEYS = ["MAE", "Corr", "Mult_acc_2", "Mult_acc_3", "Mult_acc_5", "F1_score"]
# 表格显示用的短名
METRIC_DISPLAY = {
    "MAE": "MAE ↓",
    "Corr": "Corr ↑",
    "Mult_acc_2": "Acc-2 ↑",
    "Mult_acc_3": "Acc-3 ↑",
    "Mult_acc_5": "Acc-5 ↑",
    "F1_score": "F1 ↑",
}


def _filter_state_dict_for_model(model: torch.nn.Module, ckpt_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt_state.items():
        if k not in model_state:
            continue
        if hasattr(v, "shape") and hasattr(model_state[k], "shape") and v.shape != model_state[k].shape:
            continue
        filtered[k] = v
    return filtered


def _load_checkpoint_build_model(ckpt_path: str, device: torch.device, overrides: Dict[str, Any]):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    saved_opt = ckpt.get("opt", {}) or {}

    class _Opt:
        pass

    opt = _Opt()
    for k, v in saved_opt.items():
        setattr(opt, k, v)

    for k, v in overrides.items():
        if v is not None:
            setattr(opt, k, v)

    model = build_model(opt).to(device)

    ckpt_state = ckpt.get("model_state_dict", ckpt)
    filtered_state = _filter_state_dict_for_model(model, ckpt_state)
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    ckpt_valid_mae = float(ckpt.get("valid_mae", float("nan")))
    ckpt_valid_corr = float(ckpt.get("valid_corr", float("nan")))

    return opt, model, ckpt_valid_mae, ckpt_valid_corr


@torch.no_grad()
def _eval_on_split(model: torch.nn.Module, loader, device: torch.device, metrics_fn):
    y_pred = []
    y_true = []

    for data in loader:
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
        label = data['labels']['M'].to(device).view(-1, 1)
        output, _, _, _, _, _, _ = model(inputs, None)
        y_pred.append(output.detach().cpu())
        y_true.append(label.detach().cpu())

    pred = torch.cat(y_pred, dim=0)
    true = torch.cat(y_true, dim=0)
    results = metrics_fn(pred, true)
    count = int(true.numel())
    return results, count


def _to_md_table(rows: List[Dict[str, Any]], split: str) -> str:
    headers = ["Experiment"] + [METRIC_DISPLAY[k] for k in SIMS_METRIC_KEYS] + ["Count"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [r["name"]]
        for k in SIMS_METRIC_KEYS:
            v = r["metrics"].get(k, float("nan"))
            cells.append(f"{v:.4f}")
        cells.append(str(r["count"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _print_pretty_table(rows: List[Dict[str, Any]], split: str):
    col_names = ["Experiment"] + [METRIC_DISPLAY[k] for k in SIMS_METRIC_KEYS] + ["Count"]
    # 构建每行数据
    table_rows = []
    for r in rows:
        cells = [r["name"]]
        for k in SIMS_METRIC_KEYS:
            v = r["metrics"].get(k, float("nan"))
            cells.append(f"{v:.4f}")
        cells.append(str(r["count"]))
        table_rows.append(cells)
    # 计算列宽
    col_widths = [max(len(col_names[i]), max(len(row[i]) for row in table_rows)) for i in range(len(col_names))]
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header = "| " + " | ".join(col_names[i].ljust(col_widths[i]) for i in range(len(col_names))) + " |"
    print(sep)
    print(header)
    print(sep)
    for row in table_rows:
        line = "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |"
        print(line)
    print(sep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetName", type=str, default="SIMS")
    parser.add_argument("--dataPath", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output_json", type=str, default="./results/ablation_eval_table.json")

    # 支持任意数量的checkpoint和标签
    parser.add_argument("--checkpoints", nargs='+', type=str, required=True,
                        help="Checkpoint paths to evaluate")
    parser.add_argument("--labels", nargs='+', type=str, default=None,
                        help="Labels for each checkpoint (default: derived from path)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建 (name, path) 列表
    if args.labels and len(args.labels) == len(args.checkpoints):
        ckpts = list(zip(args.labels, args.checkpoints))
    else:
        # 从路径自动推导名称
        ckpts = []
        for p in args.checkpoints:
            name = os.path.basename(os.path.dirname(p))
            ckpts.append((name, p))

    metrics_fn = MetricsTop().getMetics(args.datasetName)

    rows: List[Dict[str, Any]] = []
    for name, ckpt_path in ckpts:
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint not found, skipping: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}  ({ckpt_path})")

        overrides = {
            "datasetName": args.datasetName,
            "dataPath": args.dataPath,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }

        opt, model, ckpt_valid_mae, ckpt_valid_corr = _load_checkpoint_build_model(
            ckpt_path=ckpt_path,
            device=device,
            overrides=overrides,
        )

        loaders = MMDataLoader(opt)
        results, count = _eval_on_split(model, loaders[args.split], device, metrics_fn)

        print(f"  Split={args.split}, Count={count}")
        for k in SIMS_METRIC_KEYS:
            print(f"  {k}: {results.get(k, 'N/A')}")

        rows.append({
            "name": name,
            "ckpt_path": ckpt_path,
            "count": count,
            "metrics": {k: float(results.get(k, float('nan'))) for k in SIMS_METRIC_KEYS},
            "ckpt_valid_mae": ckpt_valid_mae,
            "ckpt_valid_corr": ckpt_valid_corr,
        })

        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    out = {
        "split": args.split,
        "datasetName": args.datasetName,
        "rows": rows,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"# Ablation Comparison on [{args.split}] set\n")
    _print_pretty_table(rows, args.split)
    print(f"\n(Markdown format):")
    print(_to_md_table(rows, args.split))
    print(f"\nResults saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
