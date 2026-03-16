"""
eval_sims_single.py
====================
SIMS 单模型评估脚本。支持：
  1. 评估单个 checkpoint
  2. 扫描 checkpoints/SIMS 汇总所有配置的 summary.json

用法：
    # 评估单个 checkpoint（目录需含 best.pth）
    python scripts/eval_sims_single.py --ckpt_dir ./checkpoints/SIMS/grcf_deep/seed_2024 --gpu 0

    # 指定 checkpoint 文件
    python scripts/eval_sims_single.py --ckpt_path ./checkpoints/SIMS/grcf_deep/seed_2024/best.pth --gpu 0

    # 汇总 run_dir 下所有配置（从 summary.json 读取，无需重新推理）
    python scripts/eval_sims_single.py --run_dir ./checkpoints/SIMS --summarize

    # 强制指定数据路径（评估 BERT checkpoint 时若需指定 BERT pkl）
    python scripts/eval_sims_single.py --ckpt_dir ./checkpoints/SIMS/grcf_deep/seed_2024 --dataPath /path/to/unaligned_39.pkl --gpu 0
"""
import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import MMDataset
from core.metric import MetricsTop
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
        if "text_encoder_pretrained" in cfg:
            opt.text_encoder_pretrained = cfg["text_encoder_pretrained"]
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


def load_model_from_ckpt(ckpt_path, device, data_path_override=None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    if data_path_override:
        opt.dataPath = data_path_override
    if "roberta" in str(getattr(opt, "dataPath", "")).lower():
        from experiment_configs import DATASET_CONFIGS
        cfg = DATASET_CONFIGS.get(str(getattr(opt, "datasetName", "")).lower(), {})
        if "text_encoder_pretrained" in cfg and not getattr(opt, "text_encoder_pretrained", None):
            opt.text_encoder_pretrained = cfg["text_encoder_pretrained"]
    opt = get_dims_from_pkl(opt)
    model = build_model(opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, opt


def eval_single(ckpt_path, device, data_override=None):
    model, opt = load_model_from_ckpt(ckpt_path, device, data_path_override=data_override)
    test_dataset = MMDataset(opt, mode="test")
    data_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=getattr(opt, "num_workers", 0),
    )
    metrics_fn = MetricsTop().getMetics(opt.datasetName)

    preds, labels = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="eval"):
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
            preds.append(pred.cpu())
            labels.append(data["labels"]["M"].view(-1, 1))
    pred = torch.cat(preds)
    label = torch.cat(labels)
    return metrics_fn(pred, label)


def summarize_run_dir(run_dir):
    """扫描 run_dir 下所有 config/seed_*/summary.json，汇总 MAE/Corr"""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        print(f"[错误] 目录不存在: {run_dir}")
        return

    rows = []
    for cfg_name in sorted(os.listdir(run_dir)):
        cfg_path = os.path.join(run_dir, cfg_name)
        if not os.path.isdir(cfg_path):
            continue
        for seed_name in sorted(os.listdir(cfg_path)):
            seed_path = os.path.join(cfg_path, seed_name)
            if not os.path.isdir(seed_path):
                continue
            summary_path = os.path.join(seed_path, "summary.json")
            if not os.path.isfile(summary_path):
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            res = s.get("test_at_best_mae", {})
            mae = res.get("MAE", float("nan"))
            corr = res.get("Corr", float("nan"))
            rows.append({
                "config": cfg_name,
                "seed": seed_name,
                "mae": mae,
                "corr": corr,
            })

    if not rows:
        print(f"[提示] 未在 {run_dir} 下找到任何 summary.json")
        return

    # 按 config 分组
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)

    print("\n" + "=" * 70)
    print(f"SIMS 单模型汇总 (run_dir={run_dir})")
    print("=" * 70)
    all_best = []
    for cfg_name in sorted(by_cfg.keys()):
        subset = by_cfg[cfg_name]
        mae_list = [r["mae"] for r in subset if isinstance(r["mae"], (int, float))]
        corr_list = [r["corr"] for r in subset if isinstance(r["corr"], (int, float))]
        if not mae_list:
            print(f"  {cfg_name}: (无有效结果)")
            continue
        best_mae = min(mae_list)
        best_r = [r for r in subset if r["mae"] == best_mae][0]
        all_best.append((cfg_name, best_mae, best_r["seed"]))
        mae_mean = sum(mae_list) / len(mae_list)
        corr_mean = sum(corr_list) / len(corr_list) if corr_list else 0
        mae_std = (sum((x - mae_mean) ** 2 for x in mae_list) / len(mae_list)) ** 0.5 if len(mae_list) > 1 else 0
        print(f"  {cfg_name}:  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}  best={best_mae:.4f}({best_r['seed']})  (n={len(subset)})")
        for r in subset:
            print(f"    {r['seed']}: MAE={r['mae']:.4f}  Corr={r['corr']:.4f}")
    print("=" * 70)
    if all_best:
        best_cfg, best_val, best_seed = min(all_best, key=lambda x: x[1])
        print(f"\n最佳: {best_cfg} {best_seed}  MAE={best_val:.4f}")
        print(f"  重跑评估: python scripts/eval_sims_single.py --ckpt_dir {run_dir}/{best_cfg}/{best_seed} --gpu 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, default="",
                    help="单个 checkpoint 目录，含 best.pth（如 checkpoints/SIMS/grcf_deep/seed_2024）")
    ap.add_argument("--ckpt_path", type=str, default="",
                    help="直接指定 best.pth 路径")
    ap.add_argument("--run_dir", type=str, default="",
                    help="汇总模式：扫描目录下所有 config/seed_*/summary.json")
    ap.add_argument("--summarize", action="store_true",
                    help="与 --run_dir 同用，明确表示汇总模式（可省略）")
    ap.add_argument("--dataPath", type=str, default="",
                    help="强制指定数据 pkl 路径")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    # 汇总模式：--run_dir 或 --summarize（且未指定单 checkpoint）
    if args.run_dir:
        summarize_run_dir(args.run_dir)
        return
    if args.summarize and not args.ckpt_dir and not args.ckpt_path:
        summarize_run_dir("./checkpoints/SIMS")
        return

    # 单 checkpoint 评估
    ckpt_path = args.ckpt_path
    if not ckpt_path and args.ckpt_dir:
        ckpt_path = os.path.join(os.path.abspath(args.ckpt_dir), "best.pth")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("请指定 --ckpt_dir 或 --ckpt_path，且路径下需有 best.pth")
        return

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_override = args.dataPath if getattr(args, "dataPath", "") else None

    print(f"评估: {ckpt_path}")
    results = eval_single(ckpt_path, device, data_override=data_override)
    print("\n" + "=" * 60)
    print("测试集结果 (test_at_best_mae)")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
