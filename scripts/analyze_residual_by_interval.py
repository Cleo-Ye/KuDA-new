"""
analyze_residual_by_interval.py
================================
按真值区间拆分残差，诊断 MAE 卡在哪些样本上：
- 接近 0 的样本误差大？
- 强正/强负样本误差大？
- 是否存在「预测向 0 收缩」？

用法：
    # 单模型
    python scripts/analyze_residual_by_interval.py --ckpt ./checkpoints/multiseed_best/seed_1111/best.pth --gpu 0

    # 集成（加权平均或加权+校准）
    python scripts/analyze_residual_by_interval.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --calibrate linear --gpu 0
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


def load_model(ckpt_path, device, use_ckpt_dataPath=True, data_path_override=None):
    """加载 checkpoint。use_ckpt_dataPath=True 时保留 checkpoint 的 dataPath。data_path_override 可强制指定数据路径。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    if data_path_override:
        opt.dataPath = data_path_override
    elif use_ckpt_dataPath:
        pass  # 评估时用训练时的 dataPath，不覆盖
    else:
        opt = apply_dataset_config(opt)
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


# 默认区间：CH-SIMS 标签 [-1, 1]
DEFAULT_INTERVALS = [
    (-1.0, -0.5, "强负 [-1, -0.5)"),
    (-0.5, -0.2, "弱负 [-0.5, -0.2)"),
    (-0.2, 0.2, "接近0 [-0.2, 0.2)"),
    (0.2, 0.5, "弱正 [0.2, 0.5)"),
    (0.5, 1.0, "强正 [0.5, 1]"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="", help="单模型 best.pth")
    ap.add_argument("--ckpt_dir", type=str, default="",
                    help="多模型目录，用集成预测（需 --tune_weights 等）")
    ap.add_argument("--tune_weights", action="store_true")
    ap.add_argument("--calibrate", type=str, default="", choices=["", "linear", "piecewise", "isotonic"],
                    help="校准方式：linear / piecewise / isotonic")
    ap.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    ap.add_argument("--dataPath", type=str, default="",
                    help="强制指定数据 pkl 路径（用于评估旧 BERT checkpoint 时指定 BERT pkl）")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data_override = args.dataPath if getattr(args, "dataPath", "") else None

    ckpt_list = []
    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt_list = [args.ckpt]
    elif args.ckpt_dir:
        for name in sorted(os.listdir(args.ckpt_dir)):
            p = os.path.join(args.ckpt_dir, name, "best.pth")
            if os.path.isfile(p):
                ckpt_list.append(p)
    if not ckpt_list:
        print("未找到 checkpoint，请指定 --ckpt 或 --ckpt_dir")
        return

    model, opt = load_model(ckpt_list[0], device, data_path_override=data_override)
    dataset = MMDataset(opt, mode=args.split)
    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    preds_list = []
    label = None
    for i, p in enumerate(ckpt_list):
        model, _ = load_model(p, device, data_path_override=data_override)
        if i == 0:
            pred, label = collect_preds(model, loader, device)
        else:
            pred, _ = collect_preds(model, loader, device)
        preds_list.append(pred)

    # 集成
    weights = np.ones(len(preds_list)) / len(preds_list)
    if args.tune_weights and len(ckpt_list) > 1:
        from scipy.optimize import minimize
        valid_ds = MMDataset(opt, mode="valid")
        valid_ld = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        v_list, y_valid = [], None
        for i, p in enumerate(ckpt_list):
            m, _ = load_model(p, device, data_path_override=data_override)
            if i == 0:
                pred_v, y_valid = collect_preds(m, valid_ld, device)
            else:
                pred_v, _ = collect_preds(m, valid_ld, device)
            v_list.append(pred_v)
        V = np.stack(v_list, axis=0)
        def valid_mae(w):
            return np.abs((V * w[:, np.newaxis]).sum(axis=0) - y_valid).mean()
        res = minimize(valid_mae, x0=weights, method="SLSQP",
                      bounds=[(0, 1)] * len(weights),
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
        if res.success:
            weights = res.x

    pred_avg = (np.stack(preds_list, axis=0) * weights[:, np.newaxis]).sum(axis=0)

    # 校准（在 valid 上拟合，再应用到当前 split 的预测）
    if args.calibrate:
        from scipy.optimize import minimize
        valid_ds = MMDataset(opt, mode="valid")
        valid_ld = DataLoader(valid_ds, batch_size=opt.batch_size, shuffle=False, num_workers=0)
        v_list, y_valid = [], None
        for i, p in enumerate(ckpt_list):
            m, _ = load_model(p, device, data_path_override=data_override)
            if i == 0:
                pv, y_valid = collect_preds(m, valid_ld, device)
            else:
                pv, _ = collect_preds(m, valid_ld, device)
            v_list.append(pv)
        P_valid = (np.stack(v_list, axis=0) * weights[:, np.newaxis]).sum(axis=0)

        if args.calibrate == "linear":
            from scipy.optimize import minimize
            def cal_mae(x):
                return np.abs(y_valid - (x[0] * P_valid + x[1])).mean()
            r = minimize(cal_mae, [1.0, 0.0], method="Powell", options={"maxfev": 500})
            if r.success:
                pred_avg = np.clip(r.x[0] * pred_avg + r.x[1], -1, 1)
        elif args.calibrate == "piecewise":
            from scipy.optimize import minimize
            edges = [-1.0, -0.3, 0.3, 1.0]
            def cal_mae(x):
                a1, b1, a2, b2, a3, b3 = x
                out = np.where(P_valid < -0.3, a1 * P_valid + b1,
                               np.where(P_valid < 0.3, a2 * P_valid + b2, a3 * P_valid + b3))
                return np.abs(y_valid - out).mean()
            r = minimize(cal_mae, [1, 0, 1, 0, 1, 0], method="Powell", options={"maxfev": 500})
            if r.success:
                a1, b1, a2, b2, a3, b3 = r.x
                pred_avg = np.where(pred_avg < -0.3, a1 * pred_avg + b1,
                                   np.where(pred_avg < 0.3, a2 * pred_avg + b2, a3 * pred_avg + b3))
                pred_avg = np.clip(pred_avg, -1, 1)
        elif args.calibrate == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(P_valid, y_valid)
                pred_avg = np.clip(ir.predict(pred_avg), -1, 1)
            except ImportError:
                print("[Isotonic] 需要 sklearn，回退到 linear")
                def cal_mae(x):
                    return np.abs(y_valid - (x[0] * P_valid + x[1])).mean()
                r = minimize(cal_mae, [1.0, 0.0], method="Powell", options={"maxfev": 500})
                if r.success:
                    pred_avg = np.clip(r.x[0] * pred_avg + r.x[1], -1, 1)

    residual = pred_avg - label

    print("\n" + "=" * 75)
    print(f"残差区间分析 ({args.split}集, {'集成' if len(ckpt_list) > 1 else '单模型'})")
    print("=" * 75)
    print(f"{'区间':<25} {'样本数':>8} {'MAE':>8} {'残差均值':>10} {'预测均值':>10} {'真值均值':>10}")
    print("-" * 75)

    total_mae = 0
    total_n = 0
    for lo, hi, name in DEFAULT_INTERVALS:
        mask = (label >= lo) & (label < hi) if hi < 1.0 else (label >= lo) & (label <= hi)
        n = mask.sum()
        if n == 0:
            print(f"{name:<25} {0:>8} {'-':>8} {'-':>10} {'-':>10} {'-':>10}")
            continue
        err = np.abs(residual[mask])
        mae = err.mean()
        mean_res = residual[mask].mean()
        mean_pred = pred_avg[mask].mean()
        mean_gt = label[mask].mean()
        total_mae += err.sum()
        total_n += n
        print(f"{name:<25} {n:>8} {mae:>8.4f} {mean_res:>+10.4f} {mean_pred:>10.4f} {mean_gt:>10.4f}")

    print("-" * 75)
    print(f"{'整体':<25} {len(label):>8} {np.abs(residual).mean():>8.4f} {residual.mean():>+10.4f} {pred_avg.mean():>10.4f} {label.mean():>10.4f}")
    print("=" * 75)
    print(f"\n预测方差: {pred_avg.var():.4f}  真值方差: {label.var():.4f}  比值: {pred_avg.var() / (label.var() + 1e-9):.4f}")
    print("(若比值 < 1，说明预测向均值收缩，幅值偏保守)")


if __name__ == "__main__":
    main()
