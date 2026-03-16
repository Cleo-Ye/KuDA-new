"""
eval_ensemble.py
================
对多个 checkpoint（如多 seed 的 best.pth）做测试集预测取平均，得到集成后的 MAE/Corr。
可选：在验证集上优化权重（--tune_weights）或线性校准（--calibrate），进一步压 MAE。

用法：
    # 简单平均
    python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --gpu 0

    # 在验证集上优化集成权重，再在测试集上评估（通常 MAE 更低）
    python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --gpu 0

    # 验证集上做线性校准 pred_cal = a*pred + b
    python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --calibrate linear --gpu 0

    # 分段线性校准（按 pred 区间 [-1,-0.3], [-0.3,0.3], [0.3,1] 分段拟合）
    python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --calibrate piecewise --gpu 0

    # isotonic regression 单调非参数校准
    python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --calibrate isotonic --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

# 保证从项目根目录可导入 core / models（在 ~/KuDA 下执行时）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from scipy.optimize import minimize
from tqdm import tqdm

from torch.utils.data import DataLoader
from core.dataset import MMDataLoader, MMDataset
from models.OverallModal import build_model
from core.metric import MetricsTop


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


def load_model_from_ckpt(ckpt_path, device, use_ckpt_dataPath=True, data_path_override=None):
    """加载 checkpoint。use_ckpt_dataPath=True 时保留 checkpoint 的 dataPath，避免用 RoBERTa 数据评估 BERT 训练的模型导致 index 越界。data_path_override 可强制指定数据路径。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    if data_path_override:
        opt.dataPath = data_path_override
    elif use_ckpt_dataPath:
        # 评估时用训练时的 dataPath，不覆盖
        pass
    else:
        opt = apply_dataset_config(opt)
    # RoBERTa pkl 时需 text_encoder_pretrained，从 config 补充
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


def collect_test_predictions_and_labels(model, data_loader, device, return_labels=False):
    preds, labels = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="pred", leave=False):
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
            if return_labels:
                labels.append(data["labels"]["M"].view(-1, 1))
    if return_labels:
        return torch.cat(preds), torch.cat(labels)
    return torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[], help="多个 best.pth 路径")
    ap.add_argument("--ckpt_dir", type=str, default="",
                    help="指定目录时，自动用该目录下所有 seed_*/best.pth 做集成")
    ap.add_argument("--tune_weights", action="store_true",
                    help="在验证集上优化集成权重以最小化 MAE，再在测试集评估")
    ap.add_argument("--calibrate", type=str, default="",
                    choices=["", "linear", "piecewise", "isotonic"],
                    help="校准方式：linear(a*pred+b) / piecewise(分段线性) / isotonic(单调回归)")
    ap.add_argument("--dataPath", type=str, default="",
                    help="强制指定数据 pkl 路径，覆盖 checkpoint 中的路径（用于评估旧 BERT checkpoint 时指定 BERT pkl）")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_list = list(args.ckpts)
    if args.ckpt_dir:
        for name in sorted(os.listdir(args.ckpt_dir)):
            p = os.path.join(args.ckpt_dir, name, "best.pth")
            if os.path.isfile(p):
                ckpt_list.append(p)
    if not ckpt_list:
        print("未找到任何 checkpoint，请指定 --ckpts 或 --ckpt_dir")
        return

    print(f"集成 {len(ckpt_list)} 个 checkpoint:")
    for p in ckpt_list:
        print(f"  {p}")

    models = []
    opt0 = None
    data_override = args.dataPath if getattr(args, "dataPath", "") else None
    for p in ckpt_list:
        model, opt = load_model_from_ckpt(p, device, data_path_override=data_override)
        models.append(model)
        if opt0 is None:
            opt0 = opt

    # 必须 shuffle=False，否则每次遍历测试集顺序不同，多模型预测与标签错位会导致集成指标异常
    test_dataset = MMDataset(opt0, mode="test")
    data_loader = DataLoader(
        test_dataset,
        batch_size=opt0.batch_size,
        shuffle=False,
        num_workers=getattr(opt0, "num_workers", 0),
    )
    metrics_fn = MetricsTop().getMetics(opt0.datasetName)

    all_preds = []
    label = None
    for i, model in enumerate(models):
        if i == 0:
            pred, label = collect_test_predictions_and_labels(model, data_loader, device, return_labels=True)
        else:
            pred = collect_test_predictions_and_labels(model, data_loader, device, return_labels=False)
        all_preds.append(pred)

    stacked = torch.stack(all_preds, dim=0)  # (n_models, n_test)
    weights = np.ones(len(models)) / len(models)
    calibrate_a, calibrate_b = 1.0, 0.0

    need_valid = getattr(args, "tune_weights", False) or bool(getattr(args, "calibrate", ""))
    V, Y_valid = None, None
    if need_valid:
        valid_dataset = MMDataset(opt0, mode="valid")
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=opt0.batch_size,
            shuffle=False,
            num_workers=getattr(opt0, "num_workers", 0),
        )
        valid_preds_list = []
        valid_label = None
        for i, model in enumerate(models):
            if i == 0:
                p, valid_label = collect_test_predictions_and_labels(
                    model, valid_loader, device, return_labels=True
                )
            else:
                p = collect_test_predictions_and_labels(
                    model, valid_loader, device, return_labels=False
                )
            valid_preds_list.append(p.numpy().squeeze(-1))  # (n_valid,)
        V = np.stack(valid_preds_list, axis=0)  # (n_models, n_valid)
        Y_valid = valid_label.numpy().ravel()

    if getattr(args, "tune_weights", False) and V is not None:
        def valid_mae(w):
            pred = (V * w[:, np.newaxis]).sum(axis=0)
            return np.abs(pred - Y_valid).mean()

        cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bnds = [(0, 1)] * len(models)
        res = minimize(
            valid_mae,
            x0=weights,
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 200},
        )
        if res.success:
            weights = res.x
            print(f"\n[验证集优化权重]  valid MAE = {res.fun:.4f}")
            print(f"  权重 w = {weights.round(4).tolist()}")
        else:
            print(f"\n[权重优化未收敛] 使用均匀权重")

    # stacked (n_models, n_test, 1) -> 与 weights (n_models,) 广播后 sum 得 (n_test,)，再 (n_test, 1) 给 metrics
    st = stacked.numpy().squeeze(-1)  # (n_models, n_test)
    pred_avg = (st * weights[:, np.newaxis]).sum(axis=0)
    pred_avg = torch.tensor(pred_avg, dtype=stacked.dtype).unsqueeze(-1)

    if getattr(args, "calibrate", "") and V is not None:
        P_valid = (V * weights[:, np.newaxis]).sum(axis=0)  # (n_valid,)
        pred_np = pred_avg.numpy().squeeze(-1)
        cal_mode = getattr(args, "calibrate", "linear")

        if cal_mode == "linear":
            def cal_mae(x):
                return np.abs(Y_valid - (x[0] * P_valid + x[1])).mean()
            res_cal = minimize(cal_mae, x0=[1.0, 0.0], method="Powell", options={"maxfev": 500})
            if res_cal.success:
                a, b = res_cal.x[0], res_cal.x[1]
                print(f"\n[线性校准]  valid MAE = {res_cal.fun:.4f}  a={a:.4f}  b={b:.4f}")
                pred_np = np.clip(a * pred_np + b, -1.0, 1.0)

        elif cal_mode == "piecewise":
            # 分段：[-1,-0.3], [-0.3,0.3], [0.3,1]，每段 y = a*x + b
            def cal_mae(x):
                a1, b1, a2, b2, a3, b3 = x
                out = np.where(P_valid < -0.3, a1 * P_valid + b1,
                               np.where(P_valid < 0.3, a2 * P_valid + b2, a3 * P_valid + b3))
                return np.abs(Y_valid - out).mean()
            res_cal = minimize(cal_mae, x0=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], method="Powell", options={"maxfev": 800})
            if res_cal.success:
                a1, b1, a2, b2, a3, b3 = res_cal.x
                print(f"\n[分段线性校准]  valid MAE = {res_cal.fun:.4f}")
                pred_np = np.where(pred_np < -0.3, a1 * pred_np + b1,
                                   np.where(pred_np < 0.3, a2 * pred_np + b2, a3 * pred_np + b3))
                pred_np = np.clip(pred_np, -1.0, 1.0)

        elif cal_mode == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(P_valid, Y_valid)
                pred_valid_cal = np.clip(ir.predict(P_valid), -1.0, 1.0)
                mae_valid = np.abs(pred_valid_cal - Y_valid).mean()
                pred_np = np.clip(ir.predict(pred_np), -1.0, 1.0)
                print(f"\n[Isotonic 校准]  valid MAE(校准后) = {mae_valid:.4f}")
            except ImportError:
                print("\n[Isotonic] 需要 sklearn，pip install scikit-learn；回退到线性")
                def cal_mae(x):
                    return np.abs(Y_valid - (x[0] * P_valid + x[1])).mean()
                res_cal = minimize(cal_mae, x0=[1.0, 0.0], method="Powell", options={"maxfev": 500})
                if res_cal.success:
                    pred_np = np.clip(res_cal.x[0] * pred_np + res_cal.x[1], -1.0, 1.0)

        pred_avg = torch.tensor(pred_np, dtype=pred_avg.dtype).unsqueeze(-1)

    results = metrics_fn(pred_avg, label)
    print("\n" + "=" * 60)
    title = "集成结果 (测试集)"
    if getattr(args, "tune_weights", False):
        title += ", 验证集优化权重"
    cal_mode = getattr(args, "calibrate", "")
    if cal_mode:
        title += f" + {cal_mode}校准"
    elif not getattr(args, "tune_weights", False):
        title += ", 预测取平均"
    print(title)
    print("=" * 60)
    print(f"  MAE:   {results.get('MAE', 0):.4f}")
    print(f"  Corr:  {results.get('Corr', 0):.4f}")
    print(f"  Acc-2: {results.get('Mult_acc_2', 0):.4f}")
    print(f"  Acc-3: {results.get('Mult_acc_3', 0):.4f}")
    print(f"  Acc-5: {results.get('Mult_acc_5', 0):.4f}")
    print(f"  F1:    {results.get('F1_score', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
