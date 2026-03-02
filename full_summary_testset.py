"""
完整实验测试集汇总：加载各配置的 best.pth 在测试集上评估
与 quick_summary_testset.py 类似，但对应 full_* 目录
"""
import os
import argparse
import torch
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from core.metric import MetricsTop


# 与 run_full_experiments.sh 中 EXPERIMENTS 对应
FULL_EXPERIMENTS = [
    ("baseline", "full_baseline"),
    ("+IEC(r=0.5)", "full_+IEC_r05"),
    ("+ICR", "full_+ICR_only"),
    ("IEC+ICR(r=0.5)", "full_IEC+ICR_full"),
    ("IEC+ICR(r=0.3)", "full_IEC+ICR_r03"),
    ("IEC+ICR(metric=KL)", "full_IEC+ICR_KL"),
]


def evaluate_ckpt(ckpt_path, device):
    """加载 checkpoint，在测试集上评估"""
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_opt = argparse.Namespace(**ckpt["opt"]) if isinstance(ckpt["opt"], dict) else ckpt["opt"]
    model = build_model(ckpt_opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_loader = MMDataLoader(ckpt_opt)
    metrics_fn = MetricsTop().getMetics(ckpt_opt.datasetName)
    all_pred, all_label = [], []

    with torch.no_grad():
        for data in data_loader["test"]:
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
            out, _, _, _, _, _, _ = model(inputs, None)
            all_pred.append(out.cpu())
            all_label.append(label.cpu())

    pred = torch.cat(all_pred)
    label = torch.cat(all_label)
    results = metrics_fn(pred, label)
    return results


def main():
    parser = argparse.ArgumentParser(description="Full Experiments Summary (test set)")
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints", help="Checkpoint root dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("")
    print("=" * 80)
    print("Full Experiments Summary (test set, 50 epochs)")
    print("=" * 80)
    print(f"{'Config':<25} {'MAE':>12} {'Corr':>10} {'Mult_acc_2':>12} {'F1_score':>10}")
    print("-" * 80)

    for display_name, ckpt_subdir in FULL_EXPERIMENTS:
        ckpt_path = os.path.join(args.ckpt_root, ckpt_subdir, "best.pth")
        try:
            results = evaluate_ckpt(ckpt_path, device)
            if results is None:
                print(f"{display_name:<25} (no checkpoint: {ckpt_path})")
                continue

            def _f(v):
                if torch.is_tensor(v):
                    return v.item()
                return float(v) if hasattr(v, "__float__") else v

            mae = _f(results.get("MAE", 0.0))
            corr = _f(results.get("Corr", 0.0))
            acc2 = _f(results.get("Mult_acc_2", results.get("Has0_acc_2", 0.0)))
            f1 = _f(results.get("F1_score", results.get("Has0_F1_score", 0.0)))
            print(f"{display_name:<25} {mae:>12.6f} {corr:>10.4f} {acc2:>12.4f} {f1:>10.4f}")
        except Exception as e:
            print(f"{display_name:<25} ERROR: {e}")

    print("=" * 80)
    print("")


if __name__ == "__main__":
    main()
