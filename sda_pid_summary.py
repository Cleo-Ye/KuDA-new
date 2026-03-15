"""
SDA-PID sweep 结果汇总：加载每个配置的 best.pth，在测试集上评估，打印对比表。

用法:
  python sda_pid_summary.py                      # 默认 ./checkpoints/sweep
  python sda_pid_summary.py --ckpt_root ./checkpoints/sweep
  python sda_pid_summary.py --split valid        # 看 valid 集结果
"""
import os
import argparse
import torch
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from core.metric import MetricsTop, sims_binary_metrics_from_logits


# 第一轮 sweep 的 8 个配置
SWEEP_CONFIGS = [
    ("C0 Polarity head",     "c0_polarity",    "--use_polarity_head True --lambda_cls 0.2"),
    ("C1 Bigger model",      "c1_big",         "--hidden_size 512 --ffn_size 1024 --lr 3e-5"),
    ("C2 More epochs(100)",  "c2_ep100",       "--n_epochs 100"),
    ("C3 Stronger NCE",      "c3_nce",         "--lambda_diff 0.25 --lambda_nce_diff 0.15"),
    ("C4 S diversity",       "c4_sdiv",        "--lambda_S_diverse 0.25 --lambda_pid 0.4"),
    ("C5 Low LR (3e-5)",     "c5_lowlr",       "--lr 3e-5 --weight_decay 5e-5"),
    ("C6 Polarity+NCE",      "c6_polar_nce",   "--use_polarity_head True + stronger NCE"),
    ("C7 Seed 2222",         "c7_seed2222",    "--seed 2222"),
]

# 第二轮 sweep（baseline + 优配 + 组合），配合 --ckpt_root ./checkpoints/sweep_round2
ROUND2_CONFIGS = [
    ("R2-0 Baseline",        "r2_baseline",    "默认参数"),
    ("R2-1 C6 复现",         "r2_c6",          "Polarity + NCE"),
    ("R2-2 C1 复现",         "r2_c1",          "Bigger model"),
    ("R2-3 C1+C6",           "r2_c1c6",        "Bigger + Polarity + NCE"),
    ("R2-4 C5+C6",           "r2_c5c6",        "Low LR + Polarity + NCE"),
    ("R2-5 C1+C5",           "r2_c1c5",        "Bigger + Low LR"),
]

# F1 整体优化：以 C1+C5 为 baseline，在其上做 F1 增强；配合 --ckpt_root ./checkpoints/sweep_f1
F1_OPT_CONFIGS = [
    ("F1-0 Baseline (C1+C5)", "f1_baseline",   "C1+C5 同参"),
    ("F1-1 λ_cls=0.35",       "f1_l035",       "C1+C5 + lambda_classification 0.35"),
    ("F1-2 λ_cls=0.4",        "f1_l04",        "C1+C5 + lambda_classification 0.4"),
    ("F1-3 λ_cls=0.35+pw",    "f1_l035_pw",    "C1+C5 + λ_cls 0.35 + cls_pos_weight 1.2"),
]

# 无正交主方案 + 消融（main_no_ortho_suite）
MAIN_NO_ORTHO_CONFIGS = [
    ("Main λ_ortho=0 R1", "main_no_ortho_r1", "λ_ortho=0 主方案 第1遍"),
    ("Main λ_ortho=0 R2", "main_no_ortho_r2", "λ_ortho=0 主方案 第2遍"),
    ("Main λ_ortho=0 R3", "main_no_ortho_r3", "λ_ortho=0 主方案 第3遍"),
    ("Main λ_ortho=0 R4", "main_no_ortho_r4", "λ_ortho=0 主方案 第4遍"),
    ("Ablate w/o PID",    "ablate_no_pid",    "λ_ortho=0 + w/o PID Routing"),
    ("Ablate w/o Contr.", "ablate_no_contrast", "λ_ortho=0 + λ_nce_diff=0"),
    ("Ablate w/o DualBr.", "ablate_no_dual_branch", "λ_ortho=0 + F_fusion=F_cons"),
]

# 消融实验：4 个核心消融；配合 --ckpt_root ./checkpoints/ablation_study
ABLATION_CONFIGS = [
    ("w/o PID Routing",   "wo_pid_routing",   "S 固定 0.5，F_fusion=0.5*F_cons+0.5*F_conf"),
    ("w/o Contrastive",   "wo_contrastive",   "λ_nce_diff=0"),
    ("w/o Orthogonal",    "wo_ortho",         "λ_ortho=0"),
    ("w/o Dual-Branch",   "wo_dual_branch",   "F_fusion=F_cons only"),
]

# 消融 8 GPU 模式：4 消融 × 80ep/120ep 共 8 条；配合 --ckpt_root ./checkpoints/ablation_study --ablation_8ep
ABLATION_8EP_CONFIGS = [
    ("w/o PID R. 80ep",   "wo_pid_routing_80",   "80 ep"),
    ("w/o PID R. 120ep",  "wo_pid_routing_120",  "120 ep"),
    ("w/o Contrast. 80ep", "wo_contrastive_80",   "80 ep"),
    ("w/o Contrast. 120ep","wo_contrastive_120",  "120 ep"),
    ("w/o Ortho 80ep",    "wo_ortho_80",         "80 ep"),
    ("w/o Ortho 120ep",   "wo_ortho_120",        "120 ep"),
    ("w/o DualBr. 80ep",  "wo_dual_branch_80",   "80 ep"),
    ("w/o DualBr. 120ep", "wo_dual_branch_120",  "120 ep"),
]

# F1 配置×2 遍：4 配置各跑 2 次，共 8 条；配合 --ckpt_root ./checkpoints/sweep_f1_2runs
F1_2RUNS_CONFIGS = [
    ("Baseline R1", "f1_baseline_r1",   "C1+C5 第1遍"),
    ("Baseline R2", "f1_baseline_r2",   "C1+C5 第2遍"),
    ("λ_cls=0.35 R1", "f1_l035_r1",     "λ_cls=0.35 第1遍"),
    ("λ_cls=0.35 R2", "f1_l035_r2",     "λ_cls=0.35 第2遍"),
    ("λ_cls=0.4 R1", "f1_l04_r1",       "λ_cls=0.4 第1遍"),
    ("λ_cls=0.4 R2", "f1_l04_r2",       "λ_cls=0.4 第2遍"),
    ("λ_cls=0.35+pw R1", "f1_l035_pw_r1", "λ_cls=0.35+pw 第1遍"),
    ("λ_cls=0.35+pw R2", "f1_l035_pw_r2", "λ_cls=0.35+pw 第2遍"),
]


def evaluate_ckpt(ckpt_path: str, device: torch.device, split: str = "test"):
    if not os.path.isfile(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_opt = (argparse.Namespace(**ckpt["opt"])
                if isinstance(ckpt["opt"], dict)
                else ckpt["opt"])
    model = build_model(ckpt_opt).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data_loader = MMDataLoader(ckpt_opt)
    metrics_fn = MetricsTop().getMetics(ckpt_opt.datasetName)
    all_pred, all_label, all_logit_cls = [], [], []

    with torch.no_grad():
        for data in data_loader[split]:
            inputs = {
                "V": data["vision"].to(device),
                "A": data["audio"].to(device),
                "T": data["text"].to(device),
                "mask": {
                    "V": data["vision_padding_mask"][:, 1:data["vision"].shape[1] + 1].to(device),
                    "A": data["audio_padding_mask"][:, 1:data["audio"].shape[1] + 1].to(device),
                    "T": [],
                },
            }
            label = data["labels"]["M"].to(device).view(-1, 1)
            out, _, _, _, _, _, logit_cls = model(inputs, None)
            all_pred.append(out.cpu())
            all_label.append(label.cpu())
            all_logit_cls.append(logit_cls.cpu())

    pred = torch.cat(all_pred)
    label = torch.cat(all_label)
    results = metrics_fn(pred, label)
    logit_cls = torch.cat(all_logit_cls)
    try:
        cls_metrics = sims_binary_metrics_from_logits(logit_cls, label)
        results.update(cls_metrics)
    except Exception:
        pass
    return results


def _fmt(v):
    if v is None:
        return "N/A"
    if torch.is_tensor(v):
        return f"{v.item():.4f}"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_root", type=str, default="./checkpoints/sweep")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    parser.add_argument("--round2", action="store_true", help="Use Round2 config list (r2_c6, r2_c1, ...)")
    parser.add_argument("--ablation_8ep", action="store_true", help="Ablation 8 GPU 模式：8 条 (4 消融×80ep/120ep)")
    args = parser.parse_args()

    if "main_no_ortho_suite" in args.ckpt_root:
        configs = MAIN_NO_ORTHO_CONFIGS
        title_suffix = " [Main-no-ortho]"
    elif "ablation" in args.ckpt_root and getattr(args, "ablation_8ep", False):
        configs = ABLATION_8EP_CONFIGS
        title_suffix = " [Ablation 80/120ep]"
    elif "ablation" in args.ckpt_root:
        configs = ABLATION_CONFIGS
        title_suffix = " [Ablation]"
    elif "sweep_f1_2runs" in args.ckpt_root:
        configs = F1_2RUNS_CONFIGS
        title_suffix = " [F1-Opt×2]"
    elif "sweep_f1" in args.ckpt_root:
        configs = F1_OPT_CONFIGS
        title_suffix = " [F1-Opt]"
    elif args.round2 or "round2" in args.ckpt_root:
        configs = ROUND2_CONFIGS
        title_suffix = " [Round2]"
    else:
        configs = SWEEP_CONFIGS
        title_suffix = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    W = 22
    print()
    print("=" * 90)
    print(f"  SDA-PID Sweep Summary  ({args.split} set){title_suffix}")
    print("=" * 90)
    header = f"{'Config':<{W}} {'MAE':>8} {'Corr':>8} {'Acc-2':>8} {'Acc-3':>8} {'Acc-5':>8} {'F1':>8} {'F1_cls':>8}"
    print(header)
    print("-" * 100)

    best_mae_row, best_corr_row = None, None
    best_mae_val, best_corr_val = float("inf"), -float("inf")
    rows = []

    for display_name, subdir, desc in configs:
        ckpt_path = os.path.join(args.ckpt_root, subdir, "best.pth")
        try:
            results = evaluate_ckpt(ckpt_path, device, args.split)
        except Exception as e:
            print(f"  {display_name:<{W-2}} ERROR: {e}")
            rows.append((display_name, None))
            continue

        if results is None:
            print(f"  {display_name:<{W-2}} (no checkpoint found: {ckpt_path})")
            rows.append((display_name, None))
            continue

        mae  = _fmt(results.get("MAE"))
        corr = _fmt(results.get("Corr"))
        acc2 = _fmt(results.get("Mult_acc_2", results.get("Has0_acc_2")))
        acc3 = _fmt(results.get("Mult_acc_3", results.get("Has0_acc_3")))
        acc5 = _fmt(results.get("Mult_acc_5"))
        f1   = _fmt(results.get("F1_score", results.get("Has0_F1_score")))
        f1_cls = _fmt(results.get("F1_score_cls"))

        row = f"  {display_name:<{W-2}} {mae:>8} {corr:>8} {acc2:>8} {acc3:>8} {acc5:>8} {f1:>8} {f1_cls:>8}"
        rows.append((display_name, row))
        print(row)

        try:
            mae_v = float(results.get("MAE", 1e9))
            corr_v = float(results.get("Corr", -1))
            if mae_v < best_mae_val:
                best_mae_val, best_mae_row = mae_v, display_name
            if corr_v > best_corr_val:
                best_corr_val, best_corr_row = corr_v, display_name
        except Exception:
            pass

    print("=" * 90)
    if best_mae_row:
        print(f"  Best MAE : {best_mae_row}  ({best_mae_val:.4f})")
    if best_corr_row:
        print(f"  Best Corr: {best_corr_row}  ({best_corr_val:.4f})")
    print()

    print("Config details:")
    for display_name, subdir, desc in configs:
        print(f"  {display_name:<{W}} {desc}")
    print()


if __name__ == "__main__":
    main()
