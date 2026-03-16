"""
run_ensemble_grcf_deep.py
==========================
对 grcf_deep 多 seed checkpoint 做集成评估（tune_weights + calibrate linear），
报告集成后的 MAE/Corr，用于验证多 seed 集成能否进一步降低 MAE。

用法：
    # 默认使用 ./checkpoints/aggressive_mae/grcf_deep
    python scripts/run_ensemble_grcf_deep.py --gpu 0

    # 指定 checkpoint 目录
    python scripts/run_ensemble_grcf_deep.py --ckpt_dir ./checkpoints/aggressive_mae/grcf_deep --gpu 0

    # 同时跑残差区间分析
    python scripts/run_ensemble_grcf_deep.py --ckpt_dir ./checkpoints/aggressive_mae/grcf_deep --residual --gpu 0
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="./checkpoints/aggressive_mae/grcf_deep",
                    help="grcf_deep 多 seed 目录，含 seed_1111/best.pth 等")
    ap.add_argument("--dataPath", type=str, default="",
                    help="强制指定数据 pkl 路径（若 checkpoint 用 BERT pkl 训练，评估时需指定 BERT pkl）")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--residual", action="store_true",
                    help="额外运行 analyze_residual_by_interval 做残差区间分析")
    args = ap.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        print(f"[错误] 目录不存在: {ckpt_dir}")
        print("请先运行 run_aggressive_mae.py 完成 grcf_deep 训练")
        sys.exit(1)

    # 检查是否有多个 seed
    subdirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    best_files = [os.path.join(ckpt_dir, d, "best.pth") for d in subdirs]
    best_files = [p for p in best_files if os.path.isfile(p)]
    if len(best_files) < 2:
        print(f"[提示] 仅找到 {len(best_files)} 个 checkpoint，建议至少 2 个 seed 做集成")
    print(f"\n[grcf_deep 集成] ckpt_dir={ckpt_dir}, 共 {len(best_files)} 个模型")
    for p in best_files:
        print(f"  - {p}")

    # 1. eval_ensemble: tune_weights + calibrate linear
    print("\n" + "=" * 60)
    print("1. 集成评估 (tune_weights + calibrate linear)")
    print("=" * 60)
    cmd = [
        sys.executable, "scripts/eval_ensemble.py",
        "--ckpt_dir", ckpt_dir,
        "--tune_weights",
        "--calibrate", "linear",
        "--gpu", str(args.gpu),
    ]
    if getattr(args, "dataPath", ""):
        cmd.extend(["--dataPath", args.dataPath])
    rc = subprocess.run(cmd, cwd=ROOT)
    if rc.returncode != 0:
        print("[错误] eval_ensemble 执行失败")
        sys.exit(1)

    # 2. 可选：残差区间分析
    if args.residual:
        print("\n" + "=" * 60)
        print("2. 残差区间分析 (test集, 集成)")
        print("=" * 60)
        cmd2 = [
            sys.executable, "scripts/analyze_residual_by_interval.py",
            "--ckpt_dir", ckpt_dir,
            "--tune_weights",
            "--calibrate", "linear",
            "--gpu", str(args.gpu),
        ]
        if getattr(args, "dataPath", ""):
            cmd2.extend(["--dataPath", args.dataPath])
        subprocess.run(cmd2, cwd=ROOT)

    print("\n[完成] 集成 MAE 见上方 eval_ensemble 输出")


if __name__ == "__main__":
    main()
