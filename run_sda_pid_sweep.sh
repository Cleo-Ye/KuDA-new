#!/bin/bash
# ============================================================
# SDA-PID 8-GPU 并行超参 Sweep
#
# 当前 Baseline（n_epochs=80，默认参数）:
#   MAE=0.3354  Corr=0.6573  Acc-2=0.7737  F1=0.7730
#
# 8 个实验方向（每块 GPU 一个），覆盖最有可能提升的维度:
#   C0 (GPU 0): Polarity head            -- 改善 Acc-2 / F1
#   C1 (GPU 1): Bigger model             -- 提高表示容量 (hidden=512)
#   C2 (GPU 2): More epochs (100)        -- 更长训练看是否继续收敛
#   C3 (GPU 3): Stronger contrastive     -- 更强解耦 loss
#   C4 (GPU 4): S diversity tuning       -- 更大 lambda_S_diverse
#   C5 (GPU 5): Lower LR baseline        -- lr=3e-5 保守收敛
#   C6 (GPU 6): Polarity + Strong NCE    -- 融合 C0+C3
#   C7 (GPU 7): Seed 2222                -- 稳定性验证
#
# 用法:
#   bash run_sda_pid_sweep.sh               # 完整 80 epoch 训练
#   bash run_sda_pid_sweep.sh 20            # 快速验证 20 epoch
#   bash run_sda_pid_sweep.sh summary       # 仅汇总已有结果（不训练）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
N_EPOCHS="${1:-80}"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep
    exit 0
fi

# 基础参数（所有配置共享）
BASE="--datasetName sims --use_cmvn True --n_epochs $N_EPOCHS"
BASE="$BASE --lambda_senti 0.05 --pid_warmup_epochs 2"

mkdir -p logs/sda_sweep
mkdir -p checkpoints/sweep

echo "============================================================"
echo " SDA-PID Sweep: $N_EPOCHS epochs, GPU 0-7 并行"
echo "============================================================"

# ---- C0: Polarity head -----------------------------------------------
# 边界敏感极性头 + 排序损失，主要针对 Acc-2/F1
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --use_polarity_head True \
    --lambda_cls 0.2 \
    --lambda_rank 0.08 \
    --rank_margin 0.2 \
    --checkpoint_dir ./checkpoints/sweep/c0_polarity \
    --log_path ./log/sweep_c0 \
    > logs/sda_sweep/c0_polarity.log 2>&1 &
echo "[GPU 0] C0: Polarity head  -> logs/sda_sweep/c0_polarity.log"

# ---- C1: Bigger model ------------------------------------------------
# hidden=512, ffn=1024, dropout=0.2，容量更大
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --hidden_size 512 \
    --ffn_size 1024 \
    --dropout 0.2 \
    --lr 3e-5 \
    --pid_lr 3e-4 \
    --checkpoint_dir ./checkpoints/sweep/c1_big \
    --log_path ./log/sweep_c1 \
    > logs/sda_sweep/c1_big.log 2>&1 &
echo "[GPU 1] C1: Bigger model   -> logs/sda_sweep/c1_big.log"

# ---- C2: More epochs -------------------------------------------------
# 默认参数，训练 max(N_EPOCHS, 100) epoch，检验是否仍有提升空间
C2_EPOCHS=$(( N_EPOCHS > 100 ? N_EPOCHS : 100 ))
BASE_C2="--datasetName sims --use_cmvn True --n_epochs $C2_EPOCHS"
BASE_C2="$BASE_C2 --lambda_senti 0.05 --pid_warmup_epochs 2"
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE_C2 \
    --checkpoint_dir ./checkpoints/sweep/c2_ep100 \
    --log_path ./log/sweep_c2 \
    > logs/sda_sweep/c2_ep100.log 2>&1 &
echo "[GPU 2] C2: More epochs($C2_EPOCHS)    -> logs/sda_sweep/c2_ep100.log"

# ---- C3: Stronger contrastive ----------------------------------------
# 适度增大 lambda_diff / lambda_nce_diff，推开 F_cons 和 F_conf
# 注意不能过大（早期主任务未收敛时会干扰梯度方向）
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --lambda_diff 0.2 \
    --lambda_nce_diff 0.08 \
    --lambda_ortho 0.008 \
    --margin 0.6 \
    --checkpoint_dir ./checkpoints/sweep/c3_nce \
    --log_path ./log/sweep_c3 \
    > logs/sda_sweep/c3_nce.log 2>&1 &
echo "[GPU 3] C3: Stronger NCE   -> logs/sda_sweep/c3_nce.log"

# ---- C4: S diversity tuning ------------------------------------------
# 加强 L_S_diverse 和 lambda_pid，让 S 的区分度更大、更能调制融合
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --lambda_S_var 0.05 \
    --lambda_S_diverse 0.25 \
    --lambda_pid 0.4 \
    --lambda_ortho 0.01 \
    --checkpoint_dir ./checkpoints/sweep/c4_sdiv \
    --log_path ./log/sweep_c4 \
    > logs/sda_sweep/c4_sdiv.log 2>&1 &
echo "[GPU 4] C4: S diversity    -> logs/sda_sweep/c4_sdiv.log"

# ---- C5: Lower LR baseline -------------------------------------------
# lr=3e-5，比默认 5e-5 更保守，有时能得到更低的 MAE
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --lr 3e-5 \
    --weight_decay 5e-5 \
    --checkpoint_dir ./checkpoints/sweep/c5_lowlr \
    --log_path ./log/sweep_c5 \
    > logs/sda_sweep/c5_lowlr.log 2>&1 &
echo "[GPU 5] C5: Low LR (3e-5)  -> logs/sda_sweep/c5_lowlr.log"

# ---- C6: Polarity + Moderate NCE combo -------------------------------
# 融合 C0（极性头）和 C3（适度解耦 loss），使用与 C3 相同的保守 NCE 参数
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --use_polarity_head True \
    --lambda_cls 0.2 \
    --lambda_rank 0.08 \
    --lambda_diff 0.2 \
    --lambda_nce_diff 0.08 \
    --lambda_ortho 0.008 \
    --margin 0.6 \
    --checkpoint_dir ./checkpoints/sweep/c6_polar_nce \
    --log_path ./log/sweep_c6 \
    > logs/sda_sweep/c6_polar_nce.log 2>&1 &
echo "[GPU 6] C6: Polar+NCE      -> logs/sda_sweep/c6_polar_nce.log"

# ---- C7: Seed 2222 ---------------------------------------------------
# 与 C5（默认参数）相同配置，换种子，评估稳定性
CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n $CONDA_ENV \
    python -u train.py $BASE \
    --seed 2222 \
    --checkpoint_dir ./checkpoints/sweep/c7_seed2222 \
    --log_path ./log/sweep_c7 \
    > logs/sda_sweep/c7_seed2222.log 2>&1 &
echo "[GPU 7] C7: Seed 2222      -> logs/sda_sweep/c7_seed2222.log"

echo ""
echo "所有 8 个实验已在后台启动。"
echo "实时监控示例:"
echo "  tail -f logs/sda_sweep/c0_polarity.log"
echo "  tail -f logs/sda_sweep/c2_ki.log"
echo ""
echo "等待所有实验完成..."
wait
echo ""
echo "============================================================"
echo " 所有训练完成！正在生成汇总表..."
echo "============================================================"
conda run --no-capture-output -n $CONDA_ENV \
    python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep
