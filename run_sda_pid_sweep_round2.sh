#!/bin/bash
# ============================================================
# SDA-PID 第二轮 Sweep：多里程碑续训，每个节点输出一次结果
#
# 用法:
#   bash run_sda_pid_sweep_round2.sh                 # 默认: 60 80 100 120（含 baseline）
#   bash run_sda_pid_sweep_round2.sh 80              # 仅 80 epoch
#   bash run_sda_pid_sweep_round2.sh 60 80 100 120   # 60 -> 80 -> 100 -> 120
#   bash run_sda_pid_sweep_round2.sh summary         # 仅汇总
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep_round2
    exit 0
fi

# 解析里程碑列表（全是数字才当作 epoch，否则默认 80 100 120）
MILESTONES=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        MILESTONES+=("$arg")
    fi
done
if [ ${#MILESTONES[@]} -eq 0 ]; then
    MILESTONES=(60 80 100 120)
fi

# 6 个实验配置（含 baseline）: NAME -> checkpoint 子目录
declare -A CKPT
CKPT["r2_baseline"]="./checkpoints/sweep_round2/r2_baseline"
CKPT["r2_c6"]="./checkpoints/sweep_round2/r2_c6"
CKPT["r2_c1"]="./checkpoints/sweep_round2/r2_c1"
CKPT["r2_c1c6"]="./checkpoints/sweep_round2/r2_c1c6"
CKPT["r2_c5c6"]="./checkpoints/sweep_round2/r2_c5c6"
CKPT["r2_c1c5"]="./checkpoints/sweep_round2/r2_c1c5"

# 每个 key 对应额外超参（baseline 无额外参数）
declare -A ARGS
ARGS["r2_baseline"]=""
ARGS["r2_c6"]="--use_polarity_head True --lambda_cls 0.2 --lambda_rank 0.08 --lambda_diff 0.2 --lambda_nce_diff 0.08 --lambda_ortho 0.008 --margin 0.6"
ARGS["r2_c1"]="--hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4"
ARGS["r2_c1c6"]="--hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --use_polarity_head True --lambda_cls 0.2 --lambda_rank 0.08 --lambda_diff 0.2 --lambda_nce_diff 0.08 --lambda_ortho 0.008 --margin 0.6"
ARGS["r2_c5c6"]="--lr 3e-5 --weight_decay 5e-5 --use_polarity_head True --lambda_cls 0.2 --lambda_rank 0.08 --lambda_diff 0.2 --lambda_nce_diff 0.08 --lambda_ortho 0.008 --margin 0.6"
ARGS["r2_c1c5"]="--hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"

# GPU 分配: key -> GPU id（0-5，共 6 块）
declare -A GPU
GPU["r2_baseline"]="0"
GPU["r2_c6"]="1"
GPU["r2_c1"]="2"
GPU["r2_c1c6"]="3"
GPU["r2_c5c6"]="4"
GPU["r2_c1c5"]="5"

BASE="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"

mkdir -p logs/sda_sweep_round2
for key in "${!CKPT[@]}"; do
    mkdir -p "${CKPT[$key]}"
done

NAMES=("r2_baseline" "r2_c6" "r2_c1" "r2_c1c6" "r2_c5c6" "r2_c1c5")

echo "里程碑列表: ${MILESTONES[*]}"
echo ""

PREV_EPOCH=0
for MILESTONE in "${MILESTONES[@]}"; do
    echo "============================================================"
    if [ $PREV_EPOCH -eq 0 ]; then
        echo " Phase: 0 → ${MILESTONE} epoch（首次训练）"
    else
        echo " Phase: ${PREV_EPOCH} → ${MILESTONE} epoch（续训）"
    fi
    echo "============================================================"

    for key in "${NAMES[@]}"; do
        CKPT_DIR="${CKPT[$key]}"
        EXTRA="${ARGS[$key]}"
        GPUID="${GPU[$key]}"
        BEST_PTH="${CKPT_DIR}/best.pth"
        LOG="logs/sda_sweep_round2/${key}_ep${MILESTONE}.log"

        if [ $PREV_EPOCH -gt 0 ] && [ -f "$BEST_PTH" ]; then
            # 续训：--resume 从最佳 checkpoint 恢复，只需传 n_epochs 和 checkpoint_dir
            CMD="--resume ${BEST_PTH} --n_epochs ${MILESTONE} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_r2"
        else
            # 首次训练：完整参数（baseline 时 EXTRA 为空）
            CMD="$BASE --n_epochs ${MILESTONE} ${EXTRA} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_r2"
        fi

        echo "  [GPU ${GPUID}] ${key} -> ${LOG}"
        CUDA_VISIBLE_DEVICES=$GPUID conda run --no-capture-output -n $CONDA_ENV \
            python -u train.py $CMD > "$LOG" 2>&1 &
    done

    wait

    echo ""
    echo "========== ${MILESTONE} epoch 汇总结果 =========="
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep_round2
    echo ""

    PREV_EPOCH=$MILESTONE
done

echo "所有里程碑训练完成。"
