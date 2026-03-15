#!/bin/bash
# ============================================================
# F1 整体优化 Sweep：以当前最佳配置（C1+C5）为 baseline，在其基础上做 F1 增强
#
# 说明：F1-0 baseline 是本脚本从 0 新训的一趟（同参 C1+C5），与 Round2 的 R2-5 是两次独立
# 训练，故 MAE/Corr 可能略不同（随机性）。若希望 F1-0 与 R2-5 完全一致，可先复制权重再汇总：
#   cp ./checkpoints/sweep_round2/r2_c1c5/best.pth ./checkpoints/sweep_f1/f1_baseline/
#   bash run_sweep_f1_optimization.sh summary
# 这样汇总表中的 F1-0 即为上一轮 R2-5 的数值。
#
# 用法:
#   bash run_sweep_f1_optimization.sh              # 默认: 60 80 100 120
#   bash run_sweep_f1_optimization.sh 80 100 120   # 指定里程碑
#   bash run_sweep_f1_optimization.sh summary      # 仅汇总
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep_f1
    exit 0
fi

# 解析里程碑
MILESTONES=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        MILESTONES+=("$arg")
    fi
done
if [ ${#MILESTONES[@]} -eq 0 ]; then
    MILESTONES=(60 80 100 120)
fi

# 新 baseline = C1+C5（与 round2 的 r2_c1c5 同参），F1 增强在其基础上加 lambda_classification / cls_pos_weight
C1C5_BASE="--hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"

declare -A CKPT
CKPT["f1_baseline"]="./checkpoints/sweep_f1/f1_baseline"
CKPT["f1_l035"]="./checkpoints/sweep_f1/f1_l035"
CKPT["f1_l04"]="./checkpoints/sweep_f1/f1_l04"
CKPT["f1_l035_pw"]="./checkpoints/sweep_f1/f1_l035_pw"

declare -A ARGS
ARGS["f1_baseline"]="$C1C5_BASE"
ARGS["f1_l035"]="$C1C5_BASE --lambda_classification 0.35"
ARGS["f1_l04"]="$C1C5_BASE --lambda_classification 0.4"
ARGS["f1_l035_pw"]="$C1C5_BASE --lambda_classification 0.35 --cls_pos_weight 1.2"

declare -A GPU
GPU["f1_baseline"]="0"
GPU["f1_l035"]="1"
GPU["f1_l04"]="2"
GPU["f1_l035_pw"]="3"

BASE="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"

mkdir -p logs/sweep_f1
for key in "${!CKPT[@]}"; do
    mkdir -p "${CKPT[$key]}"
done

NAMES=("f1_baseline" "f1_l035" "f1_l04" "f1_l035_pw")

echo "F1 优化 Sweep：baseline = C1+C5，里程碑 = ${MILESTONES[*]}"
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
        LOG="logs/sweep_f1/${key}_ep${MILESTONE}.log"

        if [ $PREV_EPOCH -gt 0 ] && [ -f "$BEST_PTH" ]; then
            CMD="--resume ${BEST_PTH} --n_epochs ${MILESTONE} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_f1"
        else
            CMD="$BASE --n_epochs ${MILESTONE} ${EXTRA} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_f1"
        fi

        echo "  [GPU ${GPUID}] ${key} -> ${LOG}"
        CUDA_VISIBLE_DEVICES=$GPUID conda run --no-capture-output -n $CONDA_ENV \
            python -u train.py $CMD > "$LOG" 2>&1 &
    done

    wait

    echo ""
    echo "========== ${MILESTONE} epoch 汇总 =========="
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root ./checkpoints/sweep_f1
    echo ""

    PREV_EPOCH=$MILESTONE
done

echo "F1 优化 Sweep 完成。"
