#!/bin/bash
# ============================================================
# F1 配置 × 2 遍 Sweep：4 种配置各跑 2 次，占满 8 GPU，多里程碑，降低随机波动
#
# 用法:
#   bash run_sweep_f1_2runs.sh              # 默认: 60 80 100 120
#   bash run_sweep_f1_2runs.sh 80 100 120   # 指定里程碑
#   bash run_sweep_f1_2runs.sh summary      # 仅汇总（8 条 + 可选均值）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
CKPT_ROOT="./checkpoints/sweep_f1_2runs"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root "$CKPT_ROOT"
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

C1C5_BASE="--hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"

# 8 个任务：4 配置 × 2 遍。子目录名与 sda_pid_summary F1_2RUNS_CONFIGS 一致
NAMES=(
    "f1_baseline_r1"   "f1_baseline_r2"
    "f1_l035_r1"       "f1_l035_r2"
    "f1_l04_r1"        "f1_l04_r2"
    "f1_l035_pw_r1"    "f1_l035_pw_r2"
)

# 每个任务对应的额外参数（r1/r2 同配置用同一套参数）
ARGS_r1="$C1C5_BASE"
ARGS_r2="$C1C5_BASE"
ARGS_l035_r1="$C1C5_BASE --lambda_classification 0.35"
ARGS_l035_r2="$C1C5_BASE --lambda_classification 0.35"
ARGS_l04_r1="$C1C5_BASE --lambda_classification 0.4"
ARGS_l04_r2="$C1C5_BASE --lambda_classification 0.4"
ARGS_pw_r1="$C1C5_BASE --lambda_classification 0.35 --cls_pos_weight 1.2"
ARGS_pw_r2="$C1C5_BASE --lambda_classification 0.35 --cls_pos_weight 1.2"

BASE="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"

mkdir -p logs/sweep_f1_2runs
for name in "${NAMES[@]}"; do
    mkdir -p "${CKPT_ROOT}/${name}"
done

echo "F1 配置×2 遍 Sweep：8 GPU，里程碑 = ${MILESTONES[*]}"
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

    for i in "${!NAMES[@]}"; do
        key="${NAMES[$i]}"
        CKPT_DIR="${CKPT_ROOT}/${key}"
        BEST_PTH="${CKPT_DIR}/best.pth"
        LOG="logs/sweep_f1_2runs/${key}_ep${MILESTONE}.log"
        GPUID=$i

        case "$key" in
            f1_baseline_r1|f1_baseline_r2) EXTRA="$ARGS_r1" ;;
            f1_l035_r1|f1_l035_r2)         EXTRA="$ARGS_l035_r1" ;;
            f1_l04_r1|f1_l04_r2)           EXTRA="$ARGS_l04_r1" ;;
            f1_l035_pw_r1|f1_l035_pw_r2)   EXTRA="$ARGS_pw_r1" ;;
            *) EXTRA="$C1C5_BASE" ;;
        esac

        if [ $PREV_EPOCH -gt 0 ] && [ -f "$BEST_PTH" ]; then
            CMD="--resume ${BEST_PTH} --n_epochs ${MILESTONE} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_f1_2runs"
        else
            CMD="$BASE --n_epochs ${MILESTONE} ${EXTRA} --checkpoint_dir ${CKPT_DIR} --log_path ./log/sweep_f1_2runs"
        fi

        echo "  [GPU ${GPUID}] ${key} -> ${LOG}"
        CUDA_VISIBLE_DEVICES=$GPUID conda run --no-capture-output -n $CONDA_ENV \
            python -u train.py $CMD > "$LOG" 2>&1 &
    done

    wait

    echo ""
    echo "========== ${MILESTONE} epoch 汇总 =========="
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root "$CKPT_ROOT"
    echo ""

    PREV_EPOCH=$MILESTONE
done

echo "F1 配置×2 遍 Sweep 完成。"
