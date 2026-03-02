#!/bin/bash
# ============================================================
# 多 GPU 并行快速对比: 5-10 epoch 跑多种配置，后几块 GPU 并行（避免占用 0 号）
#
# 配置 (与 Quick Compare Summary 一致):
#   1. baseline           2. +IEC(r=0.5)    3. +ICR
#   4. IEC+ICR(r=0.5)     5. IEC+ICR(r=0.3) 6. IEC+ICR(metric=KL)
#
# 用法:
#   bash run_quick_multi_gpu.sh              # 默认 8 epoch, GPU 2,3,4,5,6,7 并行
#   bash run_quick_multi_gpu.sh 5            # 5 epoch
#   bash run_quick_multi_gpu.sh 10           # 10 epoch
#   bash run_quick_multi_gpu.sh 8 2,3,4,5,6,7  # 指定 GPU
#   bash run_quick_multi_gpu.sh summary       # 仅打印测试集汇总（不训练）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

# 参数解析: [epochs] [gpu_ids] 或 [train|summary]
# 默认: 8 epoch, 后六块 GPU 2,3,4,5,6,7（共 8 块时避免占用 0、1 号）
if [ "$1" = "summary" ] || [ "$1" = "train" ]; then
    CMD="$1"
    N_EPOCHS=8
    GPU_IDS="2,3,4,5,6,7"
else
    N_EPOCHS="${1:-8}"
    GPU_IDS="${2:-2,3,4,5,6,7}"
    CMD="${3:-train}"   # 只传 epochs 或 epochs+gpu 时默认执行 train
fi

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}

BASE_ARGS="--datasetName sims --use_cmvn True --use_ki False --n_epochs $N_EPOCHS"
BASE_ARGS="$BASE_ARGS --lambda_nce 0.1 --lambda_senti 0.05 --lambda_js 0.1 --lambda_con 0.1 --lambda_cal 0.1"

mkdir -p logs/quick_multi

# 实验配置: NAME|EXTRA_ARGS (NAME 对应 quick_${NAME} 目录，汇总表显示名在 quick_summary_testset.py)
EXPERIMENTS=(
    "baseline|--use_conflict_js False --use_vision_pruning False"
    "+IEC_only|--use_conflict_js False --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
    "+ICR_only|--use_conflict_js True --use_vision_pruning False"
    "IEC+ICR_full|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
    "IEC+ICR_r03|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.3"
    "IEC+ICR_KL|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5 --conflict_metric kl"
)

run_train() {
    echo "============================================================"
    echo "Quick Multi-GPU Compare: $N_EPOCHS epochs, GPUs: ${GPU_IDS}"
    echo "============================================================"

    PIDS=()
    i=0
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME EXTRA <<< "$exp"
        GPU_ID=${GPUS[$((i % N_GPUS))]}
        CKPT_DIR="./checkpoints/quick_${NAME}"
        LOG_FILE="logs/quick_multi/train_${NAME}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [Train] ${NAME} (GPU ${GPU_ID}) ==="
        echo "  Args: ${EXTRA}"
        echo "  Log: ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n $CONDA_ENV python -u train.py \
            $BASE_ARGS $EXTRA \
            --checkpoint_dir "$CKPT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        ((i++))
    done

    echo ""
    echo "All ${#EXPERIMENTS[@]} experiments launched."
    echo "PIDs: ${PIDS[*]}"
    echo "Monitor: tail -f logs/quick_multi/train_<name>.log"
    echo "Kill all: kill ${PIDS[*]}"
    echo ""
    echo "Waiting for all to finish..."
    wait
    echo "All training finished!"
    # 训练结束后在测试集上评估并打印 Quick Compare Summary (test set)
    echo ""
    if command -v conda &>/dev/null; then
        conda run --no-capture-output -n $CONDA_ENV python -u quick_summary_testset.py --ckpt_root ./checkpoints
    else
        python -u quick_summary_testset.py --ckpt_root ./checkpoints
    fi
}

run_summary() {
    echo ""
    echo "============================================================"
    echo "Quick Compare Summary (best valid MAE, from logs)"
    echo "============================================================"
    printf "%-20s %10s %10s\n" "Config" "MAE" "Corr"
    echo "------------------------------------------------------------"
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME EXTRA <<< "$exp"
        LOG="logs/quick_multi/train_${NAME}.log"
        if [ -f "$LOG" ]; then
            LINE=$(grep "Best-MAE model saved" "$LOG" | tail -1)
            if [ -n "$LINE" ]; then
                MAE=$(echo "$LINE" | sed -n 's/.*MAE=\([0-9.]*\).*/\1/p')
                CORR=$(echo "$LINE" | sed -n 's/.*Corr=\([0-9.]*\).*/\1/p')
                printf "%-20s %10s %10s\n" "$NAME" "${MAE:-N/A}" "${CORR:-N/A}"
            else
                printf "%-20s %10s\n" "$NAME" "no best"
            fi
        else
            printf "%-20s %10s\n" "$NAME" "no log"
        fi
    done
    echo "============================================================"
    echo ""
    echo "Test set summary (load best.pth and evaluate on test):"
    if command -v conda &>/dev/null; then
        conda run --no-capture-output -n $CONDA_ENV python -u quick_summary_testset.py --ckpt_root ./checkpoints
    else
        python -u quick_summary_testset.py --ckpt_root ./checkpoints
    fi
}

case "$CMD" in
    train)
        run_train
        ;;
    summary)
        run_summary
        ;;
    *)
        echo "Usage: bash run_quick_multi_gpu.sh [epochs] [gpu_ids] [train|summary]"
        echo "  epochs: 5-10 (default 8), 仅快速看效果用 5-10 即可"
        echo "  gpu_ids: 默认 2,3,4,5,6,7（后六块 GPU，避免占用 0、1 号）"
        echo "  Example:"
        echo "    bash run_quick_multi_gpu.sh           # 8 epoch, GPU 2,3,4,5,6,7 并行"
        echo "    bash run_quick_multi_gpu.sh 5        # 5 epoch"
        echo "    bash run_quick_multi_gpu.sh 10 2,3,4,5,6,7"
        echo "    bash run_quick_multi_gpu.sh summary  # 仅打印测试集汇总"
        exit 1
        ;;
esac
