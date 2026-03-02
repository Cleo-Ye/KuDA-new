#!/bin/bash
# ============================================================
# 完整实验：50 epoch，6 组配置，多 GPU 并行
#
# 🔧 Step 1 + Step 2 改进版本
#   Step 1: VisionTokenPruner 权重归一化 (已完成)
#   Step 2: ConflictJS 参数优化 (gate_k=8.0, gate_tau=0.12, lambda_js=0.12)
#
# 配置（与 quick compare 一致，但跑完整 epoch）:
#   1. baseline           2. +IEC(r=0.5)    3. +ICR
#   4. IEC+ICR(r=0.5)     5. IEC+ICR(r=0.3) 6. IEC+ICR(metric=KL)
#
# 用法:
#   bash run_full_experiments.sh              # 默认 50 epoch, GPU 2,3,4,5,6,7 并行
#   bash run_full_experiments.sh 40           # 40 epoch
#   bash run_full_experiments.sh 50 2,3,4,5,6,7  # 指定 GPU
#   bash run_full_experiments.sh summary      # 仅打印测试集汇总（不训练）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

# 参数解析: [epochs] [gpu_ids] 或 [train|summary]
# 默认: 50 epoch（完整实验）, 后六块 GPU 2,3,4,5,6,7
if [ "$1" = "summary" ] || [ "$1" = "train" ]; then
    CMD="$1"
    N_EPOCHS=50
    GPU_IDS="2,3,4,5,6,7"
else
    N_EPOCHS="${1:-50}"
    GPU_IDS="${2:-2,3,4,5,6,7}"
    CMD="${3:-train}"
fi

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}

# 基础参数 - 使用 Step 1 + Step 2 改进后的默认值
# Step 1: VisionTokenPruner 归一化 (在代码中)
# Step 2: gate_k=8.0, gate_tau=0.12, lambda_js=0.12 (在 opts.py 中)
BASE_ARGS="--datasetName sims --use_cmvn True --use_ki False --n_epochs $N_EPOCHS"
BASE_ARGS="$BASE_ARGS --lambda_nce 0.1 --lambda_senti 0.05 --lambda_con 0.1 --lambda_cal 0.1"
# 注意: gate_k, gate_tau, lambda_js 现在使用 opts.py 中的新默认值

mkdir -p logs/full_multi

# 实验配置: NAME|EXTRA_ARGS
# 注意：全部使用原始参数，确保稳定性
EXPERIMENTS=(
    "baseline|--use_conflict_js False --use_vision_pruning False"
    "+IEC_r05|--use_conflict_js False --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
    "+ICR_only|--use_conflict_js True --use_vision_pruning False"
    "IEC+ICR_full|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
    "IEC+ICR_r03|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.3"
    "IEC+ICR_KL|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5 --conflict_metric kl"
)

run_train() {
    echo "============================================================"
    echo "Full Multi-GPU Experiments: $N_EPOCHS epochs, GPUs: ${GPU_IDS}"
    echo "使用原始稳定参数（与快速实验一致）"
    echo "============================================================"

    PIDS=()
    i=0
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME EXTRA <<< "$exp"
        GPU_ID=${GPUS[$((i % N_GPUS))]}
        CKPT_DIR="./checkpoints/full_${NAME}"
        LOG_FILE="logs/full_multi/train_${NAME}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [Train] ${NAME} (GPU ${GPU_ID}) ==="
        echo "  Checkpoint: ${CKPT_DIR}"
        echo "  Log: ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n $CONDA_ENV python -u train.py \
            $BASE_ARGS $EXTRA \
            --checkpoint_dir "$CKPT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        ((i++))
    done

    echo ""
    echo "All ${#EXPERIMENTS[@]} experiments launched in background."
    echo "PIDs: ${PIDS[*]}"
    echo ""
    echo "实时监控:"
    echo "  tail -f logs/full_multi/train_baseline.log"
    echo "  tail -f logs/full_multi/train_IEC+ICR_full.log"
    echo ""
    echo "查看所有进度:"
    echo "  watch -n 10 'grep \"Best-MAE\" logs/full_multi/*.log | tail -12'"
    echo ""
    echo "终止所有:"
    echo "  kill ${PIDS[*]}"
    echo ""
    echo "等待所有实验完成..."
    wait
    echo ""
    echo "============================================================"
    echo "All training finished!"
    echo "============================================================"
    # 训练结束后在测试集上评估并打印汇总
    echo ""
    if command -v conda &>/dev/null; then
        conda run --no-capture-output -n $CONDA_ENV python -u full_summary_testset.py --ckpt_root ./checkpoints
    else
        python -u full_summary_testset.py --ckpt_root ./checkpoints
    fi
}

run_summary() {
    echo ""
    echo "============================================================"
    echo "Full Experiments Summary (best valid MAE, from logs)"
    echo "============================================================"
    printf "%-20s %10s %10s %10s\n" "Config" "MAE" "Corr" "Epoch"
    echo "------------------------------------------------------------"
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME EXTRA <<< "$exp"
        LOG="logs/full_multi/train_${NAME}.log"
        if [ -f "$LOG" ]; then
            LINE=$(grep "Best-MAE model saved" "$LOG" | tail -1)
            if [ -n "$LINE" ]; then
                MAE=$(echo "$LINE" | sed -n 's/.*MAE=\([0-9.]*\).*/\1/p')
                CORR=$(echo "$LINE" | sed -n 's/.*Corr=\([0-9.]*\).*/\1/p')
                EPOCH=$(echo "$LINE" | sed -n 's/.*epoch \([0-9]*\).*/\1/p')
                printf "%-20s %10s %10s %10s\n" "$NAME" "${MAE:-N/A}" "${CORR:-N/A}" "${EPOCH:-N/A}"
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
        conda run --no-capture-output -n $CONDA_ENV python -u full_summary_testset.py --ckpt_root ./checkpoints
    else
        python -u full_summary_testset.py --ckpt_root ./checkpoints
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
        echo "Usage: bash run_full_experiments.sh [epochs] [gpu_ids] [train|summary]"
        echo "  epochs: 默认 50（完整实验），可调整为 30/40 等"
        echo "  gpu_ids: 默认 2,3,4,5,6,7（后六块 GPU，避免占用 0、1 号）"
        echo ""
        echo "Examples:"
        echo "  # 完整 50 epoch 实验"
        echo "  bash run_full_experiments.sh"
        echo ""
        echo "  # 40 epoch（缩短时间）"
        echo "  bash run_full_experiments.sh 40"
        echo ""
        echo "  # 指定其他 GPU"
        echo "  bash run_full_experiments.sh 50 3,4,5,6,7,8"
        echo ""
        echo "  # 仅打印测试集汇总（不训练）"
        echo "  bash run_full_experiments.sh summary"
        echo ""
        echo "  # 训练后查看进度"
        echo "  watch -n 10 'grep \"Best-MAE\" logs/full_multi/*.log | tail -12'"
        exit 1
        ;;
esac
