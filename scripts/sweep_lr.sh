#!/bin/bash
# ============================================================
# 学习率扫描：5e-5 → 1e-6，4 个跨度，GPU 0-3 并行
#
# 用法:
#   bash scripts/sweep_lr.sh              # 默认 4 个 lr，GPU 0,1,2,3
#   bash scripts/sweep_lr.sh summary       # 只打印已有实验汇总（不训练）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
SWEEP_DIR="${SWEEP_DIR:-./checkpoints/sweep_lr}"
N_EPOCHS="${N_EPOCHS:-100}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

# 4 个学习率：5e-5 → 1e-6 对数均匀
LRS=(5e-5 2e-5 5e-6 1e-6)

IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}

BASE_ARGS="--datasetName sims --model_type pid_dualpath --use_batch_pid_prior True"
BASE_ARGS="$BASE_ARGS --n_epochs $N_EPOCHS"

mkdir -p "$SWEEP_DIR"
mkdir -p logs/sweep_lr

run_train() {
    echo "============================================================"
    echo "LR Sweep: ${LRS[*]}  (4 spans: 5e-5 → 1e-6)"
    echo "GPUs: $GPU_IDS"
    echo "Checkpoint root: $SWEEP_DIR"
    echo "============================================================"
    echo ""

    PIDS=()
    for idx in "${!LRS[@]}"; do
        LR="${LRS[$idx]}"
        GPU_ID="${GPUS[$((idx % N_GPUS))]}"
        # run_name: lr5em5, lr2em5, lr5em6, lr1em6
        RUN_NAME="lr$(echo "$LR" | sed 's/e-/em/g')"
        CKPT_DIR="${SWEEP_DIR}/${RUN_NAME}"
        LOG_FILE="logs/sweep_lr/${RUN_NAME}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [lr=$LR] GPU=$GPU_ID, ckpt=$CKPT_DIR ==="

        if [ -n "$CONDA_ENV" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n "$CONDA_ENV" python -u train.py $BASE_ARGS \
            --lr "$LR" --checkpoint_dir "$CKPT_DIR" --log_path "$CKPT_DIR/" --no_save_model True > "$LOG_FILE" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python -u train.py $BASE_ARGS \
            --lr "$LR" \
            --checkpoint_dir "$CKPT_DIR" \
            --log_path "$CKPT_DIR/" \
            --no_save_model True \
            > "$LOG_FILE" 2>&1 &
        fi
        PIDS+=($!)
    done

    echo ""
    echo "All ${#LRS[@]} runs launched. PIDs: ${PIDS[*]}"
    echo "Monitor: tail -f logs/sweep_lr/lr5em5.log"
    echo ""
    echo "Waiting for all runs to complete..."
    wait

    echo ""
    echo "All runs finished."
    print_summary
}

print_summary() {
    echo ""
    echo "============================================================"
    echo "LR Sweep Results Summary"
    echo "============================================================"
    printf "%-12s %10s %10s %12s %12s\n" "LR" "Test MAE" "Test Corr" "Best Ep MAE" "Best Ep Corr"
    echo "------------------------------------------------------------"

    for LR in "${LRS[@]}"; do
        RUN_NAME="lr$(echo "$LR" | sed 's/e-/em/g')"
        SUMMARY="${SWEEP_DIR}/${RUN_NAME}/summary.json"
        if [ -f "$SUMMARY" ]; then
            MAE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); v=d.get('best_test_mae', d.get('best_valid_mae')); print(f'{v:.4f}' if v is not None else '-')" 2>/dev/null) || MAE="-"
            CORR=$(python3 -c "import json; d=json.load(open('$SUMMARY')); v=d.get('best_test_corr', d.get('best_valid_corr')); print(f'{v:.4f}' if v is not None else '-')" 2>/dev/null) || CORR="-"
            EP_MAE=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('best_epoch_mae','-'))" 2>/dev/null) || EP_MAE="-"
            EP_CORR=$(python3 -c "import json; d=json.load(open('$SUMMARY')); print(d.get('best_epoch_corr','-'))" 2>/dev/null) || EP_CORR="-"
            printf "%-12s %10s %10s %12s %12s\n" "$LR" "$MAE" "$CORR" "$EP_MAE" "$EP_CORR"
        else
            printf "%-12s %10s %10s %12s %12s\n" "$LR" "-" "-" "-" "-"
        fi
    done
    echo "============================================================"
}

if [ "$1" = "summary" ]; then
    print_summary
else
    run_train
fi
