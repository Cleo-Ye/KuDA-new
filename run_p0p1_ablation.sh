#!/bin/bash
# ============================================================
# P0+P1 消融实验: 双通路融合 + 冲突校准
#
# 实验设计 (8个GPU并行):
#   Exp1: baseline - 无ConflictJS (原始融合)
#   Exp2: ConflictJS + 双通路, 但 lambda_con=0, lambda_cal=0 (无校准/一致性损失)
#   Exp3: ConflictJS + 双通路, lambda_con=0.05, lambda_cal=0.05
#   Exp4: ConflictJS + 双通路, lambda_con=0.1,  lambda_cal=0.1  (默认)
#   Exp5: ConflictJS + 双通路, lambda_con=0.2,  lambda_cal=0.2
#   Exp6: ConflictJS + 双通路, lambda_con=0.1,  lambda_cal=0    (只有一致性, 无校准)
#   Exp7: ConflictJS + 双通路, lambda_con=0,    lambda_cal=0.1  (只有校准, 无一致性)
#   Exp8: ConflictJS + 双通路 + Vision剪枝, lambda_con=0.1, lambda_cal=0.1
#
# 用法:
#   训练:     bash run_p0p1_ablation.sh train
#   可视化:   bash run_p0p1_ablation.sh viz
#   评估:     bash run_p0p1_ablation.sh eval
#   全部:     bash run_p0p1_ablation.sh all
# ============================================================

CONDA_ENV="kuda"
BASE_ARGS="--datasetName SIMS --use_routing True --lambda_nce 0.1 --lambda_senti 0.05 --lambda_js 0.1 --n_epochs 50"

mkdir -p logs/p0p1

# ---- 实验配置表 ----
# 格式: NAME|GPU|EXTRA_ARGS
EXPERIMENTS=(
    "baseline|0|--use_conflict_js False --use_vision_pruning False --lambda_con 0 --lambda_cal 0"
    "dual_no_loss|1|--use_conflict_js True --use_vision_pruning False --lambda_con 0 --lambda_cal 0"
    "dual_lam005|2|--use_conflict_js True --use_vision_pruning False --lambda_con 0.05 --lambda_cal 0.05"
    "dual_lam010|3|--use_conflict_js True --use_vision_pruning False --lambda_con 0.1 --lambda_cal 0.1"
    "dual_lam020|4|--use_conflict_js True --use_vision_pruning False --lambda_con 0.2 --lambda_cal 0.2"
    "dual_con_only|5|--use_conflict_js True --use_vision_pruning False --lambda_con 0.1 --lambda_cal 0"
    "dual_cal_only|6|--use_conflict_js True --use_vision_pruning False --lambda_con 0 --lambda_cal 0.1"
    "dual_prune_lam010|7|--use_conflict_js True --use_vision_pruning True --vision_target_ratio 0.8 --lambda_con 0.1 --lambda_cal 0.1"
)

run_train() {
    echo "============================================================"
    echo "Starting P0+P1 ablation training (${#EXPERIMENTS[@]} experiments)"
    echo "============================================================"

    PIDS=()
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME GPU EXTRA <<< "$exp"
        CKPT_DIR="./checkpoints/p0p1_${NAME}"
        LOG_FILE="logs/p0p1/train_${NAME}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [Train] ${NAME} (GPU ${GPU}) ==="
        echo "  Args: ${EXTRA}"
        echo "  Checkpoint: ${CKPT_DIR}"
        echo "  Log: ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n $CONDA_ENV python -u train.py \
            $BASE_ARGS $EXTRA \
            --checkpoint_dir "$CKPT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "All ${#EXPERIMENTS[@]} experiments launched."
    echo "PIDs: ${PIDS[*]}"
    echo "Monitor: tail -f logs/p0p1/train_<name>.log"
    echo "Kill all: kill ${PIDS[*]}"
    echo ""
    echo "Waiting for all to finish..."
    wait
    echo "All training finished!"
}

run_viz() {
    echo "============================================================"
    echo "Running visualizations for all experiments..."
    echo "============================================================"

    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME GPU EXTRA <<< "$exp"
        CKPT="./checkpoints/p0p1_${NAME}/best.pth"
        VIZ_DIR="./results/p0p1_viz_${NAME}"

        if [ -f "$CKPT" ]; then
            echo "=== [Viz] ${NAME} ==="
            CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n $CONDA_ENV python -u visualize_results.py \
                --datasetName SIMS \
                --checkpoint_path "$CKPT" \
                --save_dir "$VIZ_DIR" \
                > "logs/p0p1/viz_${NAME}.log" 2>&1
            echo "  Done -> ${VIZ_DIR}"
        else
            echo "[SKIP] ${NAME}: checkpoint not found at ${CKPT}"
        fi
    done

    echo ""
    echo "All visualizations complete!"
}

run_eval() {
    echo "============================================================"
    echo "Evaluating all experiments on test set..."
    echo "============================================================"

    # 收集所有存在的checkpoint路径 (best.pth=MAE选择, best_corr.pth=Corr选择)
    CKPT_ARGS=""
    LABEL_ARGS=""
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME GPU EXTRA <<< "$exp"
        # best.pth (by MAE)
        CKPT="./checkpoints/p0p1_${NAME}/best.pth"
        if [ -f "$CKPT" ]; then
            CKPT_ARGS="${CKPT_ARGS} ${CKPT}"
            LABEL_ARGS="${LABEL_ARGS} ${NAME}"
        else
            echo "[SKIP] ${NAME}/best.pth: not found"
        fi
        # best_corr.pth (by Corr)
        CKPT_CORR="./checkpoints/p0p1_${NAME}/best_corr.pth"
        if [ -f "$CKPT_CORR" ]; then
            CKPT_ARGS="${CKPT_ARGS} ${CKPT_CORR}"
            LABEL_ARGS="${LABEL_ARGS} ${NAME}_corr"
        fi
    done

    if [ -z "$CKPT_ARGS" ]; then
        echo "No checkpoints found. Run training first."
        return 1
    fi

    echo "Evaluating checkpoints: ${LABEL_ARGS}"
    conda run --no-capture-output -n $CONDA_ENV python -u eval_valid_ablation.py \
        --checkpoints $CKPT_ARGS \
        --labels $LABEL_ARGS \
        --split test \
        --output_json ./results/p0p1_eval_results.json \
        | tee logs/p0p1/eval_results.log

    echo ""
    echo "Results saved to ./results/p0p1_eval_results.json"
    echo "Full log: logs/p0p1/eval_results.log"
}

# ---- Main ----
case "${1:-all}" in
    train)
        run_train
        ;;
    viz)
        run_viz
        ;;
    eval)
        run_eval
        ;;
    all)
        run_train
        run_viz
        run_eval
        ;;
    *)
        echo "Usage: bash run_p0p1_ablation.sh [train|viz|eval|all]"
        exit 1
        ;;
esac
