#!/bin/bash
# ============================================================
# 并行消融实验: 不同Vision剪枝配置
# 
# 用法:
#   训练:     bash run_ablation.sh train
#   可视化:   bash run_ablation.sh viz
#   全部:     bash run_ablation.sh all
#
# 每个实验独立checkpoint目录+可视化目录, 可同时在同一GPU上跑
# 如果GPU显存不够同时跑4个, 可以只选2个:
#   bash run_ablation.sh train 1 3   (只跑实验1和3)
# ============================================================

CONDA_ENV="kuda"
COMMON_ARGS="--datasetName SIMS --use_conflict_js True --use_routing True --lambda_nce 0.1 --lambda_senti 0.05 --lambda_js 0.1"

mkdir -p logs

run_train() {
    # ---- 实验1: 不剪枝 (baseline) [GPU 4] ----
    echo "=== [Train] Exp1: No Pruning (GPU 4) ==="
    CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n $CONDA_ENV python -u train.py \
        $COMMON_ARGS \
        --use_vision_pruning False \
        --checkpoint_dir ./checkpoints/SIMS_no_pruning \
        > logs/exp1_no_pruning.log 2>&1 &
    PID1=$!
    echo "  PID=$PID1, log=logs/exp1_no_pruning.log"

    # ---- 实验2: 轻度剪枝 (保留80%) [GPU 5] ----
    echo "=== [Train] Exp2: Light Pruning (ratio=0.8) (GPU 5) ==="
    CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n $CONDA_ENV python -u train.py \
        $COMMON_ARGS \
        --use_vision_pruning True \
        --vision_target_ratio 0.8 \
        --checkpoint_dir ./checkpoints/SIMS_prune_80 \
        > logs/exp2_prune_80.log 2>&1 &
    PID2=$!
    echo "  PID=$PID2, log=logs/exp2_prune_80.log"

    # ---- 实验3: 中度剪枝 (保留60%) [GPU 6] ----
    echo "=== [Train] Exp3: Medium Pruning (ratio=0.6) (GPU 6) ==="
    CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n $CONDA_ENV python -u train.py \
        $COMMON_ARGS \
        --use_vision_pruning True \
        --vision_target_ratio 0.6 \
        --checkpoint_dir ./checkpoints/SIMS_prune_60 \
        > logs/exp3_prune_60.log 2>&1 &
    PID3=$!
    echo "  PID=$PID3, log=logs/exp3_prune_60.log"

    # ---- 实验4: 重度剪枝 (保留40%) [GPU 7] ----
    echo "=== [Train] Exp4: Heavy Pruning (ratio=0.4) (GPU 7) ==="
    CUDA_VISIBLE_DEVICES=7 conda run --no-capture-output -n $CONDA_ENV python -u train.py \
        $COMMON_ARGS \
        --use_vision_pruning True \
        --vision_target_ratio 0.4 \
        --checkpoint_dir ./checkpoints/SIMS_prune_40 \
        > logs/exp4_prune_40.log 2>&1 &
    PID4=$!
    echo "  PID=$PID4, log=logs/exp4_prune_40.log"

    echo ""
    echo "All 4 training experiments launched in background."
    echo "Monitor progress:  tail -f logs/exp1_no_pruning.log"
    echo "Check GPU usage:   watch -n 2 nvidia-smi"
    echo "PIDs: $PID1 $PID2 $PID3 $PID4"
    echo "Kill all: kill $PID1 $PID2 $PID3 $PID4"
    echo ""
    echo "Waiting for all to finish..."
    wait
    echo "All training experiments finished!"
}

run_viz() {
    echo ""
    echo "============================================================"
    echo "Running visualizations for all experiments..."
    echo "============================================================"

    # Exp1: No pruning
    if [ -f ./checkpoints/SIMS_no_pruning/best.pth ]; then
        echo "=== [Viz] Exp1: No Pruning ==="
        conda run --no-capture-output -n $CONDA_ENV python -u visualize_results.py \
            --datasetName SIMS \
            --checkpoint_path ./checkpoints/SIMS_no_pruning/best.pth \
            --save_dir ./results/viz_no_pruning
    else
        echo "[SKIP] Exp1: checkpoint not found"
    fi

    # Exp2: ratio=0.8
    if [ -f ./checkpoints/SIMS_prune_80/best.pth ]; then
        echo "=== [Viz] Exp2: Prune 80% ==="
        conda run --no-capture-output -n $CONDA_ENV python -u visualize_results.py \
            --datasetName SIMS \
            --checkpoint_path ./checkpoints/SIMS_prune_80/best.pth \
            --save_dir ./results/viz_prune_80
    else
        echo "[SKIP] Exp2: checkpoint not found"
    fi

    # Exp3: ratio=0.6
    if [ -f ./checkpoints/SIMS_prune_60/best.pth ]; then
        echo "=== [Viz] Exp3: Prune 60% ==="
        conda run --no-capture-output -n $CONDA_ENV python -u visualize_results.py \
            --datasetName SIMS \
            --checkpoint_path ./checkpoints/SIMS_prune_60/best.pth \
            --save_dir ./results/viz_prune_60
    else
        echo "[SKIP] Exp3: checkpoint not found"
    fi

    # Exp4: ratio=0.4
    if [ -f ./checkpoints/SIMS_prune_40/best.pth ]; then
        echo "=== [Viz] Exp4: Prune 40% ==="
        conda run --no-capture-output -n $CONDA_ENV python -u visualize_results.py \
            --datasetName SIMS \
            --checkpoint_path ./checkpoints/SIMS_prune_40/best.pth \
            --save_dir ./results/viz_prune_40
    else
        echo "[SKIP] Exp4: checkpoint not found"
    fi

    echo ""
    echo "============================================================"
    echo "All visualizations complete! Compare results in:"
    echo "  ./results/viz_no_pruning/    (baseline, no pruning)"
    echo "  ./results/viz_prune_80/      (retain 80%)"
    echo "  ./results/viz_prune_60/      (retain 60%)"
    echo "  ./results/viz_prune_40/      (retain 40%)"
    echo "============================================================"
}

# ---- Main ----
case "${1:-all}" in
    train)
        run_train
        ;;
    viz)
        run_viz
        ;;
    all)
        run_train
        run_viz
        ;;
    *)
        echo "Usage: bash run_ablation.sh [train|viz|all]"
        exit 1
        ;;
esac
