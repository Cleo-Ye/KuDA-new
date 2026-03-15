#!/bin/bash
# ============================================================
# 消融实验 8 GPU 并行：前 4 个 GPU 跑 80 epoch，后 4 个 GPU 跑 120 epoch
# 4 个消融 × 2 档 epoch = 8 个任务同时跑
#
# 用法:
#   bash run_ablation_study_8gpu.sh        # 一键 8 任务
#   bash run_ablation_study_8gpu.sh summary   # 汇总 8 条结果
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
CKPT_ROOT="./checkpoints/ablation_study"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root "$CKPT_ROOT" --ablation_8ep
    exit 0
fi

# 与最佳模型 f1_l035_pw 同参
BEST_BASE="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"
BEST_BASE="$BEST_BASE --hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"
BEST_BASE="$BEST_BASE --lambda_classification 0.35 --cls_pos_weight 1.2"

# 8 个任务：key 子目录名，EP 档位，GPU id
# 前 4：80 epoch，GPU 0-3
# 后 4：120 epoch，GPU 4-7
NAMES_80=("wo_pid_routing_80" "wo_contrastive_80" "wo_ortho_80" "wo_dual_branch_80")
NAMES_120=("wo_pid_routing_120" "wo_contrastive_120" "wo_ortho_120" "wo_dual_branch_120")

ARGS_80=(
    "$BEST_BASE --ablation_no_pid_routing"
    "$BEST_BASE --lambda_nce_diff 0"
    "$BEST_BASE --lambda_ortho 0"
    "$BEST_BASE --ablation_single_branch"
)
ARGS_120=(
    "$BEST_BASE --ablation_no_pid_routing"
    "$BEST_BASE --lambda_nce_diff 0"
    "$BEST_BASE --lambda_ortho 0"
    "$BEST_BASE --ablation_single_branch"
)

mkdir -p logs/ablation_study
for i in "${!NAMES_80[@]}"; do
    mkdir -p "${CKPT_ROOT}/${NAMES_80[$i]}"
    mkdir -p "${CKPT_ROOT}/${NAMES_120[$i]}"
done

echo "Ablation 8 GPU: 前 4 个 80 ep (GPU 0-3)，后 4 个 120 ep (GPU 4-7)"
echo ""

# GPU 0-3: 80 epoch
for i in "${!NAMES_80[@]}"; do
    key="${NAMES_80[$i]}"
    CKPT_DIR="${CKPT_ROOT}/${key}"
    EXTRA="${ARGS_80[$i]}"
    LOG="logs/ablation_study/${key}.log"
    echo "  [GPU ${i}] ${key} (80 ep) -> ${LOG}"
    CUDA_VISIBLE_DEVICES=$i conda run --no-capture-output -n $CONDA_ENV \
        python -u train.py $EXTRA --n_epochs 80 --checkpoint_dir "$CKPT_DIR" --log_path ./log/ablation_8gpu > "$LOG" 2>&1 &
done

# GPU 4-7: 120 epoch
for i in "${!NAMES_120[@]}"; do
    gpu=$((i + 4))
    key="${NAMES_120[$i]}"
    CKPT_DIR="${CKPT_ROOT}/${key}"
    EXTRA="${ARGS_120[$i]}"
    LOG="logs/ablation_study/${key}.log"
    echo "  [GPU ${gpu}] ${key} (120 ep) -> ${LOG}"
    CUDA_VISIBLE_DEVICES=$gpu conda run --no-capture-output -n $CONDA_ENV \
        python -u train.py $EXTRA --n_epochs 120 --checkpoint_dir "$CKPT_DIR" --log_path ./log/ablation_8gpu > "$LOG" 2>&1 &
done

wait

echo ""
echo "========== 汇总（8 条）=========="
conda run --no-capture-output -n $CONDA_ENV \
    python -u sda_pid_summary.py --ckpt_root "$CKPT_ROOT" --ablation_8ep
echo ""
echo "Ablation 8 GPU 完成。"
