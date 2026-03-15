#!/bin/bash
# ============================================================
# 4 个核心消融实验（以 f1_l035_pw 最佳配置为底座，不传 ablation 时即为原方案复现）
#
# 1. w/o PID Routing    --ablation_no_pid_routing
# 2. w/o Contrastive    --lambda_nce_diff 0
# 3. w/o Orthogonal     --lambda_ortho 0
# 4. w/o Dual-Branch    --ablation_single_branch
#
# 用法:
#   bash run_ablation_study.sh              # 默认 120 epoch，4 个消融各占 1 GPU
#   bash run_ablation_study.sh 80            # 仅 80 epoch
#   bash run_ablation_study.sh summary       # 仅汇总
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
CKPT_ROOT="./checkpoints/ablation_study"

if [ "$1" = "summary" ]; then
    conda run --no-capture-output -n $CONDA_ENV \
        python -u sda_pid_summary.py --ckpt_root "$CKPT_ROOT"
    exit 0
fi

MILESTONES=()
for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        MILESTONES+=("$arg")
    fi
done
if [ ${#MILESTONES[@]} -eq 0 ]; then
    MILESTONES=(120)
fi

# 与最佳模型 f1_l035_pw 同参（C1+C5 + λ_cls=0.35 + cls_pos_weight=1.2）
BEST_BASE="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"
BEST_BASE="$BEST_BASE --hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"
BEST_BASE="$BEST_BASE --lambda_classification 0.35 --cls_pos_weight 1.2"

declare -A CKPT
CKPT["wo_pid_routing"]="${CKPT_ROOT}/wo_pid_routing"
CKPT["wo_contrastive"]="${CKPT_ROOT}/wo_contrastive"
CKPT["wo_ortho"]="${CKPT_ROOT}/wo_ortho"
CKPT["wo_dual_branch"]="${CKPT_ROOT}/wo_dual_branch"

declare -A ARGS
ARGS["wo_pid_routing"]="$BEST_BASE --ablation_no_pid_routing"
ARGS["wo_contrastive"]="$BEST_BASE --lambda_nce_diff 0"
ARGS["wo_ortho"]="$BEST_BASE --lambda_ortho 0"
ARGS["wo_dual_branch"]="$BEST_BASE --ablation_single_branch"

declare -A GPU
GPU["wo_pid_routing"]="0"
GPU["wo_contrastive"]="1"
GPU["wo_ortho"]="2"
GPU["wo_dual_branch"]="3"

NAMES=("wo_pid_routing" "wo_contrastive" "wo_ortho" "wo_dual_branch")

mkdir -p logs/ablation_study
for key in "${NAMES[@]}"; do
    mkdir -p "${CKPT[$key]}"
done

echo "Ablation Study: 4 runs, milestones = ${MILESTONES[*]}"
echo ""

PREV_EPOCH=0
for MILESTONE in "${MILESTONES[@]}"; do
    echo "============================================================"
    if [ $PREV_EPOCH -eq 0 ]; then
        echo " Phase: 0 → ${MILESTONE} epoch"
    else
        echo " Phase: ${PREV_EPOCH} → ${MILESTONE} epoch（续训）"
    fi
    echo "============================================================"

    for key in "${NAMES[@]}"; do
        CKPT_DIR="${CKPT[$key]}"
        EXTRA="${ARGS[$key]}"
        GPUID="${GPU[$key]}"
        BEST_PTH="${CKPT_DIR}/best.pth"
        LOG="logs/ablation_study/${key}_ep${MILESTONE}.log"

        if [ $PREV_EPOCH -gt 0 ] && [ -f "$BEST_PTH" ]; then
            CMD="--resume ${BEST_PTH} --n_epochs ${MILESTONE} --checkpoint_dir ${CKPT_DIR} --log_path ./log/ablation"
        else
            CMD="$EXTRA --n_epochs ${MILESTONE} --checkpoint_dir ${CKPT_DIR} --log_path ./log/ablation"
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

echo "Ablation Study 完成。"
