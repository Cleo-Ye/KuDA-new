#!/bin/bash
# ============================================================
# 主方案 + 消融（无正交）多 GPU 一起跑
#
# - 新主方案: λ_ortho=0 的 f1_l035_pw (C1+C5 + λ_cls=0.35 + cls_pos_weight=1.2)
# - 4 份主方案 (不同随机性) + 3 个消融配置
# - 共 7 个任务，绑定 GPU 1–7（GPU0 空闲）
# - 支持多里程碑：80 / 100 / 120 等，按 best.pth 续训
#
# 用法:
#   bash run_main_no_ortho_and_ablation.sh              # 默认里程碑: 80 100 120
#   bash run_main_no_ortho_and_ablation.sh 60 80 100    # 自定义里程碑
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"
CKPT_ROOT="./checkpoints/main_no_ortho_suite"

# 解析里程碑列表（数字参数），默认 80 100 120
MILESTONES=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    MILESTONES+=("$arg")
  fi
done
if [ ${#MILESTONES[@]} -eq 0 ]; then
  MILESTONES=(80 100 120)
fi

# 与 f1_l035_pw 一致的基础配置
BASE_COMMON="--datasetName sims --use_cmvn True --lambda_senti 0.05 --pid_warmup_epochs 2"
BASE_COMMON="$BASE_COMMON --hidden_size 512 --ffn_size 1024 --dropout 0.2 --lr 3e-5 --pid_lr 3e-4 --weight_decay 5e-5"
BASE_COMMON="$BASE_COMMON --lambda_classification 0.35 --cls_pos_weight 1.2 --lambda_ortho 0"

# 7 个任务: 4 个主方案 + 3 个消融 (都在 λ_ortho=0 基础上)
NAMES=(
  "main_no_ortho_r1"
  "main_no_ortho_r2"
  "main_no_ortho_r3"
  "main_no_ortho_r4"
  "ablate_no_pid"
  "ablate_no_contrast"
  "ablate_no_dual_branch"
)

ARGS=(
  "$BASE_COMMON"
  "$BASE_COMMON"
  "$BASE_COMMON"
  "$BASE_COMMON"
  "$BASE_COMMON --ablation_no_pid_routing"
  "$BASE_COMMON --lambda_nce_diff 0"
  "$BASE_COMMON --ablation_single_branch"
)

# GPU 映射: 1–7
GPUS=(1 2 3 4 5 6 7)

mkdir -p logs/main_no_ortho_suite
for name in "${NAMES[@]}"; do
  mkdir -p "${CKPT_ROOT}/${name}"
done

echo "Run main no-ortho + ablations: milestones=${MILESTONES[*]}, GPUs=${GPUS[*]}"
echo ""

PREV_EPOCH=0
for M in "${MILESTONES[@]}"; do
  echo "============================================================"
  if [ $PREV_EPOCH -eq 0 ]; then
    echo " Phase: 0 -> ${M} epoch"
  else
    echo " Phase: ${PREV_EPOCH} -> ${M} epoch（续训）"
  fi
  echo "============================================================"

  for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    extra="${ARGS[$i]}"
    gpu="${GPUS[$i]}"
    ckpt_dir="${CKPT_ROOT}/${name}"
    best_pth="${ckpt_dir}/best.pth"
    log_file="logs/main_no_ortho_suite/${name}_ep${M}.log"

    if [ $PREV_EPOCH -gt 0 ] && [ -f "$best_pth" ]; then
      # 续训：从 best.pth 恢复到更高里程碑
      CMD="--resume ${best_pth} --n_epochs ${M} --checkpoint_dir ${ckpt_dir} --log_path ./log/main_no_ortho_suite"
    else
      # 首次到当前里程碑
      CMD="${extra} --n_epochs ${M} --checkpoint_dir ${ckpt_dir} --log_path ./log/main_no_ortho_suite"
    fi

    echo "  [GPU ${gpu}] ${name} -> ${log_file}"
    CUDA_VISIBLE_DEVICES=${gpu} conda run --no-capture-output -n ${CONDA_ENV} \
      python -u train.py ${CMD} > "${log_file}" 2>&1 &
  done

  wait

  echo ""
  echo "========== ${M} epoch 汇总 (main_no_ortho_suite) =========="
  conda run --no-capture-output -n ${CONDA_ENV} \
    python -u sda_pid_summary.py --ckpt_root "${CKPT_ROOT}"
  echo ""

  PREV_EPOCH=${M}
done

echo ""
echo "所有 main_no_ortho + ablation 任务（多里程碑）完成。"
