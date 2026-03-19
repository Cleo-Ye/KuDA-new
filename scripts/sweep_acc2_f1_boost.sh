#!/usr/bin/env bash
# =============================================================================
# Acc-2 / F1 提升 sweep：基于 r3_e022_f030_t015（MAE 0.4066, Corr 0.6103）
#
# 当前 Acc-2=0.7921、F1=0.7904，低于 KuDA baseline（80.74%, 80.71%）
# 通过增强分类辅助损失提升边界判别能力：
#   - lambda_classification: 0.35（默认）→ 0.4 / 0.45
#   - cls_pos_weight: 1.0（默认）→ 1.2（正类少时提升 F1）
#
# 固定：ortho_epsilon=0.22, focal_mae_lambda=0.3, ranking_threshold=0.15
#
# 组合：3 × lambda_cls × 2 × cls_pos_weight = 6 run
#
# 用法：cd KuDA && bash scripts/sweep_acc2_f1_boost.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/acc2_f1_boost_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(4 5 6 7)

EPS=0.22
FOCAL=0.3
THRESH=0.15

run_train() {
  local gpu="$1"
  local name="$2"
  local lcls="$3"
  local pw="$4"
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] gpu=${gpu}  lambda_classification=${lcls}  cls_pos_weight=${pw}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --datasetName sims \
    --model_type pid_dualpath \
    --use_batch_pid_prior True \
    --lr 2e-5 \
    --bert_lr 1e-5 \
    --batch_size 32 \
    --seed 2024 \
    --path_layers 2 \
    --checkpoint_dir "${dir}" \
    --log_path "${dir}/" \
    --early_stop_min_epochs 60 \
    --curriculum_enable True \
    --curriculum_stage1_epochs 15 \
    --curriculum_stage2_epochs 25 \
    --curriculum_stage3_ramp_epochs 10 \
    --lambda_ortho 0.0015 \
    --ortho_epsilon "${EPS}" \
    --focal_mae_lambda "${FOCAL}" \
    --focal_mae_stage3_only True \
    --stage3_interaction_lr_decay True \
    --stage3_interaction_lr_min_ratio 0.1 \
    --ranking_threshold "${THRESH}" \
    --lambda_classification "${lcls}" \
    --cls_pos_weight "${pw}" \
    --save_best_corr False \
    >> "${dir}/train.log" 2>&1
}

wait_all() {
  local ec=0
  for pid in "$@"; do
    wait "${pid}" || ec=1
  done
  return "${ec}"
}

# 任务：(name, lambda_classification, cls_pos_weight)
TASKS=(
  "acc2f1_lcls035_pw10|0.35|1.0"
  "acc2f1_lcls035_pw12|0.35|1.2"
  "acc2f1_lcls040_pw10|0.4|1.0"
  "acc2f1_lcls040_pw12|0.4|1.2"
  "acc2f1_lcls045_pw10|0.45|1.0"
  "acc2f1_lcls045_pw12|0.45|1.2"
)

echo "============================================================================="
echo "Sweep root: ${SWEEP_ROOT}"
echo "Base: e022_f030_t015 (MAE 0.4066, Corr 0.6103, Acc-2 0.7921, F1 0.7904)"
echo "Target: 提升 Acc-2 / F1 接近 KuDA baseline (80.74%, 80.71%)"
echo "GPUs: ${GPUS[*]}"
echo "============================================================================="

idx=0
total=${#TASKS[@]}
while [ "${idx}" -lt "${total}" ]; do
  batch_pids=()
  for gi in 0 1 2 3; do
    [ "${idx}" -ge "${total}" ] && break
    IFS='|' read -r name lcls pw <<< "${TASKS[${idx}]}"
    gpu="${GPUS[${gi}]}"
    run_train "${gpu}" "${name}" "${lcls}" "${pw}" &
    batch_pids+=("$!")
    idx=$((idx + 1))
    sleep 1
  done
  [ ${#batch_pids[@]} -gt 0 ] && wait_all "${batch_pids[@]}" || echo "[WARN] 本批有任务非零退出"
done

echo "============================================================================="
echo "训练全部完成，生成汇总表..."
python scripts/collect_matrix2_table.py --sweep_root "${SWEEP_ROOT}" | tee "${SWEEP_ROOT}/table.txt"
echo "CSV: ${SWEEP_ROOT}/table.csv"
echo "============================================================================="
