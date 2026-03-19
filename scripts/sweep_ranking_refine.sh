#!/usr/bin/env bash
# =============================================================================
# 精调 sweep：基于 r3_e022_f030_t015（MAE 0.4066）做小范围超参搜索
#
# 改动方向：
#   1. 降低学习率：lr 2e-5→1.5e-5 / 1e-5，bert_lr 1e-5→7e-6 / 5e-6
#   2. 延长 ramp：curriculum_stage3_ramp_epochs 5→10
#   3. 更激进 LR 衰减：stage3_interaction_lr_min_ratio 0.1→0.05
#
# 固定：ortho_epsilon=0.22, focal_mae_lambda=0.3, ranking_threshold=0.15
#
# 组合：3 组 (lr, bert_lr) × 2 组 (ramp, lr_min_ratio) = 6 run
#   或简化为 3 组 lr 精调
#
# 用法：cd KuDA && bash scripts/sweep_ranking_refine.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/ranking_refine_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(4 5 6 7)

# 固定最佳基础配置
EPS=0.22
FOCAL=0.3
THRESH=0.15

run_train() {
  local gpu="$1"
  local name="$2"
  local lr="$3"
  local bert_lr="$4"
  local ramp="$5"
  local lr_min="$6"
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] gpu=${gpu}  lr=${lr}  bert_lr=${bert_lr}  ramp=${ramp}  lr_min=${lr_min}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    --datasetName sims \
    --model_type pid_dualpath \
    --use_batch_pid_prior True \
    --lr "${lr}" \
    --bert_lr "${bert_lr}" \
    --batch_size 32 \
    --seed 2024 \
    --path_layers 2 \
    --checkpoint_dir "${dir}" \
    --log_path "${dir}/" \
    --early_stop_min_epochs 60 \
    --curriculum_enable True \
    --curriculum_stage1_epochs 15 \
    --curriculum_stage2_epochs 25 \
    --curriculum_stage3_ramp_epochs "${ramp}" \
    --lambda_ortho 0.0015 \
    --ortho_epsilon "${EPS}" \
    --focal_mae_lambda "${FOCAL}" \
    --focal_mae_stage3_only True \
    --stage3_interaction_lr_decay True \
    --stage3_interaction_lr_min_ratio "${lr_min}" \
    --ranking_threshold "${THRESH}" \
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

# 任务：(name, lr, bert_lr, ramp, lr_min_ratio)
# 方案 A：3 组 lr 精调（ramp=10, lr_min=0.1）
# 方案 B：6 组 = 3 lr × 2 (ramp/lr_min 组合)
TASKS=(
  "refine_lr15e5_ramp10_lrmin01|1.5e-5|7e-6|10|0.1"
  "refine_lr1e5_ramp10_lrmin01|1e-5|5e-6|10|0.1"
  "refine_lr2e5_ramp10_lrmin01|2e-5|1e-5|10|0.1"
  "refine_lr15e5_ramp10_lrmin005|1.5e-5|7e-6|10|0.05"
  "refine_lr1e5_ramp10_lrmin005|1e-5|5e-6|10|0.05"
  "refine_lr2e5_ramp10_lrmin005|2e-5|1e-5|10|0.05"
)

echo "============================================================================="
echo "Sweep root: ${SWEEP_ROOT}"
echo "Base: e022_f030_t015 (MAE 0.4066)"
echo "GPUs: ${GPUS[*]}"
echo "Tasks: ${#TASKS[@]}"
echo "============================================================================="

idx=0
total=${#TASKS[@]}
while [ "${idx}" -lt "${total}" ]; do
  batch_pids=()
  for gi in 0 1 2 3; do
    [ "${idx}" -ge "${total}" ] && break
    IFS='|' read -r name lr bert_lr ramp lr_min <<< "${TASKS[${idx}]}"
    gpu="${GPUS[${gi}]}"
    run_train "${gpu}" "${name}" "${lr}" "${bert_lr}" "${ramp}" "${lr_min}" &
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
