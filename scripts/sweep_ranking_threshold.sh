#!/usr/bin/env bash
# =============================================================================
# Ranking Threshold Sweep：m2_e020_f050 / m2_e022_f050 × ranking_threshold 0.15/0.2/0.25
#
# 组合：2 个基础配置 × 3 个阈值 = 6 个 run
#   - m2_e020_f030: ortho_epsilon=0.20, focal_mae_lambda=0.3
#   - m2_e022_f030: ortho_epsilon=0.22, focal_mae_lambda=0.3
#   - ranking_threshold: 0.15 / 0.2 / 0.25
#
# 资源：GPU 4-7
# 输出：仅保留 best.pth（best-MAE），训练结束自动生成 table.txt + table.csv
#
# 用法：
#   cd KuDA && bash scripts/sweep_ranking_threshold.sh
#   SWEEP_ROOT=./checkpoints/sweep_unified/rank_thresh bash scripts/sweep_ranking_threshold.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/ranking_threshold_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(4 5 6 7)

run_train() {
  local gpu="$1"
  local name="$2"
  local eps="$3"
  local focal="$4"
  local thresh="$5"
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] gpu=${gpu}  ortho_epsilon=${eps}  focal=${focal}  ranking_threshold=${thresh}"

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
    --ortho_epsilon "${eps}" \
    --focal_mae_lambda "${focal}" \
    --focal_mae_stage3_only True \
    --stage3_interaction_lr_decay True \
    --stage3_interaction_lr_min_ratio 0.1 \
    --ranking_threshold "${thresh}" \
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

# 基础配置：(eps, focal)
BASE_E020=(0.20 0.3)
BASE_E022=(0.22 0.3)
# 阈值
THRESH_LIST=(0.15 0.2 0.25)

echo "============================================================================="
echo "Sweep root: ${SWEEP_ROOT}"
echo "GPUs: ${GPUS[*]}"
echo "Base: m2_e020_f030 (eps=0.20,f=0.3) / m2_e022_f030 (eps=0.22,f=0.3)"
echo "ranking_threshold: ${THRESH_LIST[*]}"
echo "============================================================================="

# 生成 6 个任务
TASKS=()
for base_name in "e020_f030" "e022_f030"; do
  if [ "$base_name" = "e020_f030" ]; then
    eps="${BASE_E020[0]}"
    focal="${BASE_E020[1]}"
  else
    eps="${BASE_E022[0]}"
    focal="${BASE_E022[1]}"
  fi
  for thresh in "${THRESH_LIST[@]}"; do
    thresh_tag="t$(python3 -c "print(f'{int(float(\"${thresh}\")*100):03d}')")"
    name="r3_${base_name}_${thresh_tag}"
    TASKS+=("${name}|${eps}|${focal}|${thresh}")
  done
done

# 分批并行：每批最多 4 个（GPU 4,5,6,7）
idx=0
total=${#TASKS[@]}
while [ "${idx}" -lt "${total}" ]; do
  batch_pids=()
  for gi in 0 1 2 3; do
    [ "${idx}" -ge "${total}" ] && break
    IFS='|' read -r name eps focal thresh <<< "${TASKS[${idx}]}"
    gpu="${GPUS[${gi}]}"
    run_train "${gpu}" "${name}" "${eps}" "${focal}" "${thresh}" &
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
