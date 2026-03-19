#!/usr/bin/env bash
# =============================================================================
# 消融实验 7 组：以 r3_e022_f030_t015 为底座
#
# 1. Full          完整模型（基线）
# 2. w/o PID        --ablation_no_pid_routing
# 3. w/o DualBranch --ablation_single_branch
# 4. w/o L_ortho    --lambda_ortho 0
# 5. w/o L_rank     --lambda_rank 0
# 6. w/o Curriculum --curriculum_enable False
# 7. w/o All        关闭所有改进：PID+DualBranch+Ortho+Rank+Curriculum+Focal+Stage3LRdecay
#
# 固定：seed=1111, ortho_epsilon=0.22, focal_mae_lambda=0.3, ranking_threshold=0.15
# 资源：GPU 0-7，7 任务可并行
#
# 用法：cd KuDA && bash scripts/sweep_ablation_6groups.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/ablation_6groups_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(0 1 2 3 4 5 6 7)
SEED=2024

# 底座：r3_e022_f030_t015
BASE_ARGS="--datasetName sims --model_type pid_dualpath --use_batch_pid_prior True"
BASE_ARGS="$BASE_ARGS --lr 2e-5 --bert_lr 1e-5 --batch_size 32 --seed ${SEED}"
BASE_ARGS="$BASE_ARGS --path_layers 2 --early_stop_min_epochs 60"
BASE_ARGS="$BASE_ARGS --lambda_ortho 0.0015 --ortho_epsilon 0.22 --focal_mae_lambda 0.3"
BASE_ARGS="$BASE_ARGS --focal_mae_stage3_only True --stage3_interaction_lr_decay True --stage3_interaction_lr_min_ratio 0.1"
BASE_ARGS="$BASE_ARGS --ranking_threshold 0.15 --save_best_corr False"
BASE_ARGS="$BASE_ARGS --curriculum_enable True --curriculum_stage1_epochs 15 --curriculum_stage2_epochs 25 --curriculum_stage3_ramp_epochs 10"

run_train() {
  local gpu="$1"
  local name="$2"
  local extra="$3"
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] gpu=${gpu}  extra=${extra:-无}"

  CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
    $BASE_ARGS \
    --checkpoint_dir "${dir}" \
    --log_path "${dir}/" \
    $extra \
    >> "${dir}/train.log" 2>&1
}

wait_all() {
  local ec=0
  for pid in "$@"; do
    wait "${pid}" || ec=1
  done
  return "${ec}"
}

# 任务：(name, extra_args)
TASKS=(
  "abl_full|"
  "abl_wo_pid_routing|--ablation_no_pid_routing"
  "abl_wo_dual_branch|--ablation_single_branch"
  "abl_wo_ortho|--lambda_ortho 0"
  "abl_wo_rank|--lambda_rank 0"
  "abl_wo_curriculum|--curriculum_enable False"
  "abl_wo_all|--ablation_no_pid_routing --ablation_single_branch --lambda_ortho 0 --lambda_rank 0 --curriculum_enable False --focal_mae_lambda 0 --stage3_interaction_lr_decay False"
)

echo "============================================================================="
echo "Ablation 7 groups (seed=${SEED})"
echo "Sweep root: ${SWEEP_ROOT}"
echo "Base: r3_e022_f030_t015"
echo "GPUs: ${GPUS[*]}"
echo "============================================================================="
echo "  1. abl_full           完整模型"
echo "  2. abl_wo_pid_routing w/o PID Routing"
echo "  3. abl_wo_dual_branch w/o Dual-Branch"
echo "  4. abl_wo_ortho       w/o L_ortho"
echo "  5. abl_wo_rank        w/o L_rank"
echo "  6. abl_wo_curriculum  w/o Curriculum"
echo "  7. abl_wo_all         w/o All (PID+DualBranch+Ortho+Rank+Curriculum+Focal+Stage3LRdecay)"
echo "============================================================================="

idx=0
total=${#TASKS[@]}
while [ "${idx}" -lt "${total}" ]; do
  batch_pids=()
  for gi in 0 1 2 3 4 5 6 7; do
    [ "${idx}" -ge "${total}" ] && break
    IFS='|' read -r name extra <<< "${TASKS[${idx}]}"
    gpu="${GPUS[${gi}]}"
    run_train "${gpu}" "${name}" "${extra}" &
    batch_pids+=("$!")
    idx=$((idx + 1))
    sleep 1
  done
  [ ${#batch_pids[@]} -gt 0 ] && wait_all "${batch_pids[@]}" || echo "[WARN] 本批有任务非零退出"
done

echo "============================================================================="
echo "消融实验完成，生成汇总表..."
python scripts/collect_matrix2_table.py --sweep_root "${SWEEP_ROOT}" | tee "${SWEEP_ROOT}/table.txt"
echo "CSV: ${SWEEP_ROOT}/table.csv"
echo "============================================================================="
