#!/usr/bin/env bash
# =============================================================================
# 二阶矩阵：ortho_epsilon × focal_mae_lambda （以 m01_ortho_eps020 为新基线）
#
# Grid:
#   ortho_epsilon:    0.18 / 0.20 / 0.22
#   focal_mae_lambda: 0.0  / 0.3  / 0.5
#
# 资源：
#   - 默认占用 GPU 4、5、6、7 分批并行运行
#   - 每个配置仅保存 best.pth（按 Test MAE 选优），不保存 best_corr.pth
#
# 输出：
#   - 每个 run 一个独立目录：${SWEEP_ROOT}/${RUN_NAME}/
#   - 训练日志：train.log
#   - 训练总结：summary.json（含 test_at_best_mae）
#   - 训练完成后自动生成汇总表：table.txt + table.csv
#
# 用法：
#   cd KuDA && bash scripts/sweep_matrix2_ortho_focal.sh
# 自定义输出根目录：
#   SWEEP_ROOT=./checkpoints/sweep_unified/matrix2 bash scripts/sweep_matrix2_ortho_focal.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/matrix2_ortho_focal_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(4 5 6 7)

# 与 m01_ortho_eps020 基线一致的公共参数（仅 eps 与 focal 在 grid 中覆盖）
run_train() {
  local gpu="$1"
  local name="$2"
  local eps="$3"
  local focal="$4"
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] gpu=${gpu}  ortho_epsilon=${eps}  focal_mae_lambda=${focal}"

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
    --early_stop_min_epochs 40 \
    --curriculum_enable True \
    --curriculum_stage1_epochs 15 \
    --curriculum_stage2_epochs 25 \
    --curriculum_stage3_ramp_epochs 5 \
    --lambda_ortho 0.0015 \
    --ortho_epsilon "${eps}" \
    --focal_mae_lambda "${focal}" \
    --focal_mae_stage3_only True \
    --stage3_interaction_lr_decay True \
    --stage3_interaction_lr_min_ratio 0.1 \
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

E_LIST=(0.18 0.20 0.22)
F_LIST=(0.0 0.3 0.5)

echo "============================================================================="
echo "Sweep root: ${SWEEP_ROOT}"
echo "GPUs: ${GPUS[*]}"
echo "Grid: ortho_epsilon=[${E_LIST[*]}]  focal_mae_lambda=[${F_LIST[*]}]"
echo "============================================================================="

# 生成 9 个任务（按行：eps 固定，列：focal 变化）
TASK_NAMES=()
TASK_EPS=()
TASK_FOCAL=()
for eps in "${E_LIST[@]}"; do
  for focal in "${F_LIST[@]}"; do
    # 目录名：e020_f050 这种风格（小数点去掉）
    eps_tag="$(python - <<PY
v=float("${eps}")
print(f"e{int(round(v*100)):03d}")
PY
)"
    focal_tag="$(python - <<PY
v=float("${focal}")
print(f"f{int(round(v*100)):03d}")
PY
)"
    name="m2_${eps_tag}_${focal_tag}"
    TASK_NAMES+=("${name}")
    TASK_EPS+=("${eps}")
    TASK_FOCAL+=("${focal}")
  done
done

# 分批并行：每批最多 4 个（GPU 4-7）
idx=0
total=${#TASK_NAMES[@]}
while [ "${idx}" -lt "${total}" ]; do
  pids=()
  for gi in 0 1 2 3; do
    if [ "${idx}" -ge "${total}" ]; then
      break
    fi
    gpu="${GPUS[${gi}]}"
    name="${TASK_NAMES[${idx}]}"
    eps="${TASK_EPS[${idx}]}"
    focal="${TASK_FOCAL[${idx}]}"
    run_train "${gpu}" "${name}" "${eps}" "${focal}" &
    pids+=("$!")
    idx=$((idx + 1))
    # 小间隔，避免同时启动导致 IO 抖动
    sleep 1
  done

  if ! wait_all "${pids[@]}"; then
    echo "[WARN] 有任务非零退出，请检查对应 ${SWEEP_ROOT}/m2_*/train.log"
  fi
done

echo "============================================================================="
echo "训练全部完成，生成汇总表..."
python scripts/collect_matrix2_table.py --sweep_root "${SWEEP_ROOT}" | tee "${SWEEP_ROOT}/table.txt"
echo "CSV 已写入：${SWEEP_ROOT}/table.csv"
echo "============================================================================="

