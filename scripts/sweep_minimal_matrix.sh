#!/usr/bin/env bash
# =============================================================================
# 最小对照实验矩阵（与 best_retrain_2 同基线，每次只改一类因素）
# GPU：默认使用 4、5、6、7，分两波跑（先 4 个并行，再 3 个并行）
# 磁盘：--save_best_corr False，仅写入 best.pth（Test MAE 最优），不写 best_corr.pth
#
# 用法：
#   cd /path/to/KuDA && bash scripts/sweep_minimal_matrix.sh
# 自定义输出根目录：
#   SWEEP_ROOT=./checkpoints/sweep_unified/my_matrix bash scripts/sweep_minimal_matrix.sh
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

SWEEP_ROOT="${SWEEP_ROOT:-./checkpoints/sweep_unified/minimal_matrix_$(date +%Y%m%d_%H%M)}"
mkdir -p "$SWEEP_ROOT"

GPUS=(4 5 6 7)

# 与 best_retrain_2_0.4111 一致的公共参数（checkpoint_dir 按 run 覆盖）
run_train() {
  local gpu="$1"
  local name="$2"
  shift 2
  local dir="${SWEEP_ROOT}/${name}"
  mkdir -p "$dir"
  echo ">>> [${name}] CUDA_VISIBLE_DEVICES=${gpu} log -> ${dir}/train.log"
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
    --curriculum_enable True \
    --curriculum_stage1_epochs 15 \
    --curriculum_stage2_epochs 25 \
    --curriculum_stage3_ramp_epochs 5 \
    --lambda_ortho 0.0015 \
    --ortho_epsilon 0.15 \
    --focal_mae_lambda 0.5 \
    --focal_mae_stage3_only True \
    --stage3_interaction_lr_decay True \
    --stage3_interaction_lr_min_ratio 0.1 \
    --save_best_corr False \
    "$@" >> "${dir}/train.log" 2>&1
}

wait_all() {
  local ec=0
  for pid in "$@"; do
    wait "${pid}" || ec=1
  done
  return "${ec}"
}

echo "============================================================================="
echo "Sweep root: ${SWEEP_ROOT}"
echo "Matrix (7 runs):"
echo "  m00_base              — 基线（与 best_retrain_2 一致）"
echo "  m01_ortho_eps020      — 正交截断更宽 ortho_epsilon=0.2"
echo "  m02_ortho_l001        — 正交权重 lambda_ortho=0.001"
echo "  m03_ortho_l001_eps020 — lambda_ortho=0.001 + ortho_epsilon=0.2"
echo "  m04_focal030          — Focal MAE lambda=0.3"
echo "  m05_focal080          — Focal MAE lambda=0.8"
echo "  m06_lrmin005          — 阶段三交互 LR 末值比例 0.05"
echo "GPUs: ${GPUS[*]}"
echo "============================================================================="

# 第一波：4 个并行
run_train "${GPUS[0]}" m00_base &
P0=$!
run_train "${GPUS[1]}" m01_ortho_eps020 --ortho_epsilon 0.2 &
P1=$!
run_train "${GPUS[2]}" m02_ortho_l001 --lambda_ortho 0.001 &
P2=$!
run_train "${GPUS[3]}" m03_ortho_l001_eps020 --lambda_ortho 0.001 --ortho_epsilon 0.2 &
P3=$!

if ! wait_all "${P0}" "${P1}" "${P2}" "${P3}"; then
  echo "[WARN] 第一波中有任务非零退出，请检查对应 train.log"
fi

# 第二波：3 个并行（仍占满 4 卡中的 3 张）
run_train "${GPUS[0]}" m04_focal030 --focal_mae_lambda 0.3 &
P4=$!
run_train "${GPUS[1]}" m05_focal080 --focal_mae_lambda 0.8 &
P5=$!
run_train "${GPUS[2]}" m06_lrmin005 --stage3_interaction_lr_min_ratio 0.05 &
P6=$!

if ! wait_all "${P4}" "${P5}" "${P6}"; then
  echo "[WARN] 第二波中有任务非零退出，请检查对应 train.log"
fi

echo "============================================================================="
echo "全部结束。各目录仅应含 best.pth（无 best_corr.pth）："
find "${SWEEP_ROOT}" -maxdepth 2 -name 'best.pth' -print 2>/dev/null | sort
echo "汇总 summary.json 路径："
find "${SWEEP_ROOT}" -maxdepth 2 -name 'summary.json' -print 2>/dev/null | sort
echo "============================================================================="
