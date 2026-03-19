#!/usr/bin/env bash
# =============================================================================
# 为 matrix2 的指定 run 目录生成 synergy/decouple 可视化
#
# 用法：
#   bash scripts/run_synergy_vis_matrix2.sh <run_dir>
#
# 例：
#   bash scripts/run_synergy_vis_matrix2.sh ./checkpoints/sweep_unified/matrix2_ortho_focal_20260318_0726/m2_e020_f050
# =============================================================================
set -u

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/run_synergy_vis_matrix2.sh <run_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUDA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KUDA_ROOT" || exit 1

RUN_DIR="$1"
CKPT="${RUN_DIR}/best.pth"
PYTHON_BIN="${PYTHON:-python}"

if [ ! -d "$RUN_DIR" ]; then
  echo "Run dir not found: $RUN_DIR"
  exit 1
fi

if [ ! -f "$CKPT" ]; then
  echo "Checkpoint not found: $CKPT"
  echo "请确认该 run 已训练完成且保存了 best.pth（按 Test MAE）"
  exit 1
fi

echo "Run dir : $RUN_DIR"
echo "CKPT    : $CKPT"

"${PYTHON_BIN}" visualize_synergy_decouple.py \
  --checkpoint_path "$CKPT" \
  --save_dir "$RUN_DIR" \
  --model_name "$(basename "$RUN_DIR")"

echo ""
echo "Output PNGs:"
ls -la "$RUN_DIR"/*.png 2>/dev/null || echo "  (no png generated)"

