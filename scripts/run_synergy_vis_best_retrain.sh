#!/bin/bash
# 为 best_retrain 生成 synergy 可视化（synergy_distribution.png, binned_mae_corr_by_s.png 等）
# 用法: bash scripts/run_synergy_vis_best_retrain.sh
# 或激活 conda 后: python visualize_synergy_decouple.py --checkpoint_path ... --save_dir ...

cd "$(dirname "$0")/.."
CKPT="./checkpoints/sweep_unified/best_retrain/best.pth"
SAVE_DIR="./checkpoints/sweep_unified/best_retrain"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint not found: $CKPT"
    echo "请先用最佳配置重训得到 best.pth"
    exit 1
fi

python visualize_synergy_decouple.py \
  --checkpoint_path "$CKPT" \
  --save_dir "$SAVE_DIR"

echo ""
echo "输出文件:"
ls -la "$SAVE_DIR"/*.png 2>/dev/null || echo "  (无 png 文件)"
