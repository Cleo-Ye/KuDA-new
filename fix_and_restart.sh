#!/bin/bash
# ============================================================
# 修复 NaN 问题并重启完整实验
#
# 问题原因：opts.py 中的默认值仍是激进参数，导致 NaN
# 解决方案：已将 opts.py 恢复为原始稳定参数
#
# 此脚本将：
#   1. 停止所有正在运行的 full_ 实验
#   2. 清理失败的 checkpoint 和日志
#   3. 重启完整实验（50 epoch，使用原始稳定参数）
# ============================================================

echo "============================================================"
echo "🛠️  修复 NaN 问题 - 恢复 opts.py 为原始稳定参数"
echo "============================================================"

# 1. 停止所有正在运行的 full_ 实验
echo ""
echo "1️⃣ 停止所有正在运行的 full_ 实验..."
pkill -f "train.py.*full_" && echo "  ✓ 已停止所有 full_ 实验进程" || echo "  ℹ️ 没有正在运行的 full_ 实验"

# 2. 清理失败的 checkpoint 和日志
echo ""
echo "2️⃣ 清理失败的实验数据..."
rm -rf checkpoints/full_*
rm -rf logs/full_multi/train_*.log
echo "  ✓ 已清理 checkpoints/full_* 和 logs/full_multi/"

# 3. 验证参数已修复
echo ""
echo "3️⃣ 验证 opts.py 参数..."
echo "  当前参数值："
grep -A 2 "gate_k" KuDA/opts.py 2>/dev/null || grep -A 2 "gate_k" opts.py
grep -A 2 "lambda_nce" KuDA/opts.py 2>/dev/null || grep -A 2 "lambda_nce" opts.py

# 4. 重启实验
echo ""
echo "4️⃣ 重启完整实验（50 epoch，原始稳定参数）..."
echo "  GPU: 2,3,4,5,6,7"
echo "  配置: baseline, +IEC_r05, +ICR_only, IEC+ICR_full, IEC+ICR_r03, IEC+ICR_KL"
echo ""

bash run_full_experiments.sh

echo ""
echo "============================================================"
echo "✅ 实验已重启！"
echo "============================================================"
echo ""
echo "监控进度："
echo "  tail -f logs/full_multi/train_baseline.log"
echo "  tail -f logs/full_multi/train_IEC+ICR_full.log"
echo ""
echo "检查 GPU 使用："
echo "  watch -n 2 nvidia-smi"
echo ""
echo "查看所有日志首部（检查参数）："
echo "  head -20 logs/full_multi/*.log"
echo ""
