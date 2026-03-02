#!/bin/bash
# ============================================================
# 修复 NaN 问题的最终方案
#
# 根本原因：不仅是 opts.py 的默认参数，还有代码中的"改进逻辑"
#   1. opts.py: 激进的默认参数值
#   2. VisionTokenPruner.py: senti_norm归一化+sqrt导致数值不稳定
#   3. DyRoutFusion_CLS.py: sqrt(rho)和残差连接导致数值不稳定
#
# 已修复：
#   - opts.py: 所有参数恢复为原始稳定值
#   - VisionTokenPruner.py: 恢复为简单的 w_t = |senti_t| * mask_t
#   - DyRoutFusion_CLS.py: 移除 sqrt(rho) 和残差项
#
# 此脚本将：
#   1. 停止所有正在运行的 full_ 实验
#   2. 清理失败的 checkpoint 和日志  
#   3. 重启完整实验（50 epoch，使用完全稳定的代码+参数）
# ============================================================

echo "============================================================"
echo "🛠️  修复 NaN 问题 - 代码和参数全部恢复为原始稳定版本"
echo "============================================================"
echo ""
echo "修复内容："
echo "  ✓ opts.py: 参数恢复 (gate_k=10.0, gate_tau=0.15, lambda_js=0.1等)"
echo "  ✓ VisionTokenPruner.py: 简化权重计算 (w_t = |senti_t| * mask_t)"
echo "  ✓ DyRoutFusion_CLS.py: 移除 sqrt(rho) 和残差连接"
echo ""

# 1. 停止所有正在运行的 full_ 实验
echo "1️⃣ 停止所有正在运行的 full_ 实验..."
pkill -f "train.py.*full_" && echo "  ✓ 已停止所有 full_ 实验进程" || echo "  ℹ️ 没有正在运行的 full_ 实验"

# 2. 清理失败的 checkpoint 和日志
echo ""
echo "2️⃣ 清理失败的实验数据..."
rm -rf checkpoints/full_*
rm -rf logs/full_multi/train_*.log
echo "  ✓ 已清理 checkpoints/full_* 和 logs/full_multi/"

# 3. 重启实验
echo ""
echo "3️⃣ 重启完整实验（50 epoch，完全稳定的代码+参数）..."
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
echo "  tail -f logs/full_multi/train_+IEC_r05.log"
echo "  tail -f logs/full_multi/train_IEC+ICR_full.log"
echo ""
echo "检查 GPU 使用："
echo "  watch -n 2 nvidia-smi"
echo ""
echo "验证不再有 NaN（等待几分钟后检查）："
echo "  grep -c 'NaN' logs/full_multi/train_+IEC_r05.log  # 应该是 0"
echo ""
