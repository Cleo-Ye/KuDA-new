#!/bin/bash
# ============================================================
# 清理旧实验并启动 Step 1 + Step 2 完整实验
#
# 改进内容：
#   Step 1: VisionTokenPruner 权重归一化 (models/VisionTokenPruner.py)
#   Step 2: ConflictJS 参数优化 (opts.py: gate_k=8.0, gate_tau=0.12, lambda_js=0.12)
#
# 此脚本将：
#   1. 停止所有正在运行的 full_ 实验
#   2. 清理旧的 checkpoint 和日志
#   3. 启动完整 50 epoch 实验（6个配置，后6块GPU并行）
# ============================================================

echo "============================================================"
echo "🚀 启动 Step 1 + Step 2 完整实验"
echo "============================================================"
echo ""
echo "改进内容："
echo "  ✅ Step 1: VisionTokenPruner 权重归一化"
echo "     - w_t = w_t / sum(w_t)"
echo "     - Step 1 消融结果：+IEC MAE 0.310→0.302 (−2.5%)"
echo ""
echo "  ✅ Step 2: ConflictJS 参数优化"
echo "     - gate_k: 10.0 → 8.0 (门控更平滑)"
echo "     - gate_tau: 0.15 → 0.12 (更容易触发冲突分支)"
echo "     - lambda_js: 0.1 → 0.12 (加强冲突正则化)"
echo ""

# 1. 停止旧实验
echo "1️⃣ 停止所有正在运行的 full_ 实验..."
pkill -f "train.py.*full_" && echo "  ✓ 已停止实验进程" || echo "  ℹ️ 没有运行中的实验"

# 2. 清理旧数据
echo ""
echo "2️⃣ 清理旧实验数据..."
read -p "  是否清理旧的 checkpoints 和日志？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf checkpoints/full_*
    rm -rf logs/full_multi/train_*.log
    echo "  ✓ 已清理 checkpoints/full_* 和 logs/full_multi/"
else
    echo "  ℹ️ 保留旧数据（将被覆盖）"
fi

# 3. 验证改进已应用
echo ""
echo "3️⃣ 验证改进已应用..."
echo "  检查 opts.py 参数..."
if grep -q "default=8.0" KuDA/opts.py 2>/dev/null || grep -q "default=8.0" opts.py 2>/dev/null; then
    echo "  ✓ gate_k=8.0"
else
    echo "  ⚠️ gate_k 可能未正确设置"
fi

if grep -q "default=0.12.*gate_tau" KuDA/opts.py 2>/dev/null || grep -q "default=0.12.*gate_tau" opts.py 2>/dev/null; then
    echo "  ✓ gate_tau=0.12"
else
    echo "  ⚠️ gate_tau 可能未正确设置"
fi

if grep -q "default=0.12.*lambda_js" KuDA/opts.py 2>/dev/null || grep -q "default=0.12.*lambda_js" opts.py 2>/dev/null; then
    echo "  ✓ lambda_js=0.12"
else
    echo "  ⚠️ lambda_js 可能未正确设置"
fi

# 4. 启动实验
echo ""
echo "4️⃣ 启动完整实验..."
echo "  配置: baseline, +IEC(r=0.5), +ICR, IEC+ICR(r=0.5), IEC+ICR(r=0.3), IEC+ICR(KL)"
echo "  Epochs: 50"
echo "  GPU: 2,3,4,5,6,7"
echo "  预计时间: 2-3 小时"
echo ""

bash run_full_experiments.sh

echo ""
echo "============================================================"
echo "✅ 实验已启动！"
echo "============================================================"
echo ""
echo "监控进度："
echo "  tail -f logs/full_multi/train_baseline.log"
echo "  tail -f logs/full_multi/train_+IEC_r05.log"
echo "  tail -f logs/full_multi/train_+ICR_only.log          # 重点关注"
echo "  tail -f logs/full_multi/train_IEC+ICR_full.log       # 期望最好"
echo ""
echo "检查GPU："
echo "  watch -n 2 nvidia-smi"
echo ""
echo "检查NaN（应该没有）："
echo "  grep -c 'NaN' logs/full_multi/*.log"
echo ""
echo "查看中间结果（每隔1小时检查）："
echo "  for log in logs/full_multi/train_*.log; do"
echo "    echo \"=== \$(basename \$log) ===\";"
echo "    grep 'Best-MAE' \$log | tail -2;"
echo "  done"
echo ""
