#!/bin/bash
# 验证修复：重新启动完整实验并监控

echo "======================================================================"
echo "重新启动完整实验（已修复 NaN 问题）"
echo "======================================================================"
echo ""
echo "修复内容："
echo "  1. baseline 和 +IEC 使用原始稳定参数"
echo "  2. IEC+ICR 配置使用改进参数"
echo "  3. 添加 NaN 检测与跳过机制"
echo ""

# 清理旧的失败日志和 checkpoint
echo "清理旧文件..."
rm -rf logs/full_multi/train_+IEC_r05.log
rm -rf logs/full_multi/train_+IEC_only.log
rm -rf checkpoints/full_+IEC_r05
rm -rf checkpoints/full_+IEC_only

echo "完成！"
echo ""

# 重新运行
echo "重新启动实验..."
bash run_full_experiments.sh

echo ""
echo "======================================================================"
echo "监控命令："
echo "======================================================================"
echo ""
echo "实时查看进度:"
echo "  watch -n 10 'grep \"Best-MAE\" logs/full_multi/*.log | tail -12'"
echo ""
echo "检查 NaN 警告:"
echo "  grep \"Warning: NaN\" logs/full_multi/train_*.log"
echo ""
echo "查看单个日志:"
echo "  tail -f logs/full_multi/train_+IEC_r05.log"
echo ""
