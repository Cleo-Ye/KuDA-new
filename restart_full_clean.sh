#!/bin/bash
# 清理所有失败的完整实验并重启

echo "======================================================================" echo "清理失败的实验并重新启动（使用原始稳定参数）"
echo "======================================================================"
echo ""

# 停止可能还在运行的进程
echo "停止现有训练进程..."
pkill -f "train.py.*full_"
sleep 2

# 清理失败的日志和checkpoint
echo "清理失败的文件..."
rm -rf logs/full_multi/train_+ICR_only.log
rm -rf logs/full_multi/train_IEC+ICR_full.log
rm -rf logs/full_multi/train_IEC+ICR_r03.log
rm -rf logs/full_multi/train_IEC+ICR_KL.log
rm -rf checkpoints/full_+ICR_only
rm -rf checkpoints/full_IEC+ICR_full
rm -rf checkpoints/full_IEC+ICR_r03
rm -rf checkpoints/full_IEC+ICR_KL

echo "完成！"
echo ""

# 重新运行
echo "重新启动实验（使用原始参数，所有配置一致）..."
bash run_full_experiments.sh

echo ""
echo "======================================================================"
echo "监控命令："
echo "======================================================================"
echo ""
echo "实时查看进度:"
echo "  watch -n 10 'grep \"Best-MAE\" logs/full_multi/*.log | tail -12'"
echo ""
echo "检查是否还有 NaN:"
echo "  grep \"loss=nan\" logs/full_multi/train_IEC+ICR_full.log"
echo ""
echo "查看训练loss（应该是正常数值）:"
echo "  tail -50 logs/full_multi/train_IEC+ICR_full.log | grep \"train:\""
echo ""
