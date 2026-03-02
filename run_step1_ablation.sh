#!/bin/bash
# ============================================================
# Step 1 消融实验：测试 VisionTokenPruner 改进效果
#
# 改进内容：w_t 归一化（/sum），使权重分布更一致
# 
# 测试配置：
#   1. baseline（对照组，不受影响）
#   2. +IEC(r=0.5)（主要测试目标）
#   3. IEC+ICR(r=0.5)（验证不破坏组合效果）
#
# 用法:
#   bash run_step1_ablation.sh              # 默认 15 epoch, GPU 2,3,4
#   bash run_step1_ablation.sh 20           # 指定 epoch
#   bash run_step1_ablation.sh 15 4,5,6     # 指定 GPU
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

# 参数解析
N_EPOCHS="${1:-15}"
GPU_IDS="${2:-2,3,4}"

# 解析 GPU 列表
IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}

# 基础参数（使用当前稳定参数）
BASE_ARGS="--datasetName sims --use_cmvn True --use_ki False --n_epochs $N_EPOCHS"
BASE_ARGS="$BASE_ARGS --lambda_nce 0.1 --lambda_senti 0.05 --lambda_js 0.1 --lambda_con 0.1 --lambda_cal 0.1"

mkdir -p logs/step1_ablation

# 实验配置
EXPERIMENTS=(
    "baseline|--use_conflict_js False --use_vision_pruning False"
    "+IEC_r05|--use_conflict_js False --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
    "IEC+ICR_r05|--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
)

echo "============================================================"
echo "Step 1 Ablation: VisionTokenPruner 归一化改进"
echo "============================================================"
echo "Epochs: $N_EPOCHS"
echo "GPUs: ${GPU_IDS}"
echo "改进: w_t 归一化 (w_t / sum(w_t))"
echo "============================================================"
echo ""

run_train() {
    PIDS=()
    i=0
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME EXTRA <<< "$exp"
        GPU_ID=${GPUS[$((i % N_GPUS))]}
        CKPT_DIR="./checkpoints/step1_${NAME}"
        LOG_FILE="logs/step1_ablation/train_${NAME}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [Train] ${NAME} (GPU ${GPU_ID}) ==="
        echo "  Checkpoint: ${CKPT_DIR}"
        echo "  Log: ${LOG_FILE}"

        CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n $CONDA_ENV python -u train.py \
            $BASE_ARGS $EXTRA \
            --checkpoint_dir "$CKPT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
        ((i++))
    done

    echo ""
    echo "All ${#EXPERIMENTS[@]} experiments launched."
    echo "PIDs: ${PIDS[@]}"
    echo ""
    echo "Monitor progress:"
    echo "  tail -f logs/step1_ablation/train_+IEC_r05.log"
    echo "  tail -f logs/step1_ablation/train_IEC+ICR_r05.log"
    echo ""
    echo "Check GPU:"
    echo "  watch -n 2 nvidia-smi"
    echo ""
    echo "Waiting for all experiments to finish..."
    wait
    echo ""
    echo "✅ All training experiments finished!"
    echo ""
    
    # 自动运行汇总
    run_summary
}

run_summary() {
    echo "============================================================"
    echo "Generating test set summary..."
    echo "============================================================"
    
    # 创建汇总脚本
    cat > logs/step1_ablation/step1_summary.py << 'PYEOF'
#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '.')
from opts import get_opt
from core.dataset import get_dataloader
from core.metric import MetricsTop
from models.OverallModal import OverallModal

def load_and_eval(checkpoint_path, opt):
    """Load checkpoint and evaluate on test set"""
    model = OverallModal(opt)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    
    dataLoader = get_dataloader(opt)
    test_loader = dataLoader['test']
    metrics = MetricsTop(opt.train_mode).getMetics(opt.datasetName)
    
    preds, truths = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            vision = batch_data['vision'].cuda()
            audio = batch_data['audio'].cuda()
            text = batch_data['text'].cuda()
            labels = batch_data['labels']['M']
            
            outputs = model(text, vision, audio)
            preds.append(outputs['M'].cpu())
            truths.append(labels)
    
    preds = torch.cat(preds, dim=0)
    truths = torch.cat(truths, dim=0)
    results = metrics(preds, truths)
    return results

# 配置
configs = [
    ('baseline', '--use_conflict_js False --use_vision_pruning False'),
    ('+IEC(r=0.5)', '--use_conflict_js False --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5'),
    ('IEC+ICR(r=0.5)', '--use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5'),
]

print("=" * 80)
print("Step 1 Ablation Summary (test set)")
print("=" * 80)
print(f"{'Config':<20} {'MAE':<12} {'Corr':<8} {'Mult_acc_2':<12} {'F1_score'}")
print("-" * 80)

for name, args in configs:
    ckpt_name = name.replace('(', '').replace(')', '').replace('=', '').replace('.', '').replace(' ', '_').replace('+', '')
    if ckpt_name == 'baseline':
        ckpt_path = f'./checkpoints/step1_baseline/best.pth'
    elif 'IEC' in ckpt_name and 'ICR' not in ckpt_name:
        ckpt_path = f'./checkpoints/step1_+IEC_r05/best.pth'
    else:
        ckpt_path = f'./checkpoints/step1_IEC+ICR_r05/best.pth'
    
    try:
        # 构建 opt
        import argparse
        parser = argparse.ArgumentParser()
        # 添加所有必需的参数（这里简化，实际应该用完整的 get_opt）
        opt_args = f'--datasetName sims {args}'.split()
        # 这里简化处理，实际需要完整参数
        opt = get_opt()
        
        results = load_and_eval(ckpt_path, opt)
        print(f"{name:<20} {results['MAE']:<12.6f} {results['Corr']:<8.4f} {results['Mult_acc_2']:<12.4f} {results['F1_score']:.4f}")
    except Exception as e:
        print(f"{name:<20} Error: {e}")

print("=" * 80)
PYEOF

    chmod +x logs/step1_ablation/step1_summary.py
    
    # 简单版：直接从日志提取最佳结果
    echo ""
    echo "📊 Quick Results (from logs, best epoch):"
    echo "----------------------------------------"
    
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r NAME _ <<< "$exp"
        LOG_FILE="logs/step1_ablation/train_${NAME}.log"
        
        if [ -f "$LOG_FILE" ]; then
            echo "Config: $NAME"
            # 提取最佳 MAE 对应的那一行（简化版）
            grep -A 3 "Best-MAE" "$LOG_FILE" | tail -4 || echo "  (训练中或未完成)"
            echo ""
        fi
    done
    
    echo "============================================================"
    echo "完整测试集评估（如需要）："
    echo "  python logs/step1_ablation/step1_summary.py"
    echo "============================================================"
}

# Main
case "${1:-train}" in
    train)
        run_train
        ;;
    summary)
        run_summary
        ;;
    *)
        run_train
        ;;
esac
