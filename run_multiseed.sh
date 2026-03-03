#!/bin/bash
# ============================================================
# 多种子（5 seeds）批量训练 + mean±std 汇总
#
# 使用全配置 IEC+ICR_full（--use_conflict_js True + text_guided IEC）
# 在 5 个随机种子上训练，训练结束后自动打印 mean±std 到 stdout
#
# 用法:
#   bash run_multiseed.sh                          # 默认 50 epoch, GPU 2,3,4,5,6
#   bash run_multiseed.sh 40                       # 40 epoch
#   bash run_multiseed.sh 50 2,3,4,5,6            # 指定 GPU
#   bash run_multiseed.sh summary                  # 只打印已有实验汇总（不训练）
# ============================================================

CONDA_ENV="${CONDA_ENV:-kuda}"

if [ "$1" = "summary" ]; then
    CMD="summary"
    N_EPOCHS=50
    GPU_IDS="2,3,4,5,6"
else
    N_EPOCHS="${1:-50}"
    GPU_IDS="${2:-2,3,4,5,6}"
    CMD="train"
fi

IFS=',' read -ra GPUS <<< "$GPU_IDS"
N_GPUS=${#GPUS[@]}

SEEDS=(1111 2222 3333 4444 5555)
BASE_ARGS="--datasetName sims --use_cmvn True --use_ki False --n_epochs $N_EPOCHS"
BASE_ARGS="$BASE_ARGS --use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5"
BASE_ARGS="$BASE_ARGS --lambda_nce 0.1 --lambda_senti 0.05 --lambda_con 0.1 --lambda_cal 0.1"

mkdir -p logs/multiseed

run_train() {
    echo "============================================================"
    echo "Multi-seed Training: ${#SEEDS[@]} seeds, $N_EPOCHS epochs"
    echo "Config: IEC+ICR full (text_guided, r=0.5)"
    echo "GPUs: $GPU_IDS"
    echo "============================================================"
    echo ""

    PIDS=()
    for idx in "${!SEEDS[@]}"; do
        SEED="${SEEDS[$idx]}"
        GPU_ID="${GPUS[$((idx % N_GPUS))]}"
        CKPT_DIR="./checkpoints/multiseed_seed${SEED}"
        LOG_FILE="logs/multiseed/train_seed${SEED}.log"
        mkdir -p "$CKPT_DIR"

        echo "=== [Seed $SEED] GPU=$GPU_ID, ckpt=$CKPT_DIR ==="

        CUDA_VISIBLE_DEVICES=$GPU_ID conda run --no-capture-output -n $CONDA_ENV \
            python -u train.py $BASE_ARGS \
            --seed "$SEED" \
            --checkpoint_dir "$CKPT_DIR" \
            > "$LOG_FILE" 2>&1 &
        PIDS+=($!)
    done

    echo ""
    echo "All ${#SEEDS[@]} seeds launched. PIDs: ${PIDS[*]}"
    echo "Monitor: tail -f logs/multiseed/train_seed1111.log"
    echo ""
    echo "Waiting for all seeds to complete..."
    wait

    echo ""
    echo "All seeds finished. Computing mean±std..."
    echo ""
    print_summary
}

print_summary() {
    echo "============================================================"
    echo "Multi-seed Results Summary (IEC+ICR full)"
    echo "============================================================"
    printf "%-12s %8s %8s %8s %8s %8s %8s\n" "Seed" "MAE" "Corr" "Acc-2" "Acc-3" "Acc-5" "F1"
    echo "------------------------------------------------------------"

    declare -a ALL_MAE ALL_CORR ALL_ACC2 ALL_F1
    for SEED in "${SEEDS[@]}"; do
        LOG="logs/multiseed/train_seed${SEED}.log"
        if [ ! -f "$LOG" ]; then
            printf "%-12s %8s\n" "seed$SEED" "no log"
            continue
        fi
        # 从日志中提取 Best-MAE 的测试集结果（INFO 行在 tqdm 进度条之后，需足够 -A 行数）
        # 只取第一个 Best-MAE 块内的第一组指标，避免混入后面的 Best-Corr 块
        BLOCK=$(grep -A120 "Best-MAE Test Results" "$LOG" | head -100)
        MAE=$(echo "$BLOCK" | grep "MAE:" | head -1 | grep -oP '[\d.]+$')
        CORR=$(echo "$BLOCK" | grep "Corr:" | head -1 | grep -oP '[\d.]+$')
        ACC2=$(echo "$BLOCK" | grep "Acc-2:" | head -1 | grep -oP '[\d.]+$')
        ACC3=$(echo "$BLOCK" | grep "Acc-3:" | head -1 | grep -oP '[\d.]+$')
        ACC5=$(echo "$BLOCK" | grep "Acc-5:" | head -1 | grep -oP '[\d.]+$')
        F1=$(echo "$BLOCK" | grep "F1:" | head -1 | grep -oP '[\d.]+$')

        printf "%-12s %8s %8s %8s %8s %8s %8s\n" "seed$SEED" "${MAE:-N/A}" "${CORR:-N/A}" "${ACC2:-N/A}" "${ACC3:-N/A}" "${ACC5:-N/A}" "${F1:-N/A}"
        [ -n "$MAE" ] && ALL_MAE+=("$MAE")
        [ -n "$CORR" ] && ALL_CORR+=("$CORR")
        [ -n "$ACC2" ] && ALL_ACC2+=("$ACC2")
        [ -n "$F1" ] && ALL_F1+=("$F1")
    done

    echo "------------------------------------------------------------"
    # 用 python 计算 mean±std
    if command -v conda &>/dev/null; then
        conda run --no-capture-output -n $CONDA_ENV python - <<'PYEOF'
import sys, os, re, numpy as np

seeds = [1111, 2222, 3333, 4444, 5555]
metrics = {'MAE': [], 'Corr': [], 'Acc-2': [], 'F1': []}
metric_keys = {'MAE': 'MAE:', 'Corr': 'Corr:', 'Acc-2': 'Acc-2:', 'F1': 'F1:'}

for seed in seeds:
    log = f'logs/multiseed/train_seed{seed}.log'
    if not os.path.exists(log):
        continue
    # 日志中包含 tqdm 进度条和可能的非 UTF-8 字节，这里用 errors=\"ignore\" 做鲁棒解析
    with open(log, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # find Best-MAE Test Results block
    blocks = re.findall(r'Best-MAE Test Results.*?={10,}', content, re.DOTALL)
    if not blocks:
        continue
    block = blocks[-1]
    for key, pat in metric_keys.items():
        m = re.search(rf'{pat}\s*([\d.]+)', block)
        if m:
            metrics[key].append(float(m.group(1)))

print(f"\n{'Metric':<10} {'Mean':>8} {'Std':>8} {'N':>4}")
print('-' * 35)
for key, vals in metrics.items():
    if vals:
        arr = np.array(vals)
        print(f"{key:<10} {arr.mean():>8.4f} {arr.std():>8.4f} {len(arr):>4}")
    else:
        print(f"{key:<10} {'N/A':>8}")
PYEOF
    fi
    echo "============================================================"
}

case "$CMD" in
    train)
        run_train
        ;;
    summary)
        print_summary
        ;;
    *)
        echo "Usage: bash run_multiseed.sh [epochs] [gpu_ids]"
        exit 1
        ;;
esac
