#!/bin/bash
# KuDA 改进版本快速测试脚本

echo "================================"
echo "KuDA IEC+ICR 改进版本测试"
echo "================================"

DATASET="sims"
SEED=0
EPOCHS=50

echo ""
echo "[1/4] 测试 Baseline (无IEC/ICR)..."
python train.py \
    --datasetName $DATASET \
    --seed $SEED \
    --n_epochs $EPOCHS \
    --use_ki False \
    --use_conflict_js False \
    --use_vision_pruning False \
    --checkpoint_dir ./checkpoints/baseline_seed${SEED}

echo ""
echo "[2/4] 测试 +IEC only..."
python train.py \
    --datasetName $DATASET \
    --seed $SEED \
    --n_epochs $EPOCHS \
    --use_ki False \
    --use_conflict_js False \
    --use_vision_pruning True \
    --iec_mode text_guided \
    --vision_keep_ratio 0.5 \
    --checkpoint_dir ./checkpoints/iec_only_seed${SEED}

echo ""
echo "[3/4] 测试 +ICR only..."
python train.py \
    --datasetName $DATASET \
    --seed $SEED \
    --n_epochs $EPOCHS \
    --use_ki False \
    --use_conflict_js True \
    --use_vision_pruning False \
    --checkpoint_dir ./checkpoints/icr_only_seed${SEED}

echo ""
echo "[4/4] 测试 IEC+ICR full (改进版)..."
python train.py \
    --datasetName $DATASET \
    --seed $SEED \
    --n_epochs $EPOCHS \
    --use_ki False \
    --use_conflict_js True \
    --use_vision_pruning True \
    --iec_mode text_guided \
    --vision_keep_ratio 0.5 \
    --use_alignment_ref True \
    --conflict_metric js \
    --checkpoint_dir ./checkpoints/iec_icr_full_seed${SEED}

echo ""
echo "================================"
echo "所有测试完成!"
echo "================================"
echo ""
echo "查看结果:"
echo "  - Baseline:     ./checkpoints/baseline_seed${SEED}/"
echo "  - +IEC only:    ./checkpoints/iec_only_seed${SEED}/"
echo "  - +ICR only:    ./checkpoints/icr_only_seed${SEED}/"
echo "  - IEC+ICR full: ./checkpoints/iec_icr_full_seed${SEED}/"
echo ""
echo "评估校准效果:"
echo "  python evaluate_calibration.py --datasetName $DATASET --use_conflict_js True"
