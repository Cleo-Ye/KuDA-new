# Sweep 较优配置与可视化

## 较优配置（按 MAE 与 Corr 综合）

| 排名 | run_name | MAE | Corr | 参数摘要 |
|------|----------|-----|------|----------|
| 1 | lr2e-5_bl1e-5_bs32_sd2024_pl2_... | **0.4112** | **0.6000** | lr=2e-5, bs=32, seed=2024, path_layers=2 |
| 2 | lr5e-5_bl1e-5_bs32_sd2024_pl3_... | 0.4118 | 0.5878 | lr=5e-5, bs=32, seed=2024, path_layers=3 |
| 3 | lr5e-6_bl1e-5_bs32_sd2024_pl3_... | 0.4155 | 0.5925 | lr=5e-6, bs=32, seed=2024, path_layers=3 |
| 4 | lr1e-6_bl1e-5_bs32_sd2024_pl3_... | 0.4153 | 0.5766 | lr=1e-6, bs=32, seed=2024, path_layers=3 |
| 5 | lr5e-5_bl1e-5_bs16_sd1111_pl3_... | 0.4136 | 0.5909 | lr=5e-5, bs=16, seed=1111, path_layers=3 |

## 1. 训练曲线可视化（plot_valid_curve.py）

无需模型，直接读取 valid_curve.csv：

```bash
cd /home/yechenlu/KuDA

# 最佳配置 (MAE 0.4112, Corr 0.6000)
python plot_valid_curve.py --csv checkpoints/sweep_unified/lr2e-5_bl1e-5_bs32_sd2024_pl2_la5e-2_lo3e-3_lrk5e-2_lpid2e-1_lcls3e-1_lt15e-2_lt25e-1_loss/valid_curve.csv --out_dir checkpoints/sweep_unified/best_vis

# 第二优
python plot_valid_curve.py --csv checkpoints/sweep_unified/lr5e-5_bl1e-5_bs32_sd2024_pl3_la5e-2_lo3e-3_lrk5e-2_lpid2e-1_lcls3e-1_lt15e-2_lt25e-1_loss/valid_curve.csv --out_dir checkpoints/sweep_unified/2nd_vis

# 第三优（Corr 最高之一）
python plot_valid_curve.py --csv checkpoints/sweep_unified/lr5e-6_bl1e-5_bs32_sd2024_pl3_la5e-2_lo3e-3_lrk5e-2_lpid2e-1_lcls3e-1_lt15e-2_lt25e-1_loss/valid_curve.csv --out_dir checkpoints/sweep_unified/3rd_vis
```

## 2. 高冲突样本评估（evaluate_high_conflict_pid.py）

**注意**：sweep 默认 `--no_save_model` 不保存 best.pth。要评估高冲突子集，需先用最佳配置重训（train.py 默认会保存模型）：

```bash
# 用最佳配置重训（默认保存 best.pth）
python train.py --datasetName sims --model_type pid_dualpath --use_batch_pid_prior True \
  --lr 2e-5 --bert_lr 1e-5 --batch_size 32 --seed 2024 --path_layers 2 \
  --checkpoint_dir ./checkpoints/sweep_unified/best_retrain

# 评估高冲突子集
python scripts/evaluate_high_conflict_pid.py --ckpt_dir ./checkpoints/sweep_unified/best_retrain --threshold 0.4
```

高冲突定义：`max(|y_T - y_V|, |y_T - y_A|) > 0.4`，即文本与视觉/音频模态标签分歧较大的样本。

## 3. 最佳配置完整参数（用于重训）

```bash
python train.py \
  --datasetName sims --model_type pid_dualpath --use_batch_pid_prior True \
  --lr 2e-5 --bert_lr 1e-5 --batch_size 32 --seed 2024 --path_layers 2 \
  --lambda_aux 0.05 --lambda_ortho 0.0025 --lambda_rank 0.05 \
  --lambda_pid 0.2 --lambda_classification 0.3 \
  --lambda_task_stage1 0.05 --lambda_task_stage2 0.5 \
  --loss_fn smoothl1 --grad_clip 0.5 \
  --checkpoint_dir ./checkpoints/sweep_unified/best_retrain
```

（train.py 默认保存模型；sweep 脚本才传 --no_save_model 以省磁盘）
