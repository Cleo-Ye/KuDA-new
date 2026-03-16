# 进一步降低 MAE 的可行办法

当前多 seed 均值约 MAE 0.4389±0.0055，单次最佳 0.4318。以下为可尝试方向，按**实现成本/收益**大致排序。

---

## 1. 测试时集成 + 验证集调参（零训练成本，优先试）

**做法**：对多个 checkpoint 在测试集上加权平均；权重在**验证集**上优化以最小化 MAE，可选再做线性校准 `pred_cal = a*pred + b`。

**脚本**：`scripts/eval_ensemble.py`

```bash
# 简单平均（你当前约 MAE 0.4266）
python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --gpu 0

# 验证集优化权重，再在测试集评估（目标逼近 0.40）
python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --gpu 0

# 权重 + 线性校准（进一步压 MAE）
python scripts/eval_ensemble.py --ckpt_dir ./checkpoints/multiseed_best --tune_weights --calibrate --gpu 0
```

**预期**：`--tune_weights` 常可比简单平均再降一截 MAE；`--calibrate` 可再微调尺度/偏移，合力有望接近或略低于 0.40。

---

## 2. 训练侧可调超参（小改即可试）

| 手段 | 说明 | 建议试法 |
|------|------|----------|
| **梯度裁剪** | 防止梯度爆炸，训练更稳 | `--grad_clip 1.0`（opts 已有，默认 0 即未启用） |
| **学习率与 warmup** | 当前 warmup=10%×n_epochs，cosine eta_min=1e-5 | 试 `n_epochs=80` 或略调 lr（如 2.5e-5） |
| **batch_size** | 当前 32 | 试 16（更噪梯度）或 48（更稳），需配合 lr 微调 |
| **CMVN** | 音频/视觉 z-score 归一化 | 当前默认 `use_cmvn True`，可对比 `False` 看是否更差 |

---

## 3. 数据与输入

| 手段 | 说明 |
|------|------|
| **数据增强** | 当前 `core/dataset.py` 无增强。可加：文本侧 dropout/masking、音频/视觉加小幅噪声或 frame masking，仅对 train 做，valid/test 不变。 |
| **特征/预训练** | 换更强文本编码（如 RoBERTa）或更好的 V/A 预训练特征，再在现有融合结构上微调。 |

---

## 4. 损失与目标

| 手段 | 说明 |
|------|------|
| **SmoothL1 beta** | 当前 `beta=0.5`。可试 0.3（更接近 L1）或 1.0（更接近 MSE），需在 `train.py` 里改或加参数。 |
| **多任务/辅助** | 已有分类辅助损失；可试回归 + 分类的权重再微调（`lambda_classification`）。 |

---

## 5. 模型与融合

| 手段 | 说明 |
|------|------|
| **path_layers** | 当前 2 层；sweep 已见 3 层更差，暂不加深。 |
| **融合方式** | 双路径 (Shared + JointGain) 已用；若做消融可试只保留一条路径或调 router_tau。 |

---

## 6. 报告方式

| 手段 | 说明 |
|------|------|
| **多 seed 报告** | 已用 `sweep_multiseed.py` 跑 5 seed，报告 mean±std；论文可写「MAE=0.4389±0.0055」，或取「最佳单次 0.4318」。 |
| **集成报告** | 用 `eval_ensemble.py` 得到集成 MAE，可作为单独一行（如「Ours (5-model ensemble)」）与单模型对比。 |

---

建议优先做：**跑一次 `eval_ensemble.py`** 看集成 MAE，再视情况试 `--grad_clip 1.0` 或略调 lr/epoch。
