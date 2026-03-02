# KuDA IEC+ICR 代码改进总结

## 问题诊断

根据实验结果分析，原始代码存在以下问题：

1. **文本引导不够强** - `TextGuidedVisionPruner` 中 `|senti_t|` 权重范围太小，导致pruning效果不明显
2. **冲突检测不够敏感** - gate_k=10 导致sigmoid饱和，gate_tau=0.15 阈值过高
3. **证据分割不够激进** - conf_ratio=0.25 和 con_ratio=0.25 选择的证据token太少
4. **损失权重不平衡** - 冲突相关的loss (js/con/cal) 权重被稀释
5. **融合策略过于简单** - 双分支直接相加，没有考虑全局冲突强度

## 核心改进

### 1. 增强文本引导机制 (`VisionTokenPruner.py`)

**改动**：
```python
# 原始: w_t = |senti_t| * mask_t  # 范围 [0, 3]
# 改进: 归一化 + 平方根变换 + 基础权重
senti_abs = torch.abs(senti_t).float()
senti_norm = (senti_abs - senti_abs.min()) / (senti_abs.max() - senti_abs.min() + 1e-8)
w_t = (senti_norm ** 0.5) * mask_t.float()  # 范围 [0, 1]
w_t = w_t + 0.1  # 添加基础权重避免全0
```

**效果**：
- 增强文本情感强度对vision token选择的影响
- 归一化保证跨样本权重一致性
- 平方根变换保持平滑性，避免过度极端化

### 2. 改进冲突检测参数 (`ConflictJS.py`)

**改动**：
```python
# 原始配置
conf_ratio=0.25, con_ratio=0.25, rel_min=0.20

# 改进配置
conf_ratio=0.30,  # 提升到0.30，选更多冲突证据
con_ratio=0.30,   # 提升到0.30，选更多一致证据  
rel_min=0.15      # 降低到0.15，包含更多中等可靠度token
```

**效果**：
- 增加有效证据token数量，避免信息丢失
- 降低可靠度阈值，包含更多有用的中等置信度信息

### 3. 优化门控函数 (`DyRoutFusion_CLS.py`)

**改动1 - 门控参数**：
```python
# 原始: gate_k=10.0, gate_tau=0.15
# 改进: gate_k=5.0, gate_tau=0.08

gate_k = 5.0    # 降低增益，使门控更线性，避免饱和
gate_tau = 0.08 # 降低阈值，更容易触发冲突分支
```

**改动2 - 门控计算**：
```python
# 原始: gate_m = C_m * rho_m
# 改进: gate_m = C_m * sqrt(rho_m)  # 密度影响更平滑

gate_t = conflict_C_m['T'] * torch.sqrt(conflict_rho_m['T'].clamp(min=1e-6))
```

**效果**：
- 门控函数更线性，避免在高冲突时饱和到1
- 更低的阈值使中等冲突样本也能激活冲突分支
- 平方根变换使密度影响更平滑

### 4. 改进双分支融合策略

**改动**：
```python
# 原始: 简单相加
output = h_conf + h_con

# 改进: 残差连接 + 动态加权
h_conf = layernorm(alpha_t * cross_conf_t + alpha_v * cross_conf_v + alpha_a * cross_conf_a + 0.1 * source)
h_con = layernorm((1-alpha_t) * cross_con_t + (1-alpha_v) * cross_con_v + (1-alpha_a) * cross_con_a + 0.1 * source)

# 根据全局冲突强度C动态调整两分支权重
conf_weight = sigmoid(gate_k * (C - gate_tau))
output = conf_weight * h_conf + (1 - conf_weight) * h_con
```

**效果**：
- 添加残差连接避免分支退化
- 高冲突样本加强冲突分支，低冲突样本平衡两分支

### 5. 调整损失权重平衡 (`train.py`)

**改动**：
```python
# 原始权重
lambda_nce=0.1, lambda_senti=0.05, lambda_js=0.1, lambda_con=0.1, lambda_cal=0.1

# 改进权重
lambda_nce=0.08,    # 降低: NCE不是核心贡献
lambda_senti=0.03,  # 降低: 辅助监督，不应喧宾夺主
lambda_js=0.15,     # 提升: 冲突保持是核心机制
lambda_con=0.12,    # 提升: 一致性对齐很重要
lambda_cal=0.12     # 提升: 校准损失直接对齐真实冲突
```

**效果**：
- 强化冲突路由相关的loss，使其不被主任务loss稀释
- NCE和senti降权，避免喧宾夺主

### 6. 添加温度缩放校准 (新增)

**新增模块**: `TemperatureScaling.py`

**功能**：
- 训练后在验证集上拟合最优温度T
- 校准后验概率分布，使置信度更准确
- 计算ECE (Expected Calibration Error) 评估校准效果

**使用**：
```bash
python evaluate_calibration.py --datasetName sims --use_conflict_js True
```

## 预期效果提升

基于改进原理，预期在CH-SIMS上的提升：

### 定量指标
- **MAE**: 从 0.3282 → **0.31-0.32**（目标提升 0.008-0.018）
- **Corr**: 从 0.6847 → **0.69-0.70**（目标提升 0.005-0.015）
- **Acc-2**: 预期提升 1-2%
- **ECE**: 预期降低 15-25%（校准改善）

### 机制理解
改进后的模型：
1. **文本引导更有效** - Vision token筛选更准确，保留情感相关帧
2. **冲突检测更敏感** - 中等冲突样本也能被正确路由
3. **证据利用更充分** - 30% vs 25% 的证据选择率
4. **分支融合更智能** - 根据冲突强度动态调整
5. **损失更平衡** - 冲突机制得到足够训练

## 使用建议

### 完整实验流程

1. **Baseline (无IEC/ICR)**
```bash
python train.py --datasetName sims --use_ki False --use_conflict_js False --use_vision_pruning False
```

2. **+IEC only**
```bash
python train.py --datasetName sims --use_ki False --use_conflict_js False --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5
```

3. **+ICR only**
```bash
python train.py --datasetName sims --use_ki False --use_conflict_js True --use_vision_pruning False
```

4. **IEC+ICR full (改进版)**
```bash
python train.py --datasetName sims --use_ki False --use_conflict_js True --use_vision_pruning True --iec_mode text_guided --vision_keep_ratio 0.5
```

### 关键超参数消融

**Vision keep ratio** (主图):
```bash
for r in 0.3 0.4 0.5 0.6 0.8; do
    python train.py --vision_keep_ratio $r --seed 0
done
```

**Gate threshold** (门控敏感度):
```bash
for tau in 0.05 0.08 0.10 0.12; do
    python train.py --gate_tau $tau --seed 0
done
```

**Loss weight** (js权重):
```bash
for w in 0.10 0.15 0.20; do
    python train.py --lambda_js $w --seed 0
done
```

### 多种子实验
```bash
for seed in 0 1 2 3 4; do
    python train.py --seed $seed
done
```

## 代码质量改进

### 新增功能
- ✅ 温度缩放校准模块 (`TemperatureScaling.py`)
- ✅ ECE评估脚本 (`evaluate_calibration.py`)
- ✅ 改进的pruning权重计算
- ✅ 动态双分支融合

### 改进的鲁棒性
- ✅ 权重归一化避免数值不稳定
- ✅ 残差连接防止梯度消失
- ✅ clamp和epsilon避免除零
- ✅ 更平滑的门控函数

## 论文写作要点

### 主要贡献陈述
1. "我们提出了**文本引导的视觉证据压缩**，通过对齐注意力选择情感相关帧并合并冗余信息"
2. "我们设计了**对齐感知的冲突参照**，将token级冲突定义为与对齐后文本局部参照的差异"
3. "我们引入了**冲突强度与密度联合门控**，仅当强度大且密度高时才激活冲突分支"
4. "我们通过**温度缩放校准**改善概率校准，使冲突度量更可靠"

### 实验必备图表
1. **Main Results**: 4数据集×5seeds，报告 mean±std
2. **IEC/ICR Ablation**: baseline / +IEC / +ICR / full
3. **r-Performance Curve**: vision_keep_ratio vs MAE/Corr
4. **Gate Behavior**: alpha vs (C, rho) 散点图
5. **Calibration Plot**: ECE before/after TS
6. **Conflict Binning**: 按C分桶，展示高冲突增益

### 审稿人防守
- "JSD只是稳定实例化，核心贡献是对齐感知冲突定义"
- "IEC降低冗余稳定ICR估计，并提供效率曲线"
- "5 seeds mean±std + ECE + 冲突分桶证明结构性收益"

## 下一步工作

### 短期 (1-2天)
- [x] 运行改进后的full模型，验证MAE提升
- [ ] 绘制r-performance曲线
- [ ] 运行5 seeds实验，计算mean±std
- [ ] ECE评估并绘制calibration plot

### 中期 (1周)
- [ ] 在MOSEI上验证泛化性
- [ ] 完成所有消融实验
- [ ] 准备论文图表和表格

### 长期 (2周+)
- [ ] 撰写论文初稿
- [ ] 准备代码开源
- [ ] 补充文档和README

## 改进前后对比

| 模块 | 原始实现 | 改进后 | 改进效果 |
|-----|---------|--------|---------|
| 文本引导 | `w_t = |senti_t|` | 归一化+sqrt变换+基础权重 | 权重范围更合理 |
| 证据选择 | 25% conf/con | 30% conf/con | 包含更多有效证据 |
| 可靠度阈值 | rel_min=0.20 | rel_min=0.15 | 利用中等置信度token |
| 门控增益 | gate_k=10 | gate_k=5 | 避免饱和，更线性 |
| 门控阈值 | gate_tau=0.15 | gate_tau=0.08 | 更容易触发 |
| 密度影响 | gate=C*rho | gate=C*sqrt(rho) | 更平滑 |
| 分支融合 | 简单相加 | 残差+动态加权 | 更智能 |
| JS权重 | 0.10 | 0.15 | 强化冲突保持 |
| 校准损失 | 0.10 | 0.12 | 更强对齐 |

## 技术细节说明

### 为什么降低gate_k?
- gate_k=10时，sigmoid(10*(x-0.15))在x>0.2时就接近1，失去区分度
- gate_k=5时，sigmoid保持在0.2-0.8的线性区间更长，更敏感

### 为什么提升conf_ratio?
- 25%可能只选中5-10个token，信息量不足
- 30%约15-18个token，既避免噪声又保留足够信息

### 为什么使用sqrt(rho)?
- rho=0.1 vs 0.5，线性差5倍，但实际影响不应相差这么多
- sqrt后差异缩小到2.2倍，更合理

### 为什么提升lambda_js?
- JS loss鼓励冲突证据保持差异，不被压制
- 原始0.1可能被主任务loss淹没
- 提升到0.15使冲突机制得到足够梯度

---

**最后更新**: 2026-03-02  
**改进版本**: v2.1 (基于实验结果的针对性优化)
