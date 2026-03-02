# Conflict-JS: 情感不一致建模框架

基于KuDA的二次开发，针对多模态情感分析中的情感不一致场景。

## 核心创新

1. **证据层JS**: 将JS从token级升级到证据集合级，避免被单个噪声token主导
2. **双证据保留**: 同时保留一致证据(互补融合)和冲突证据(冲突解析)，而非压制冲突
3. **冲突驱动路由**: 用冲突强度C控制融合策略(高冲突→强交互，低冲突→轻量融合)

## 与I²C和KuDA的区别

| 维度 | KuDA | I²C | Conflict-JS(本框架) |
|------|------|-----|---------------------|
| 控制信号 | 知识引导的senti_ratio | 一致性I²CS | 冲突强度C |
| 对冲突的态度 | 不显式建模 | 压制(删除不一致token) | 保留并利用 |
| JS作用 | 不使用JS | 正则损失+token压缩 | 路由信号 |
| 适用场景 | 一般MSA | 一般MSA | 情感不一致场景 |

## 分阶段运行

### Phase 1: Baseline (去KI + 音频CMVN)

```bash
python train.py --use_ki False --use_cmvn True
```

验证去掉知识注入后的基础性能。

### Phase 2: Conflict-JS (证据拆分 + Evidence-JS + 路由)

```bash
python train.py --use_ki False --use_cmvn True \
    --tau_conf 0.3 --tau_con 0.1 --tau_rel 0.5
```

核心Conflict-JS模块会自动启用。

### Phase 3: Token筛选 (视频token不一致保留+去冗余)

```bash
python train.py --use_ki False --use_cmvn True \
    --use_vision_pruning True \
    --vision_target_ratio 0.3 --vision_conf_ratio 0.5
```

启用视频token压缩，保留30%的token，其中一半是冲突证据。

## 运行实验评估

### 消融实验

```bash
# 1. Baseline
python train.py --use_ki False --use_cmvn True

# 2. +证据拆分 (ConflictJS启用但不做路由)
# 需要在代码中设置flag

# 3. Full Phase2
python train.py --use_ki False --use_cmvn True

# 4. Full Phase3  
python train.py --use_ki False --use_cmvn True --use_vision_pruning True
```

### 可视化冲突强度分布

```python
from evaluate_experiments import visualize_conflict_intensity_distribution

visualize_conflict_intensity_distribution(
    model, 
    test_loader, 
    device,
    save_path='./results/conflict_distribution.png'
)
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_conf` | 0.3 | 冲突证据阈值(情感差异) |
| `tau_con` | 0.1 | 一致证据阈值 |
| `tau_rel` | 0.5 | 置信度阈值 |
| `senti_num_classes` | 7 | 情感后验类别数 |
| `vision_target_ratio` | 0.3 | 视频token保留比例 |
| `vision_conf_ratio` | 0.5 | 冲突证据占比 |

## 预期结果

- 整体数据集: MAE持平或略优于KuDA baseline
- 不一致子集: MAE提升10%+ (相比baseline)
- 推理速度: 提升1.5-2x (Phase3开启token筛选后)
- 可解释性: 冲突强度C可量化不一致程度

## 文件结构

```
KuDA/
├── models/
│   ├── ConflictJS.py          # Phase 2: 核心Conflict-JS模块
│   ├── SentimentProjector.py  # Phase 2: 情感投影头
│   ├── VisionTokenPruner.py   # Phase 3: 视频token筛选
│   ├── Encoder_KIAdapter.py   # 修改: 添加情感投影
│   ├── DyRoutFusion_CLS.py    # 修改: 支持冲突驱动路由(待完成)
│   └── OverallModal.py         # 修改: 集成ConflictJS
├── core/
│   ├── dataset.py              # 修改: 添加CMVN
│   └── utils.py                # 修改: 添加不一致子集评估函数
├── train.py                    # 修改: KI可选加载
├── opts.py                     # 修改: 添加新参数
├── evaluate_experiments.py     # 新增: 实验评估脚本
└── CONFLICT_JS_README.md       # 新增: 本文档
```

## Citation

如果使用本框架，请引用:

```
@article{your_conflict_js,
  title={Conflict-JS: Evidence-Level Jensen-Shannon Divergence for Multimodal Sentiment Analysis with Emotional Inconsistency},
  author={...},
  year={2026}
}
```

以及原始KuDA工作:

```
@inproceedings{kuda2024,
  title={Knowledge-Guided Dynamic Modality Attention Fusion Framework for Multimodal Sentiment Analysis},
  author={...},
  booktitle={EMNLP},
  year={2024}
}
```
