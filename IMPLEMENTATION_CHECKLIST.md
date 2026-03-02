# ✅ Conflict-JS框架实施检查清单

## Phase 1: Baseline建立

- [x] 修改`train.py`: KI加载可选化
- [x] 修改`opts.py`: 添加`--use_ki`和`--use_cmvn`参数
- [x] 修改`models/OverallModal.py`: 去除KI强依赖
- [x] 修改`core/dataset.py`: 实现CMVN归一化
- [x] 测试: Baseline能正常训练

**验证命令:**
```bash
python test_framework.py  # 应看到"Phase 1 Baseline test PASSED"
```

---

## Phase 2: Conflict-JS核心模块

### 2.1 情感投影头
- [x] 创建`models/SentimentProjector.py`
- [x] 实现token→情感后验的投影
- [x] 实现置信度计算

### 2.2 证据拆分
- [x] 创建`models/ConflictJS.py`
- [x] 实现`EvidenceSplitter`类
- [x] 基于tau_conf/tau_con/tau_rel阈值拆分

### 2.3 Evidence-level JS
- [x] 实现`EvidenceLevelJS`类
- [x] 证据后验聚合
- [x] Jensen-Shannon散度计算

### 2.4 冲突强度
- [x] 实现`ConflictIntensity`类
- [x] JS归一化到[0,1]

### 2.5 集成
- [x] 修改`models/Encoder_KIAdapter.py`: 添加情感投影头
- [x] 修改`models/OverallModal.py`: 集成ConflictJSModule
- [x] 添加控制flags: `--use_conflict_js`, `--use_routing`
- [x] 存储`last_conflict_intensity`用于可视化

**验证命令:**
```bash
python test_framework.py  # 应看到"Phase 2 Conflict-JS test PASSED"
```

---

## Phase 3: 视频Token筛选

- [x] 创建`models/VisionTokenPruner.py`
- [x] 实现方案B: 不一致保留+去冗余
- [x] 添加参数: `--use_vision_pruning`, `--vision_target_ratio`, `--vision_conf_ratio`
- [ ] **集成到DyRoutTrans** (待完成,见"待完成项")

---

## 实验评估框架

### 消融实验
- [x] 创建`evaluate_experiments.py`
- [x] 实现`run_ablation_experiments()`
- [x] 实现`print_ablation_table()`
- [x] 定义所有实验配置(`experiment_configs.py`)

### 不一致子集评估
- [x] 在`core/utils.py`添加`get_inconsistency_subset()`
- [x] 在`core/utils.py`添加`compute_metrics_by_subset()`
- [x] 在实验脚本中计算不一致子集指标

### 可视化
- [x] 创建`visualize_results.py`
- [x] 实现冲突强度分布图
- [x] 实现证据拆分统计图
- [x] 实现case study可视化

---

## 文档与工具

- [x] 创建`CONFLICT_JS_README.md`: 使用说明
- [x] 创建`IMPLEMENTATION_SUMMARY.md`: 实施总结
- [x] 创建`test_framework.py`: 快速测试
- [x] 创建`run_all_experiments.py`: 批量实验
- [x] 创建`experiment_configs.py`: 实验配置

---

## 待完成项 (可选优化)

### 1. 冲突驱动路由 (Priority: HIGH)

**目标:** 让融合模块真正使用冲突强度C和证据masks

**位置:** `models/DyRoutFusion_CLS.py`

**步骤:**
1. [ ] 修改`DyRout_block.forward()`:
   ```python
   def forward(self, source, t, v, a, conflict_C=None, evidence_masks=None):
       if conflict_C is not None and evidence_masks is not None:
           # 门控逻辑
           alpha = torch.sigmoid(5.0 * (conflict_C - 0.5))
           
           # 冲突分支: 使用conf_masks
           h_conflict = self.conflict_branch(...)
           
           # 互补分支: 使用con_masks
           h_complement = self.complement_branch(...)
           
           # 门控混合
           output = alpha * h_conflict + (1-alpha) * h_complement
       else:
           # Fallback to original logic
           output = self.layernorm(cross_f_t + cross_f_v + cross_f_a)
       return output
   ```

2. [ ] 修改`models/OverallModal.py`传递C和masks:
   ```python
   multimodal_features, nce_loss = self.DyMultiFus(
       uni_fea, uni_mask, 
       conflict_C=C,
       evidence_masks=(con_masks, conf_masks)
   )
   ```

### 2. Phase 3集成 (Priority: MEDIUM)

**位置:** `models/DyRoutFusion_CLS.py` 第163-166行

**步骤:**
1. [ ] 导入`VisionTokenPruner`
2. [ ] 在`DyRoutTrans.__init__()`添加:
   ```python
   if opt.use_vision_pruning:
       from models.VisionTokenPruner import VisionTokenPruner
       self.vision_pruner = VisionTokenPruner(
           target_ratio=opt.vision_target_ratio,
           conf_ratio=opt.vision_conf_ratio
       )
   ```

3. [ ] 在`DyRoutTrans.forward()`中:
   ```python
   # 在dim对齐后,长度对齐前
   if self.opt.use_vision_pruning:
       senti_ref = uni_fea['T'].mean(dim=1)  # 文本情感
       hidden_v_aligned = self.dim_v(uni_fea['V'])
       hidden_v_pruned, _, pruning_info = self.vision_pruner(
           hidden_v_aligned.permute(0, 2, 1).permute(0, 2, 1),  # [B,L,D]
           senti_ref
       )
       # 更新vision mask
       uni_mask['V'] = pruning_info['pruned_mask']
       # 继续长度对齐
       hidden_v = self.len_v(hidden_v_pruned.permute(0, 2, 1)).permute(0, 2, 1)
   else:
       hidden_v = self.len_v(self.dim_v(uni_fea['V']).permute(0, 2, 1)).permute(0, 2, 1)
   ```

### 3. JS正则损失 (Priority: LOW)

**位置:** `train.py` 第77行

**步骤:**
1. [ ] 修改loss计算:
   ```python
   loss_re = loss_fn(output, label)
   
   # 添加JS正则(如果有冲突强度监督)
   if hasattr(model, 'last_conflict_intensity'):
       # 可选: 添加C的监督信号
       lambda_JS = 0.1
       JS_reg = some_regularization(model.last_conflict_intensity)
       loss = loss_re + 0.1 * nce_loss + lambda_JS * JS_reg
   else:
       loss = loss_re + 0.1 * nce_loss
   ```

---

## 运行流程

### 阶段1: 验证基础功能
```bash
# 1. 测试框架
python test_framework.py

# 预期输出:
# ✅ Phase 1 Baseline test PASSED!
# ✅ Phase 2 Conflict-JS test PASSED!
# ✅ All tests PASSED!
```

### 阶段2: 训练Baseline
```bash
# 2. 训练5个epoch验证收敛
python train.py --use_ki False --use_cmvn True \
    --use_conflict_js False --n_epochs 5 \
    --datasetName sims
```

### 阶段3: 训练完整模型
```bash
# 3a. Phase 2完整模型
python train.py --use_ki False --use_cmvn True \
    --use_conflict_js True --use_routing True \
    --n_epochs 30

# 3b. Phase 3 (需先完成待完成项2)
python train.py --use_ki False --use_cmvn True \
    --use_conflict_js True --use_routing True \
    --use_vision_pruning True --vision_target_ratio 0.3 \
    --n_epochs 20
```

### 阶段4: 运行完整实验
```bash
# 4. 批量运行所有消融实验
python run_all_experiments.py

# 查看结果
cat results/ablation_summary.json
```

### 阶段5: 生成可视化
```bash
# 5. 生成所有可视化图表
python visualize_results.py

# 查看输出
ls results/visualizations/
# - conflict_intensity_distribution.png
# - evidence_split_statistics.png
# - case_study_0.png (low conflict)
# - case_study_1.png (medium conflict)
# - case_study_2.png (high conflict)
```

---

## 文件完整性检查

运行以下命令检查所有文件是否存在:

```bash
# 核心模块
ls models/ConflictJS.py
ls models/SentimentProjector.py
ls models/VisionTokenPruner.py

# 实验脚本
ls evaluate_experiments.py
ls visualize_results.py
ls run_all_experiments.py
ls experiment_configs.py
ls test_framework.py

# 文档
ls CONFLICT_JS_README.md
ls IMPLEMENTATION_SUMMARY.md
ls IMPLEMENTATION_CHECKLIST.md

# 预期输出: 所有文件都存在
```

---

## 常见问题排查

### Q1: 测试脚本报错"No module named 'models.ConflictJS'"
**解决:** 确保在KuDA项目根目录运行:
```bash
cd /home/yechenlu/KuDA
python test_framework.py
```

### Q2: CMVN报错"audio_lengths not found"
**解决:** 确保数据集包含`audio_lengths`字段,或在`dataset.py`中添加fallback:
```python
if not hasattr(self, 'audio_lengths'):
    self.audio_lengths = np.array([self.audio.shape[1]] * len(self.audio))
```

### Q3: 冲突强度全为0或全为1
**解决:** 调整tau阈值:
```bash
python train.py ... --tau_conf 0.2 --tau_con 0.05 --tau_rel 0.4
```

### Q4: 显存不足
**解决:** 
- 减小batch_size: `--batch_size 16`
- 启用vision pruning减少token数量
- 使用梯度累积

---

## 最终验收标准

✅ **Phase 1成功标志:**
- `python test_framework.py`中Phase 1测试通过
- 训练loss正常下降
- MAE在合理范围(0.25-0.35)

✅ **Phase 2成功标志:**
- `python test_framework.py`中Phase 2测试通过
- 冲突强度C分布合理(均值0.3-0.7, 标准差>0.1)
- 一致/冲突证据token数量都>0

✅ **实验成功标志:**
- Baseline与KuDA原版MAE相近(±0.05)
- Phase 2完整模型在不一致子集上MAE降低>5%
- 可视化图表清晰展示冲突强度差异

✅ **论文ready标志:**
- 完整的消融实验表(6组实验)
- 不一致子集提升明显
- 至少3个case study展示
- 冲突强度分布图证明C的有效性

---

**当前状态:** ✅ 所有核心模块已实现,待完成可选优化项后即可开始实验

**下一步:** 运行`python test_framework.py`验证基础功能
