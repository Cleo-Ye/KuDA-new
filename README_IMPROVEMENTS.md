# KuDA 项目代码全面改进总结

## 📊 问题诊断

根据实验结果分析，当前代码的主要问题：

1. **IEC+ICR_full vs Baseline**: MAE 从 0.3456 降到 0.3282，提升仅 **0.0174** (5%)
2. **Corr提升**: 从 0.6716 到 0.6847，提升 **0.0131** (1.9%)
3. **效果不够显著**，需要针对性改进

## 🎯 核心改进策略

### 1. 增强文本引导机制 (⭐⭐⭐)
**问题**: `|senti_t|` 权重范围太小，跨样本不一致  
**解决**: 归一化 + 平方根变换 + 基础权重  
**预期提升**: MAE +0.003

### 2. 更激进的证据选择 (⭐⭐⭐)
**问题**: 25% 选择率太保守，信息量不足  
**解决**: conf_ratio 和 con_ratio 提升到 30%，rel_min 降到 0.15  
**预期提升**: MAE +0.005

### 3. 优化门控函数 (⭐⭐⭐)
**问题**: gate_k=10 导致饱和，gate_tau=0.15 阈值太高  
**解决**: gate_k 降到 5，gate_tau 降到 0.08，使用 sqrt(rho)  
**预期提升**: MAE +0.006

### 4. 改进融合策略 (⭐⭐)
**问题**: 简单相加，分支可能退化  
**解决**: 残差连接 + 根据全局C动态加权两分支  
**预期提升**: MAE +0.003

### 5. 平衡损失权重 (⭐⭐)
**问题**: 冲突相关loss被稀释  
**解决**: 提升 lambda_js/con/cal，降低 lambda_nce/senti  
**预期提升**: MAE +0.002

### 6. 添加温度缩放校准 (⭐)
**新增功能**: 训练后校准 + ECE评估  
**预期效果**: ECE 降低 15-25%

---

## 📝 修改文件列表

| 文件 | 改动行数 | 改动类型 | 重要性 |
|-----|---------|---------|--------|
| `models/VisionTokenPruner.py` | 15行 | 增强文本引导 | ⭐⭐⭐ |
| `models/ConflictJS.py` | 10行 | 证据选择参数 | ⭐⭐⭐ |
| `models/DyRoutFusion_CLS.py` | 30行 | 门控+融合策略 | ⭐⭐⭐ |
| `train.py` | 8行 | 损失权重调整 | ⭐⭐ |
| `opts.py` | 15行 | 默认超参数 | ⭐ |
| `models/TemperatureScaling.py` | 全新 | 校准模块 | ⭐⭐ |
| `evaluate_calibration.py` | 全新 | 评估脚本 | ⭐ |
| `run_improved_experiments.sh` | 全新 | 实验脚本 | ⭐ |
| `analyze_improvements.py` | 全新 | 分析工具 | ⭐ |

---

## 🚀 预期效果

### 定量目标
- **MAE**: 0.3282 → **0.310-0.320** (目标提升 **0.008-0.018**)
- **Corr**: 0.6847 → **0.690-0.700** (目标提升 **0.005-0.015**)
- **Acc-2**: 提升 **1-2%**
- **ECE**: 降低 **15-25%**

### 提升幅度对比
| 配置 | 原始MAE | 预期MAE | 提升幅度 |
|-----|---------|---------|---------|
| Baseline | 0.3456 | - | - |
| +IEC only | 0.3325 | 0.325 | ✅ 小幅提升 |
| +ICR only | 0.3422 | 0.332 | ✅ 小幅提升 |
| **IEC+ICR full** | **0.3282** | **0.310-0.320** | ✅✅ **显著提升** |

---

## 💡 关键参数对比

| 参数 | 原始值 | 改进值 | 改进理由 |
|-----|-------|--------|---------|
| **gate_k** | 10.0 | **5.0** | 避免sigmoid饱和 |
| **gate_tau** | 0.15 | **0.08** | 更容易触发冲突分支 |
| **conf_ratio** | 0.25 | **0.30** | 选择更多冲突证据 |
| **con_ratio** | 0.25 | **0.30** | 选择更多一致证据 |
| **rel_min** | 0.20 | **0.15** | 包含中等可靠度token |
| **lambda_js** | 0.10 | **0.15** | 强化冲突保持 |
| **lambda_con** | 0.10 | **0.12** | 强化一致性 |
| **lambda_cal** | 0.10 | **0.12** | 强化校准 |

---

## 🔧 使用方法

### 快速测试改进效果
```bash
# 一键运行所有对比实验
chmod +x run_improved_experiments.sh
./run_improved_experiments.sh

# 分析结果
python analyze_improvements.py

# 评估校准
python evaluate_calibration.py --datasetName sims --use_conflict_js True
```

### 单独运行改进后的full模型
```bash
python train.py \
    --datasetName sims \
    --seed 0 \
    --use_ki False \
    --use_conflict_js True \
    --use_vision_pruning True \
    --iec_mode text_guided \
    --vision_keep_ratio 0.5
```

---

## 📈 验证计划

### 第一阶段：验证单点改进
- [ ] 运行改进后的full模型 (seed=0)
- [ ] 检查MAE是否达到 0.310-0.320
- [ ] 检查各项loss数值是否合理
- [ ] 可视化gate行为 (alpha分布)

### 第二阶段：多种子稳定性
- [ ] 运行5个随机种子 (0,1,2,3,4)
- [ ] 计算 mean ± std
- [ ] 确认提升具有统计显著性

### 第三阶段：完整实验
- [ ] 绘制 r-performance 曲线
- [ ] ECE校准曲线
- [ ] 冲突分桶分析
- [ ] 准备论文图表

---

## ⚠️ 注意事项

### 如果效果不佳
1. **检查loss数值**: 确保 js_loss, con_loss, cal_loss 不是 NaN 或过大
2. **检查gate行为**: alpha 应该在 [0.2, 0.8] 范围有分布，不应全是0或1
3. **检查证据数量**: conf_mask 和 con_mask 的 token 数量应该合理 (约15-20个)
4. **逐项回滚**: 可以选择性回滚某些改进，找出最有效的组合

### 调试命令
```python
# 在 OverallModal.py 的 forward 中添加打印
print(f"JS loss: {js_loss.item():.4f}")
print(f"Con loss: {con_loss.item():.4f}")
print(f"Cal loss: {cal_loss.item():.4f}")
print(f"Conflict C: {C.mean().item():.4f}")
print(f"Alpha_T: {alpha_t.mean().item():.4f}")
print(f"Conf token count: {conf_masks['T'].sum(dim=1).float().mean().item():.1f}")
```

---

## 📚 相关文档

- **详细改进说明**: `IMPROVEMENTS.md`
- **改动清单**: `CHANGELOG.md`
- **设计方案**: `.cursor/plans/kuda_高含金量升级方案_ea5bf216.plan.md`
- **深度分析**: `deep-improve.md`

---

## ✅ 改进状态

- [x] 分析当前实验效果问题
- [x] 修复SentimentProjector (已确认正确)
- [x] 增强TextGuidedVisionPruner的文本引导机制
- [x] 改进ConflictJS的冲突检测强度和门控函数
- [x] 优化DyRoutFusion的双分支融合策略
- [x] 调整loss权重使各项损失更平衡
- [x] 添加温度缩放校准和ECE评估
- [x] 创建实验脚本和分析工具
- [x] 完善文档

**状态**: ✅ **所有改进已完成，等待实验验证**

---

**最后更新**: 2026-03-02  
**版本**: v2.1  
**作者**: Cursor AI Assistant
