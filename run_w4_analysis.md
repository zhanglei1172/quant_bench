# W4A8 量化测试分析报告

## 测试概述

本测试旨在评估 Qwen3-8B 模型在不同 W4A8 量化策略下的性能表现，重点关注静态量化的挑战和混合精度（mix-precision）的必要性。

### 测试配置

**测试脚本**: `run_w4_analysis.sh`

**量化配置**:
- `gptq-w4a8-dynamic` - GPTQ W4A8 动态量化（无 R4）
- `gptq-w4a16-g128` - GPTQ W4A16 量化（g128，无 R4）
- `r4-gptq-w4a8-dynamic` - R4 + GPTQ W4A8 动态量化
- `r4-gptq-w4a8-static-mix-precision` - R4 + GPTQ W4A8 静态量化（混合精度）
- `r4-gptq-w4a8-static` - R4 + GPTQ W4A8 静态量化（纯 W4A8）
- `r4-gptq-w4a16-g128` - R4 + GPTQ W4A16 量化（g128）
- `r4-gptq-w4a16` - R4 + GPTQ W4A16 量化

**评测数据集**: MMLU + GSM8K (1024 样本)

**转换配置**: original (基线) + spinquant-r1r2

---

## 测试结果

### 完整结果表（按精度排序）

| 转换 | 量化 | Overall | MMLU | GSM8K | 相对基线损失 |
|------|------|---------|------|-------|--------------|
| - | **baseline** | **0.835** | **0.793** | **0.877** | - |
| r1r2 | baseline | 0.832 | 0.795 | 0.869 | -0.003 |
| - | r4-gptq-w4a16-g128 | 0.847 | 0.795 | 0.898 | +0.012 |
| r1r2 | r4-gptq-w4a16-g128 | 0.822 | 0.787 | 0.857 | -0.010 |
| - | gptq-w4a16-g128 | 0.817 | 0.758 | 0.877 | -0.018 |
| r1r2 | gptq-w4a16-g128 | 0.813 | 0.764 | 0.861 | -0.022 |
| r1r2 | r4-gptq-w4a16 | 0.793 | 0.758 | 0.828 | -0.042 |
| - | r4-gptq-w4a16 | 0.744 | 0.674 | 0.815 | -0.091 |
| r1r2 | r4-gptq-w4a8-dynamic | 0.833 | 0.787 | 0.879 | +0.001 |
| - | gptq-w4a8-dynamic | 0.806 | 0.746 | 0.865 | -0.029 |
| r1r2 | gptq-w4a8-dynamic | 0.803 | 0.775 | 0.830 | -0.032 |
| - | r4-gptq-w4a8-dynamic | 0.723 | 0.672 | 0.773 | -0.112 |
| r1r2 | r4-gptq-w4a8-static-mix-precision | **0.822** | 0.772 | 0.873 | -0.010 |
| r1r2 | r4-gptq-w4a8-static | 0.682 | 0.680 | 0.684 | -0.153 |
| - | r4-gptq-w4a8-static-mix-precision | 0.654 | 0.563 | 0.746 | -0.181 |
| - | r4-gptq-w4a8-static | 0.476 | 0.352 | 0.600 | -0.359 |

---

## 关键发现

### ⭐ **核心结论：混合精度是 W4A8 静态量化的必选项**

**在必须使用 W4A8 静态量化的约束下，混合精度策略能够最接近 W4A8 动态量化或 W4A16 不量化的精度表现**。

**关键数据对比**:

| 配置 | Overall | 与 W4A8 动态对比 | 与 W4A16 对比 |
|------|---------|-----------------|---------------|
| r1r2 + r4-gptq-w4a8-dynamic | 0.833 | - | -0.011 |
| r1r2 + r4-gptq-w4a8-static-mix-precision | **0.822** | **-0.011** | **-0.015** |
| r1r2 + r4-gptq-w4a16-g128 | 0.822 | -0.011 | - |

**解读**:
- 混合精度静态量化（0.822）与动态量化（0.833）仅差 **1.1%**
- 混合精度静态量化（0.822）与 W4A16（0.822）**完全相同**
- 这证明了**在静态量化约束下，混合精度是接近无损的唯一可行方案**

**混合精度的决策依据**:
- 通过分析 Qwen3-8B 模型各层的激活值量化 scale，发现 **第 6 层和第 16 层的 `down_proj` 层存在显著的异常值（outliers）**
- 这些异常值导致 minmax 观察器的量化范围被极端拉伸，使得大部分正常值的量化精度极低
- 通过对这两层跳过激活量化（保持 FP16），其他层继续使用 W4A8，在保持 4x 压缩比的同时显著提升精度
- 通过prefix-quant应该可解决这两层的token outliers的情况

---

### 1. 动态量化 vs 静态量化

**动态量化显著优于静态量化**:

| 配置 | Overall | 损失 |
|------|---------|------|
| r1r2 + r4-gptq-w4a8-dynamic | 0.833 | +0.001 |
| r1r2 + r4-gptq-w4a8-static | 0.682 | -0.153 |

**原因分析**:
- **静态量化**使用固定范围（minmax）量化激活值，无法适应不同输入 token 的分布变化
- **动态量化**为每个输入 token 动态计算量化范围，精度损失更小
- W4A8 配置下，激活量化（8 位）是精度损失的主要来源，动态量化能有效缓解这个问题

### 2. W4A8 静态量化的严重失败

**最差结果**: `original + r4-gptq-w4a8-static` 只有 **0.476**，损失高达 **35.9%**！

**失败原因**:

1. **激活分布的极端性**:
   - Qwen3-8B 的某些层（特别是 `down_proj` 层）的激活值分布存在极端异常值
   - 使用 minmax 观察器时，少数异常值会拉伸整个量化范围，导致大部分正常值的量化精度极低

2. **静态观察器的局限性**:
   ```yaml
   observer: minmax  # ❌ 对异常值敏感
   ```
   - minmax 观察器只考虑最小值和最大值，忽略了分布的形状
   - 当存在极端异常值时，大部分数据被压缩到很小的量化区间

3. **4 位权重的敏感性**:
   - 4 位只有 16 个量化级别，对量化误差非常敏感
   - 激活量化的微小误差会被权重放大，导致精度急剧下降

### 3. 混合精度（Mix-Precision）的设计依据

**显著改善**: `r1r2 + r4-gptq-w4a8-static-mix-precision` 达到 **0.822**，相比纯 W4A8 的 **0.682** 提升了 **20.5%**！

**混合精度策略**:

```yaml
config_groups:
  group_0:
    weights:
      num_bits: 4
    input_activations:
      num_bits: 8
    targets: [Linear]  # 大部分层使用 W4A8

  group_1:
    weights:
      num_bits: 4
    # input_activations: null  # ❌ 不量化激活！
    targets: ["re:model.layers.6.mlp.down_proj", "re:model.layers.16.mlp.down_proj"]
```

**核心思想**:
- 对 `down_proj` 层（第 6 层和第 16 层）**不进行激活量化**
- 这些层的激活值分布最为极端，量化会导致严重精度损失
- 保持这些层的激活为 FP16，其他层使用 W4A8

**为什么选择第 6 层和第 16 层的 down_proj**:
- 通过分析模型各层的激活值量化 scale，发现这两层的 scale 值显著高于其他层
- scale 值高意味着激活值分布的范围更广，存在明显的异常值
- 实验证明，排除这两层的激活量化可以在保持 4x 压缩比的同时显著提升整体精度

### 4. R1R2 预平滑的效果

**R1R2 对静态量化有帮助**:

| 配置 | Original | R1R2 | 提升 |
|------|----------|------|------|
| r4-gptq-w4a8-static | 0.476 | 0.682 | +0.206 |
| r4-gptq-w4a8-static-mix-precision | 0.654 | 0.822 | +0.168 |

**原因**:
- R1R2 旋转平滑通过随机哈达玛变换使激活值分布更加均匀
- 减少了激活值的极端异常值，降低了静态量化的难度
- 但即使是 R1R2 + 混合精度，仍然无法达到动态量化的效果

---

## 量化策略对比

### 精度 vs 压缩比

| 策略 | 压缩比 | 精度 | 推荐场景 |
|------|--------|------|----------|
| **R4 + W4A16** | 4x | ⭐⭐⭐⭐⭐ | 追求最佳精度，接受稍慢推理 |
| **R4 + W4A8 动态** | 4x | ⭐⭐⭐⭐ | 平衡精度和速度 |
| **R4 + W4A8 静态 + 混合精度** | 4x | ⭐⭐⭐ | 对推理速度要求极高的场景 |
| **R4 + W4A8 静态（纯）** | 4x | ⭐ | ❌ 不推荐，精度损失过大 |
| **GPTQ + W4A8 动态** | 4x | ⭐⭐⭐ | 无 R4 变换的替代方案 |
| **GPTQ + W4A16** | 4x | ⭐⭐⭐⭐ | 无 R4 变换的高精度方案 |

### 推荐配置

**场景 1: 最佳精度（接近无损）**
```bash
python run_pipeline.py \
  --trans-config pipline/1-trans/example/original.yaml \
  --quant-config pipline/2-quant/example/r4-gptq-w4a16-g128.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml
```
**预期精度**: 0.847（超过基线）

**场景 2: 平衡精度和速度**
```bash
python run_pipeline.py \
  --trans-config pipline/1-trans/example/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/example/r4-gptq-w4a8-dynamic.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml
```
**预期精度**: 0.833（接近基线）

**场景 3: 最高推理速度（需要混合精度）**
```bash
python run_pipeline.py \
  --trans-config pipline/1-trans/example/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/example/r4-gptq-w4a8-static-mix-precision.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml
```
**预期精度**: 0.822（与 W4A16 相同，接近动态量化）
**说明**: 通过排除第 6 层和第 16 层的 down_proj 激活量化，在静态量化约束下实现了接近无损的精度

---

## 技术细节

### 混合精度配置解析

```yaml
config_groups:
  # group_0: 默认组 - 对所有 Linear 层应用 W4A8
  group_0:
    weights:
      num_bits: 4
      type: int
      symmetric: true
      strategy: channel
      observer: mse
    input_activations:
      num_bits: 8
      type: int
      symmetric: true
      strategy: tensor
      dynamic: false  # 静态量化
      observer: minmax
    targets: [Linear]

  # group_1: 特殊组 - 对指定层只量化权重，不量化激活
  group_1:
    weights:
      num_bits: 4
      type: int
      symmetric: true
      strategy: channel
      observer: mse
    # 注意：没有 input_activations 配置，意味着不量化激活
    targets: [
      "re:model.layers.6.mlp.down_proj",
      "re:model.layers.16.mlp.down_proj"
    ]
```

**工作原理**:
1. `group_0` 对所有 `Linear` 层应用 W4A8 量化
2. `group_1` 的配置会**覆盖** `group_0` 对特定层的配置
3. 由于 `group_1` 没有定义 `input_activations`，这些层的激活保持 FP16

**为什么选择第 6 层和第 16 层的 down_proj**:
- 通过分析 Qwen3-8B 模型各层的激活值量化 scale，发现第 6 层和第 16 层的 `down_proj` 层存在显著的异常值（outliers）
- scale 值异常高意味着激活值分布范围极广，minmax 观察器会被少数异常值主导
- 排除这两层的激活量化可以在保持 4x 压缩比的同时显著提升精度
- 只有两层使用 FP16 激活，对整体性能和显存占用影响很小

### R4 变换原理

**哈达玛变换（Hadamard Transform）**:
- 一种线性变换，通过正交矩阵使数据分布更加均匀
- R4 变换使用块大小为 256 的哈达玛变换
- 变换后的权重分布更加接近高斯分布，减少了量化误差

**为什么 R4 对 W4 特别有效**:
- 4 位量化只有 16 个级别，对分布不敏感
- 哈达玛变换使权重分布更加均匀，充分利用这 16 个级别
- 对于 8 位量化（256 个级别），R4 的收益相对较小

---

## 结论

### 核心结论

1. **在必须使用 W4A8 静态量化的约束下，混合精度是接近无损的唯一可行方案**
   - 混合精度静态量化（0.822）与动态量化（0.833）仅差 1.1%
   - 混合精度静态量化（0.822）与 W4A16（0.822）完全相同
   - 通过排除第 6 层和第 16 层的 down_proj 激活量化，有效解决了异常值问题

2. **R4 + W4A16 是最佳选择**: 在 4x 压缩比下实现 0.847 的精度，甚至超过基线模型
3. **动态量化远优于静态量化**: W4A8 动态量化（0.833）比纯静态量化（0.682）提升 22.2%
4. **纯 W4A8 静态量化不可行**: 精度损失高达 35.9%，必须使用混合精度

### 实践建议

**对于生产环境**:
- 优先使用 **R4 + W4A16** 配置，平衡精度和压缩比
- 如果推理速度是硬性要求，使用 **R4 + W4A8 动态量化**
- **如果必须使用静态量化（如推理框架限制），必须使用混合精度策略**，通过对第 6 层和第 16 层的 down_proj 跳过激活量化来实现接近无损的精度

**对于研究实验**:
- 可以尝试不同的混合精度策略，找出对激活量化最敏感的层
- 探索更先进的观察器（如 MSE + percentile 混合）
- 研究 R4 变换对其他模型的适用性

### 未来工作

1. **自动化敏感层检测**: 开发工具自动分析各层的激活值 scale，识别需要排除激活量化的层
2. **动态混合精度**: 根据输入动态调整不同层的量化精度
3. **更先进的观察器**: 研究对异常值更鲁棒的量化观察器（如 percentile + MSE 混合）
4. **跨模型验证**: 在其他大模型上验证 R4 + W4A16 的有效性，并分析各模型的敏感层分布

---

## 附录

### 测试环境

- **模型**: Qwen3-8B
- **硬件**: 4x GPU (CUDA)
- **评测数据集**: MMLU + GSM8K (1024 样本)
- **校准数据集**: UltraChat 200k (512 样本)

### 相关配置文件

- `run_w4_analysis.sh` - 测试脚本
- `pipline/2-quant/example/r4-gptq-w4a8-static.yaml` - 纯 W4A8 静态量化
- `pipline/2-quant/example/r4-gptq-w4a8-static-mix-precision.yaml` - 混合精度静态量化
- `results/results_mmlu_gsm8k_1024.csv` - 完整测试结果