# Qwen3-8B 量化工作流自动化工具

本项目提供了 Qwen3-8B 模型从预平滑 (Pre-smoothing) 到量化 (Quantization) 再到评测 (Evaluation) 的完整自动化工具链。

## 核心流程

1. **预平滑 (Pre-smoothing)**：在保持模型结构不变的前提下，通过旋转平滑（Rotation Smoothing）和权重等效变换处理激活值
2. **模型量化 (Quantization)**：采用 RTN/GPTQ 等算法对权重进行量化
3. **指标评测 (Evaluation)**：基于 vLLM 部署推理服务，结合 evalscope 进行自动化测评

## 目录结构

```
quant_example/
├── 1-trans/              # 转换配置目录
│   ├── example/         # 配置模板
│   ├── original.yaml    # 原始模型配置
│   └── spinquant-r1r2.yaml  # SpinQuant R1+R2 配置
├── 2-quant/              # 量化配置目录
│   ├── example/         # 配置模板
│   ├── rtn-w8a8-dynamic.yaml   # RTN W8A8 动态量化
│   └── rtn-w8a8-static.yaml    # RTN W8A8 静态量化
├── 3-eval/               # 评测配置目录
│   ├── example/         # 配置模板
│   └── mmlu_gsm8k.yaml  # MMLU + GSM8K 评测配置
├── run_pipeline.py       # 主流程编排脚本
└── outputs/              # 输出目录
    └── mmlu_gsm8k_128/  # 评测结果
        ├── predictions/ # 预测结果
        ├── reports/     # 评测报告
        └── configs/     # 配置文件
```

## 使用方式

### 基本使用

```bash
# 使用默认配置（目录下的所有 YAML 文件）
python run_pipeline.py
```

### 指定配置文件

```bash
# 指定转换、量化、评测配置
python run_pipeline.py \
  --trans-config 1-trans/example/spinquant-r1r2.yaml \
  --quant-config 2-quant/example/gptq-w8a8-static.yaml \
  --eval-config 3-eval/example/mmlu_gsm8k.yaml

# 支持多个配置文件
python run_pipeline.py \
  --trans-config 1-trans/example/spinquant-r1r2.yaml 1-trans/example/spinquant-r4.yaml \
  --quant-config 2-quant/example/gptq-w8a8-static.yaml 2-quant/example/gptq-w8a8-dynamic.yaml
```

### 自定义输出路径

```bash
python run_pipeline.py --output-csv my_results.csv
```

## 配置说明

### 转换配置 (1-trans/)

**original.yaml** - 原始模型，无需转换
```yaml
output_suffix: null  # 不添加后缀，直接使用原始模型
```

**spinquant-r1r2.yaml** - SpinQuant R1+R2 旋转平滑
```yaml
output_suffix: -r1r2
quant_modifiers:
  SpinQuantModifier:
    rotations: ["R1", "R2"]
    transform_type: random-hadamard
```

### 量化配置 (2-quant/)

支持多种量化方案：

1. **Weight-only**: 仅权重量化，无激活量化
2. **W8A8 Static**: 权重和激活都 8-bit，静态量化
3. **W8A8 Dynamic**: 权重和激活都 8-bit，动态量化

### 评测配置 (3-eval/)

```yaml
dataset_name: mmlu_gsm8k_128  # 数据集名称

datasets:
  - name: mmlu
    weight: 1
    few_shot_num: 0
  - name: gsm8k
    weight: 1
    few_shot_num: 0

sampling:
  total_samples: 128  # 总样本数

gpu_resources:
  devices: [0,1]       # 可用的 GPU
  tensor_parallel_size: 1

output:
  work_dir: outputs/mmlu_gsm8k_128  # 输出目录
```

## 特性

- ✅ **自动跳过已存在模型**：检测模型路径，如果存在则跳过
- ✅ **数据集复用**：根据 `dataset_name` 自动复用已生成的数据集
- ✅ **评测结果复用**：如果评测结果已存在，直接读取跳过
- ✅ **自动端口分配**：从 8001 开始自动查找可用端口
- ✅ **服务健康检查**：等待 vLLM 服务真正可用后再运行评测
- ✅ **结果自动汇总**：自动生成 CSV 结果汇总表

## 输出结果

运行完成后会生成 `results_summary.csv`，包含所有配置组合的评测结果：

| transform | quantization | model_path | mmlu | gsm8k |
|-----------|--------------|------------|------|-------|
| original | rtn-w8a8-dynamic | ... | 0.873 | 0.953 |
| original | rtn-w8a8-static | ... | 0.214 | 0.297 |

## 测评结果

### 主流程测评

| 模型版本 | MMLU | GSM8K | 备注 |
| :--- | :---: | :---: | :--- |
| Qwen3-8B (Base) | 0.8321 | 0.8809 | 原始 FP16 模型 |
| Qwen3-8B-R1R2 (Smooth) | 0.8428 | 0.8887 | 预平滑处理后 |

### 量化方案对比

| 量化方案 | MMLU | GSM8K | 说明 |
| :--- | :---: | :---: | :--- |
| RTN W8A8 Dynamic | 0.873 | 0.953 | 动态量化效果接近原始模型 |
| RTN W8A8 Static | 0.214 | 0.297 | 静态量化损失较大 |

## 注意事项

1. **3-eval 目录下只能有 1 个 YAML 文件**，否则会报错
2. **GPU 资源配置**：根据可用 GPU 数量配置 `devices` 列表
3. **模型路径**：确保 `input_model` 指向的模型路径存在
4. **评测结果复用**：如果修改了模型但想重新评测，请删除对应的评测结果目录