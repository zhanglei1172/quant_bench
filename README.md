# Qwen3-8B 量化评测自动化流程

本项目提供了 Qwen3-8B 模型从转换（如 SpinQuant 预平滑）到量化再到评测的完整自动化流程。

## 核心流程

1. **转换阶段**：FP32 权重 + 等价变换（如 SpinQuant R1/R2），理论上数学等价，无精度损失
2. **量化阶段**：采用 RTN/GPTQ 等算法对权重进行量化，会有精度损失
3. **评测阶段**：基于 vLLM 部署推理服务，结合 evalscope 进行自动化测评

## 目录结构

```
quant_example/
├── pipline/              # 自动化流程配置
│   ├── 1-trans/         # 转换配置
│   │   ├── example/     # 配置模板
│   │   ├── original.yaml
│   │   └── spinquant-r1r2.yaml
│   ├── 2-quant/         # 量化配置
│   │   ├── example/
│   │   ├── rtn-w8a8-dynamic.yaml
│   │   └── rtn-w8a8-static.yaml
│   └── 3-eval/          # 评测配置
│       ├── example/
│       └── mmlu_gsm8k.yaml
├── test/                # 手动测试实验区
│   ├── 1-trans.py       # 转换脚本
│   ├── 2-quant.py       # 量化脚本
│   ├── 3-eval.py        # 评测脚本
│   ├── extra/           # 额外实验
│   └── TEST.md          # 实验说明
├── data/                # 数据目录
│   ├── models/          # 模型文件
│   ├── datasets/        # HuggingFace 数据集
│   └── *.jsonl          # 生成的评测数据集
├── logs/                # 日志目录
│   └── 20260201_xxxxxx/ # 时间戳目录
├── outputs/             # 评测输出
├── results/             # CSV 结果汇总
├── run_pipeline.py      # 主流程脚本
└── requirements.txt     # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 下载模型和数据集
# 详见 data/HELP.md
```

### 3. 运行自动化流程

```bash
# 使用默认配置（pipline 目录下的所有 YAML）
python run_pipeline.py

# 指定配置文件
python run_pipeline.py \
  --trans-config pipline/1-trans/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/rtn-w8a8-dynamic.yaml \
  --eval-config pipline/3-eval/mmlu_gsm8k.yaml
```

## 配置说明

### 路径配置

所有相对路径统一相对于项目根目录解析：

```yaml
input_model: ./data/models/Qwen/Qwen3-8B  # 项目根目录下的 data/...
```

### 转换配置示例

```yaml
name: original
input_model: ./data/models/Qwen/Qwen3-8B
output_suffix: null  # null 表示无需转换
```

```yaml
name: spinquant-r1r2
input_model: ./data/models/Qwen/Qwen3-8B
output_suffix: -r1r2
quant_modifiers:
  SpinQuantModifier:
    rotations: ["R1", "R2"]
    transform_type: random-hadamard
```

### 量化配置示例

```yaml
name: rtn-w8a8-dynamic
output_suffix: -rtn-W8A8-dynamic
quant_modifiers:
  QuantizationModifier:
    config_groups:
      group_0:
        weights: {num_bits: 8, type: int, symmetric: true, strategy: channel}
        input_activations: {num_bits: 8, type: int, symmetric: true, strategy: token, dynamic: true}
save_compressed: True
```

### 评测配置示例

```yaml
dataset_name: mmlu_gsm8k_20
datasets:
  - name: mmlu
    weight: 1
    task_type: exam
    tags: [en]
    few_shot_num: 0
  - name: gsm8k
    weight: 1
    task_type: math
    tags: [en]
    few_shot_num: 0
sampling:
  total_samples: 20
gpu_resources:
  devices: [0,1]
  tensor_parallel_size: 1
output:
  work_dir: outputs/mmlu_gsm8k_20
```

## 特性

- ✅ **自动跳过**：模型和数据集已存在时自动跳过
- ✅ **并行执行**：量化和评测任务并行执行
- ✅ **日志分离**：vLLM、llmcompressor、evalscope 日志分离
- ✅ **Debug 脚本**：临时脚本保存到 logs 目录，方便 debug
- ✅ **增量结果**：CSV 结果支持增量写入和更新

## 输出说明

```
logs/
└── 20260201_xxxxxx/
    ├── {model_name}_vllm.log           # vLLM 日志
    ├── {model_name}_llmcompressor.log  # llmcompressor 日志
    ├── {model_name}_evalscope.log      # evalscope 日志
    ├── {model_name}_trans.py           # 转换脚本（debug 用）
    ├── {model_name}_quant.py           # 量化脚本（debug 用）
    └── {model_name}_eval.py            # 评测脚本（debug 用）

results/
└── results_{dataset_name}.csv          # 结果汇总 CSV

outputs/{work_dir}/
    ├── predictions/                    # 预测结果
    ├── reports/                        # 评测报告
    └── configs/                        # 配置文件
```

## 测试实验

对于新配置的实验验证，建议先在 `test/` 目录下手动测试：

```bash
cd test
python 1-trans.py   # 转换
python 2-quant.py   # 量化
python 3-eval.py    # 评测
```

验证通过后再集成到 `pipline/` 进行自动化测试。详见 `test/TEST.md`。

## 实验结果

### 主流程测评

| 模型版本 | MMLU | GSM8K | 备注 |
| :--- | :---: | :---: | :--- |
| Qwen3-8B (Base) | 0.8321 | 0.8809 | 原始 FP16 模型 |
| Qwen3-8B-R1R2 (Smooth) | 0.8428 | 0.8887 | FP32 预平滑，等价变换 |
| Qwen3-8B-R1R2-gptq-W8A8-static | 0.7598 | 0.7891 | W8A8 静态量化 |

### 消融实验

| 模型版本 | MMLU | GSM8K |
| :--- | :---: | :---: |
| Qwen3-8B-R1R2-gptq-W8-weightonly | 0.8301 | 0.8672 |
| Qwen3-8B-R1R2-rtn-W8-weightonly | 0.8301 | 0.8691 |
| Qwen3-8B-R1R2-gptq-W8A8-dynamic | 0.8359 | 0.8652 |
| Qwen3-8B-R1R2-gptq-ignore-down-W8A8-static | 0.8438 | 0.8770 |

**关键发现**：
- 激活量化是瓶颈（weight-only 损失极小）
- 动态量化显著优于静态量化
- down_proj 层对量化误差最敏感