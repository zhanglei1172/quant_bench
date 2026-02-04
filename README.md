# Qwen3-8B 量化评测自动化流程

Qwen3-8B 模型从转换（SpinQuant）→ 量化（RTN/GPTQ）→ 评测的完整自动化流程。

## 快速开始

### 安装依赖

**推荐方式：使用 uv**
```bash
./install_env.sh
# 或手动执行：
uv sync -i https://pypi.tuna.tsinghua.edu.cn/simple
source ./.venv/bin/activate
```

**使用 pip**
```bash
pip install -r requirements.txt
```

### 运行流程

**方式1：使用默认配置**
运行pipline中文件夹内的yaml配置
```bash
python run_pipeline.py
```

**方式2：指定配置文件**
```bash
python run_pipeline.py \
  --trans-config pipline/1-trans/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/rtn-w8a8-dynamic.yaml \
  --eval-config pipline/3-eval/mmlu_gsm8k.yaml
```

**方式3：运行示例配置**
```bash
./run_example.sh
```

## 评测配置

`pipline/3-eval/mmlu_gsm8k.yaml`:
```yaml
dataset_name: mmlu_gsm8k_128
base_model: ./data/models/Qwen/Qwen3-8B  # 模型路径
gpu_resources:
  devices: [0,1,2,3]
  pipeline_parallel_size: 1  # PP=1时每个任务用1个GPU，PP=2时用2个GPU
```

## 目录结构

```
pipline/
├── 1-trans/          # 转换配置（SpinQuant等）
├── 2-quant/          # 量化配置（RTN/GPTQ）
│   └── example/      # 示例配置
└── 3-eval/           # 评测配置（MMLU/GSM8K）

data/                 # 模型和数据集文件
results/              # CSV结果汇总
logs/                 # 临时脚本和日志
outputs/              # 评测报告
```

## 输出结果

```bash
# CSV结果
results/results_{dataset_name}.csv

# 评测报告
outputs/{dataset_name}/reports/{model_name}/
```

## 特性

- ✅ 自动跳过已存在的模型和结果
- ✅ 量化和评测任务并行执行，一键完成复杂量化测评流程
- ✅ 临时py脚本保存到logs目录，可debug
- ✅ GPU自动并行分配（按pipeline_parallel_size, R4与tensor_parallel不兼容）
- ✅ 从文件名自动提取配置名称，无需在 YAML 中重复指定

## 测试已有模型

如果你想测试已有的转换或量化模型，可以按照以下步骤操作：

### 测试已有转换模型

1. 在 `pipline/1-trans/` 目录下创建一个空的 YAML 文件，文件名即作为转换名称（例如 `my-custom-trans.yaml`）
2. 在 `data/models/` 目录下创建对应的模型文件夹，文件名需要加上 `-` 前缀（例如 `Qwen/Qwen3-8B-my-custom-trans`）
3. 将你已有的转换模型文件放入该文件夹中
4. 运行流程，系统会自动跳过转换步骤，直接使用已有模型进行量化和评测

### 测试已有量化模型

1. 在 `pipline/2-quant/` 目录下创建一个空的 YAML 文件，文件名即作为量化名称（例如 `my-custom-quant.yaml`）
2. 在 `data/models/` 目录下创建对应的模型文件夹，路径需要是 `基础模型路径-转换名称-量化名称`（例如 `Qwen/Qwen3-8B-r1r2-my-custom-quant`）
3. 将你已有的量化模型文件放入该文件夹中
4. 运行流程，系统会自动跳过量化步骤，直接评测已有模型

**示例**：
```
# 目录结构
data/models/Qwen/
├── Qwen3-8B                    # 原始模型
├── Qwen3-8B-r1r2               # 已有的转换模型
└── Qwen3-8B-r1r2-w4a8-custom    # 已有的量化模型

# 配置文件
pipline/1-trans/r1r2.yaml       # 空文件或仅包含配置
pipline/2-quant/w4a8-custom.yaml # 空文件或仅包含配置
```

系统会自动检测模型文件夹是否存在，如果存在则跳过对应的转换或量化步骤。