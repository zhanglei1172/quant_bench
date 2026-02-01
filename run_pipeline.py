#!/usr/bin/env python3
"""
量化评测自动化流程
支持通过 YAML 配置文件进行转换、量化和评测
"""

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json
import time

PROJECT_ROOT = Path(__file__).parent


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_configs_from_dir(directory: Path) -> List[Path]:
    """获取目录下所有 YAML 配置文件"""
    return sorted(directory.glob("*.yaml"))


def run_transform(config: Dict[str, Any], base_model: str) -> str:
    """执行模型转换"""
    print(f"\n{'='*60}")
    print(f"[TRANSFORM] 开始转换: {config['name']}")
    print(f"{'='*60}")

    output_suffix = config.get('output_suffix')

    # 如果 output_suffix 为 None，直接使用基础模型路径
    if output_suffix is None:
        print(f"原始模型，无需转换: {base_model}")
        return base_model

    # 拼接输出路径
    output_model = base_model + output_suffix

    # 检查模型是否已存在
    if Path(output_model).exists():
        print(f"[TRANSFORM] 模型已存在，跳过转换: {output_model}")
        return output_model

    # 获取 quant_modifiers
    quant_modifiers = config.get('quant_modifiers', {})

    # 构建 recipe（将 quant_modifiers 包装在 recipe 结构中）
    recipe = {'quant_stage': {'quant_modifiers': [quant_modifiers]}}
    recipe_yaml_str = yaml.dump(recipe, default_flow_style=False, sort_keys=False)

    # 构建转换脚本
    trans_script = f"""
import torch
import yaml
import json
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

MODEL_ID = "{base_model}"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="float32")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Save recipe to temp file and pass file path
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    recipe_path = f.name
    f.write('''{recipe_yaml_str}''')

try:
    oneshot(model=model, recipe=recipe_path, pipeline="datafree")
finally:
    os.unlink(recipe_path)

SAVE_DIR = "{output_model}"
model.save_pretrained(SAVE_DIR, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR)

# Remove quantization_config if exists
config_path = os.path.join(SAVE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        cfg = json.load(f)
    if "quantization_config" in cfg:
        del cfg["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Removed quantization_config from {{config_path}}")
"""

    # 保存临时脚本
    tmp_script = PROJECT_ROOT / "tmp_trans.py"
    with open(tmp_script, 'w') as f:
        f.write(trans_script)

    try:
        subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT)
        print(f"[TRANSFORM] 转换完成: {output_model}")
        return output_model
    finally:
        if tmp_script.exists():
            tmp_script.unlink()


def run_quantization(input_model: str, quant_config: Dict[str, Any]) -> str:
    """执行模型量化"""
    quant_name = quant_config['name']

    print(f"\n{'='*60}")
    print(f"[QUANTIZE] 开始量化: {quant_name}")
    print(f"{'='*60}")

    # 构建输出路径
    quant_suffix = quant_config.get('output_suffix', '')
    output_model = input_model + quant_suffix

    # 检查模型是否已存在
    if Path(output_model).exists():
        print(f"[QUANTIZE] 模型已存在，跳过量化: {output_model}")
        return output_model

    # 构建量化脚本
    quant_script = build_quant_script(input_model, output_model, quant_config)

    # 保存临时脚本
    tmp_script = PROJECT_ROOT / "tmp_quant.py"
    with open(tmp_script, 'w') as f:
        f.write(quant_script)

    try:
        subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT)
        print(f"[QUANTIZE] 量化完成: {output_model}")
        return output_model
    finally:
        if tmp_script.exists():
            tmp_script.unlink()


def build_quant_script(input_model: str, output_model: str, config: Dict[str, Any]) -> str:
    """构建量化脚本"""
    calib_cfg = config.get('calibration')
    quant_modifiers = config.get('quant_modifiers', {})
    save_compressed = config.get('save_compressed', False)

    # 构建 recipe（将 quant_modifiers 包装在 recipe 结构中）
    recipe = {'quant_stage': {'quant_modifiers': quant_modifiers}}
    recipe_yaml_str = yaml.dump(recipe, default_flow_style=False, sort_keys=False)

    # 检查是否是 data-free 量化
    is_data_free = calib_cfg is None

    # 检查是否是 weight-only (用于清理 quantization_config)
    is_weight_only = False
    if 'QuantizationModifier' in quant_modifiers:
        mod_cfg = quant_modifiers['QuantizationModifier']
        groups = mod_cfg.get('config_groups', {})
        for group in groups.values():
            # 如果没有 input_activations 或者 input_activations 被注释掉了，则是 weight-only
            act = group.get('input_activations')
            if act is None:
                is_weight_only = True
                break

    script = f"""
import torch
import yaml
import json
import os
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
{'from datasets import load_dataset' if not is_data_free else ''}

MODEL_ID = "{input_model}"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
"""

    if not is_data_free:
        script += f"""
# Load calibration dataset
ds = load_dataset("{calib_cfg['dataset_id']}", split="{calib_cfg['dataset_split']}")
ds = ds.shuffle(seed=42).select(range({calib_cfg['num_samples']}))

def preprocess(example):
    return {{"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}}
ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length={calib_cfg['max_sequence_length']},
        truncation=True,
        add_special_tokens=False,
    )
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Save recipe to temp file and pass file path
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    recipe_path = f.name
    f.write('''{recipe_yaml_str}''')

try:
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe_path,
        max_seq_length={calib_cfg['max_sequence_length']},
        num_calibration_samples={calib_cfg['num_samples']},
        save_compressed={save_compressed},
        trust_remote_code_model=True,
    )
finally:
    os.unlink(recipe_path)
"""
    else:
        script += f"""
# Data-free quantization
# Save recipe to temp file and pass file path
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    recipe_path = f.name
    f.write('''{recipe_yaml_str}''')

try:
    oneshot(
        model=model,
        recipe=recipe_path,
        pipeline="datafree",
        save_compressed={save_compressed},
        trust_remote_code_model=True,
    )
finally:
    os.unlink(recipe_path)
"""

    script += f"""
SAVE_DIR = "{output_model}"
model.save_pretrained(SAVE_DIR, save_compressed={save_compressed})
tokenizer.save_pretrained(SAVE_DIR)
"""

    return script


def prepare_dataset(eval_config: Dict[str, Any]) -> str:
    """准备评测数据集"""
    from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
    from evalscope.utils.io_utils import dump_jsonl_data

    # 从配置获取数据集名称
    dataset_name = eval_config['dataset_name']
    dataset_path = PROJECT_ROOT / f"test/{dataset_name}.jsonl"

    # 检查数据集是否已存在
    if dataset_path.exists():
        print(f"[DATASET] 数据集已存在，直接使用: {dataset_path}")
        return str(dataset_path)

    print(f"[DATASET] 数据集不存在，开始生成...")

    # 构建数据集配置
    datasets = []
    for ds_cfg in eval_config['datasets']:
        datasets.append(DatasetInfo(
            name=ds_cfg['name'],
            weight=ds_cfg['weight'],
            task_type=ds_cfg['task_type'],
            tags=ds_cfg['tags'],
            args={'few_shot_num': ds_cfg['few_shot_num']}
        ))

    schema = CollectionSchema(
        name=eval_config['name'],
        datasets=[
            CollectionSchema(name='Mixed', weight=1, datasets=datasets)
        ]
    )

    mixed_data = WeightedSampler(schema).sample(eval_config['sampling']['total_samples'])

    # 保存数据集
    dump_jsonl_data(mixed_data, str(dataset_path))

    print(f"[DATASET] 数据集已保存到: {dataset_path}")
    return str(dataset_path)


def find_available_port(start_port: int = 8001) -> int:
    """查找可用端口"""
    import socket
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:  # 端口未被占用
                return port
        port += 1
    raise RuntimeError("无法找到可用端口")


def wait_for_service(port: int, timeout: int = 300) -> bool:
    """等待服务启动，检查端口是否可访问"""
    import socket
    import urllib.request
    import urllib.error

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # 尝试连接端口
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    # 端口可连接，尝试发送 HTTP 请求
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
                        return True
                    except (urllib.error.URLError, urllib.error.HTTPError):
                        # health 端点可能不存在，但服务已经启动
                        pass
                    return True
        except (socket.timeout, socket.error):
            pass
        time.sleep(2)
    return False


def run_evaluation(model_path: str, eval_config: Dict[str, Any], dataset_path: str, task_idx: int, total_tasks: int) -> Dict[str, float]:
    """执行评测"""
    model_name = Path(model_path).name
    port = find_available_port()

    # 获取 GPU 资源配置
    gpu_resources = eval_config.get('gpu_resources', {})
    devices = gpu_resources.get('devices', [0])
    tensor_parallel_size = gpu_resources.get('tensor_parallel_size', 1)

    # 简单的 GPU 分配策略：循环使用配置的设备
    gpu_idx = devices[task_idx % len(devices)]

    # 获取输出目录配置
    work_dir = eval_config.get('output', {}).get('work_dir', 'outputs')
    dataset_name = eval_config['dataset_name']

    # 固定使用 no_timestamp=True，输出目录为 eval_<config_name>
    output_dir = PROJECT_ROOT / work_dir

    # 检查评测结果是否已存在
    report_dir = PROJECT_ROOT / work_dir / "reports" / model_name
    report_path = report_dir / "data_collection.json"

    if report_path.exists():
        print(f"\n{'='*60}")
        print(f"[EVAL] 评测结果已存在，跳过评测: {model_name}")
        print(f"[EVAL] 结果目录: {report_dir}")
        print(f"{'='*60}")

        # 解析已存在的结果
        results = parse_evaluation_results(model_name, output_dir)
        return results

    print(f"\n{'='*60}")
    print(f"[EVAL] 开始评测: {model_name} (端口: {port}, GPU: {gpu_idx}, TP: {tensor_parallel_size})")
    print(f"[EVAL] 输出目录: {output_dir}")
    print(f"{'='*60}")

    # 启动 vLLM 服务
    vllm_cmd = [
        "vllm", "serve", model_path,
        f"--port={port}",
        f"--tensor-parallel-size={tensor_parallel_size}"
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    env['VLLM_USE_MODELSCOPE'] = 'true'

    print(f"[EVAL] 启动 vLLM 服务: {' '.join(vllm_cmd)}")
    # 使用 None 重定向输出到控制台，便于调试
    vllm_process = subprocess.Popen(vllm_cmd, env=env, stdout=None, stderr=None)

    # 等待服务真正可用
    print(f"[EVAL] 等待服务启动...")
    if not wait_for_service(port):
        vllm_process.terminate()
        vllm_process.wait()
        raise RuntimeError(f"vLLM 服务启动超时 (端口: {port})")
    print(f"[EVAL] 服务已就绪")

    try:
        # 运行评测
        eval_script = build_eval_script(model_path, eval_config, dataset_path, port, str(output_dir))
        tmp_script = PROJECT_ROOT / "tmp_eval.py"
        with open(tmp_script, 'w') as f:
            f.write(eval_script)

        result = subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT)

        # 解析结果
        results = parse_evaluation_results(model_name, output_dir)
        print(f"[EVAL] 评测完成: {model_name}")
        return results

    finally:
        # 停止 vLLM 服务
        vllm_process.terminate()
        vllm_process.wait()
        if tmp_script.exists():
            tmp_script.unlink()


def build_eval_script(model_path: str, eval_config: Dict[str, Any], dataset_path: str, port: int, output_dir: str) -> str:
    """构建评测脚本"""
    gen_cfg = eval_config['eval_config']['generation_config']

    script = f"""
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType
import os

task = TaskConfig(
    model="{model_path}",
    api_url="http://127.0.0.1:{port}/v1/chat/completions",
    api_key="EMPTY",
    eval_type=EvalType.SERVICE,
    datasets=['data_collection'],
    dataset_args={{
        'data_collection': {{
            'local_path': '{dataset_path}',
        }}
    }},
    eval_batch_size={eval_config['eval_config']['eval_batch_size']},
    generation_config={{
        'max_tokens': {gen_cfg['max_tokens']},
        'temperature': {gen_cfg['temperature']},
    }},
    no_timestamp=True,
    work_dir="{output_dir}",
)

run_task(task)
"""
    return script


def parse_evaluation_results(model_name: str, output_dir: Path = None) -> Dict[str, float]:
    """解析评测结果"""
    # 在 outputs 目录中查找最新的结果
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        return {}

    # 如果指定了输出目录，直接使用
    if output_dir:
        reports_dir = output_dir / "reports" / model_name
    else:
        # 找到最新的评测目录
        try:
            latest_dir = max(outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            reports_dir = latest_dir / "reports" / model_name
        except (ValueError, FileNotFoundError):
            return {}

    if not reports_dir.exists():
        return {}

    # 解析结果文件
    results = {}
    report_path = reports_dir / "data_collection.json"

    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
            # 从 subsets 中提取 mmlu 和 gsm8k 的分数
            metrics = report.get('metrics', [])
            if metrics:
                categories = metrics[0].get('categories', [])
                if categories:
                    subsets = categories[0].get('subsets', [])
                    for subset in subsets:
                        name = subset.get('name', '')
                        score = subset.get('score', 0.0)
                        if 'gsm8k' in name:
                            results['gsm8k'] = score
                        elif name.startswith('mmlu/'):
                            # MMLU 是所有 mmlu 子集的平均分
                            if 'mmlu' not in results:
                                results['mmlu'] = []
                            results['mmlu'].append(score)

    # 计算 MMLU 平均分
    if 'mmlu' in results and isinstance(results['mmlu'], list):
        results['mmlu'] = sum(results['mmlu']) / len(results['mmlu']) if results['mmlu'] else 0.0

    return results


def save_summary_csv(all_results: List[Dict[str, Any]], output_path: str):
    """保存结果汇总 CSV"""
    import csv

    if not all_results:
        print("[SUMMARY] 没有结果需要保存")
        return

    # 获取所有字段
    fieldnames = ['transform', 'quantization', 'model_path', 'mmlu', 'gsm8k']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"[SUMMARY] 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='量化评测自动化流程')
    parser.add_argument('--trans-config', type=str, default=None, nargs='+',
                        help='转换配置文件路径 (支持多个，默认使用 1-trans 目录下所有 YAML)')
    parser.add_argument('--quant-config', type=str, default=None, nargs='+',
                        help='量化配置文件路径 (支持多个，默认使用 2-quant 目录下所有 YAML)')
    parser.add_argument('--eval-config', type=str, default=None,
                        help='评测配置文件路径 (默认使用 3-eval 目录下的单个 YAML)')
    parser.add_argument('--output-csv', type=str, default='results_summary.csv',
                        help='结果汇总 CSV 文件路径')

    args = parser.parse_args()

    # 加载配置
    trans_configs = []
    quant_configs = []

    # 加载评测配置
    if args.eval_config:
        eval_config = load_yaml_config(args.eval_config)
        print(f"[CONFIG] 使用评测配置: {args.eval_config}")
    else:
        eval_dir = PROJECT_ROOT / "3-eval"
        eval_files = list(eval_dir.glob("*.yaml"))
        if len(eval_files) == 0:
            raise RuntimeError(f"评测配置目录为空: {eval_dir}")
        elif len(eval_files) > 1:
            raise RuntimeError(f"评测配置目录下有多个 YAML 文件: {[f.name for f in eval_files]}，请只保留一个")
        else:
            eval_config = load_yaml_config(eval_files[0])
            print(f"[CONFIG] 使用评测配置: {eval_files[0].name}")

    # 加载转换配置
    if args.trans_config:
        for cfg_path in args.trans_config:
            trans_configs.append(load_yaml_config(cfg_path))
    else:
        for cfg_path in get_configs_from_dir(PROJECT_ROOT / "1-trans"):
            trans_configs.append(load_yaml_config(cfg_path))

    # 加载量化配置
    if args.quant_config:
        for cfg_path in args.quant_config:
            quant_configs.append(load_yaml_config(cfg_path))
    else:
        for cfg_path in get_configs_from_dir(PROJECT_ROOT / "2-quant"):
            quant_configs.append(load_yaml_config(cfg_path))

    print(f"\n{'='*60}")
    print(f"配置加载完成:")
    print(f"  转换配置: {len(trans_configs)} 个")
    print(f"  量化配置: {len(quant_configs)} 个")
    print(f"  总任务数: {len(trans_configs) * len(quant_configs)} 个")
    print(f"{'='*60}\n")

    # 准备评测数据集
    dataset_path = prepare_dataset(eval_config)

    # 存储所有结果
    all_results = []

    # 执行流程
    task_idx = 0
    total_tasks = len(trans_configs) * len(quant_configs)

    # 获取基础模型路径（从第一个转换配置中获取，或者需要统一配置）
    base_model = trans_configs[0].get('input_model') if 'input_model' in trans_configs[0] else "/workspace/lim42@xiaopeng.com/github/quant_example/outputs/Qwen3-8B"

    for trans_cfg in trans_configs:
        print(f"\n{'#'*60}")
        print(f"# 处理转换配置: {trans_cfg['name']}")
        print(f"{'#'*60}")

        # 执行转换
        trans_output = run_transform(trans_cfg, base_model)

        for quant_cfg in quant_configs:
            task_idx += 1
            print(f"\n{'#'*60}")
            print(f"# 任务 {task_idx}/{len(trans_configs) * len(quant_configs)}: {trans_cfg['name']} + {quant_cfg['name']}")
            print(f"{'#'*60}")

            # 执行量化
            quant_output = run_quantization(trans_output, quant_cfg)

            # 执行评测
            results = run_evaluation(quant_output, eval_config, dataset_path, task_idx, total_tasks)

            # 记录结果
            result = {
                'transform': trans_cfg['name'],
                'quantization': quant_cfg['name'],
                'model_path': quant_output,
                'mmlu': results.get('mmlu', 0.0),
                'gsm8k': results.get('gsm8k', 0.0),
            }
            all_results.append(result)

    # 保存汇总结果
    if all_results:
        save_summary_csv(all_results, args.output_csv)

    print(f"\n{'='*60}")
    print("所有任务完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()