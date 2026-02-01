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
from threading import Lock
from loguru import logger

PROJECT_ROOT = Path(__file__).parent
PIPELINE_DIR = PROJECT_ROOT / "pipline"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# 创建带时间戳的日志目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGS_DIR = LOGS_DIR / timestamp
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 配置 loguru
logger.remove()  # 移除默认 handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 全局端口分配器（线程安全）
class PortAllocator:
    def __init__(self, start_port=8001):
        self.next_port = start_port
        self.lock = Lock()
    
    def allocate(self):
        with self.lock:
            port = self.next_port
            self.next_port += 1
            return port

port_allocator = PortAllocator()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 解析路径字段，支持相对路径
    if 'input_model' in config:
        config['input_model'] = resolve_path(config['input_model'], path.parent)
    
    return config


def resolve_path(path_str: str, base_dir: Path = None) -> str:
    """解析路径，支持相对路径和绝对路径"""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    else:
        # 相对路径统一相对于项目根目录解析
        return str((PROJECT_ROOT / path).resolve())


def get_configs_from_dir(directory: Path) -> List[Path]:
    """获取目录下所有 YAML 配置文件"""
    return sorted(directory.glob("*.yaml"))


def run_transform(config: Dict[str, Any], base_model: str) -> str:
    """执行模型转换"""
    output_suffix = config.get('output_suffix')

    # 如果 output_suffix 为 None，直接使用基础模型路径
    if output_suffix is None:
        return base_model

    # 拼接输出路径
    output_model = base_model + output_suffix

    # 检查模型是否已存在
    if Path(output_model).exists():
        logger.info(f"跳过 {config['name']} (模型已存在)")
        return output_model

    logger.info(f"转换 {config['name']}")

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
"""

    # 保存临时脚本到 logs 目录（不删除，用于 debug）
    model_name = Path(output_model).name
    tmp_script = LOGS_DIR / f"{model_name}_trans.py"
    with open(tmp_script, 'w') as f:
        f.write(trans_script)

    subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT)
    return output_model


def run_quantization(input_model: str, quant_config: Dict[str, Any]) -> str:
    """执行模型量化"""
    quant_name = quant_config['name']

    # 构建输出路径
    quant_suffix = quant_config.get('output_suffix', '')
    output_model = input_model + quant_suffix
    model_name = Path(output_model).name

    # 检查模型是否已存在
    if Path(output_model).exists():
        logger.info(f"跳过 {quant_name} (模型已存在)")
        return output_model

    logger.info(f"量化 {quant_name}")

    # 构建量化脚本
    quant_script = build_quant_script(input_model, output_model, quant_config)

    # 保存临时脚本到 logs 目录（不删除，用于 debug）
    tmp_script = LOGS_DIR / f"{model_name}_quant.py"
    with open(tmp_script, 'w') as f:
        f.write(quant_script)

    # 日志文件
    log_file = LOGS_DIR / f"{model_name}_llmcompressor.log"

    with open(log_file, 'w') as log_f:
        subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT, stdout=log_f, stderr=log_f)
    return output_model


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
    dataset_path = PROJECT_ROOT / f"data/{dataset_name}.jsonl"

    # 检查数据集是否已存在
    if dataset_path.exists():
        logger.info(f"使用现有数据集: {dataset_path.name}")
        return str(dataset_path)

    logger.info(f"生成数据集: {dataset_name}")

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
        name=eval_config['dataset_name'],
        datasets=[
            CollectionSchema(name='Mixed', weight=1, datasets=datasets)
        ]
    )

    mixed_data = WeightedSampler(schema).sample(eval_config['sampling']['total_samples'])

    # 保存数据集
    dump_jsonl_data(mixed_data, str(dataset_path))
    return str(dataset_path)


def find_available_port() -> int:
    """获取下一个可用端口（线程安全）"""
    return port_allocator.allocate()


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

    # 日志文件
    vllm_log_file = LOGS_DIR / f"{model_name}_vllm.log"
    eval_log_file = LOGS_DIR / f"{model_name}_evalscope.log"

    # 检查评测结果是否已存在
    report_dir = PROJECT_ROOT / work_dir / "reports" / model_name
    report_path = report_dir / "data_collection.json"

    if report_path.exists():
        logger.info(f"跳过 {model_name} (结果已存在)")
        results = parse_evaluation_results(model_name, output_dir)
        return results

    logger.info(f"评测 {model_name} (GPU:{gpu_idx}, TP:{tensor_parallel_size})")

    # 启动 vLLM 服务
    vllm_cmd = [
        "vllm", "serve", model_path,
        f"--port={port}",
        f"--tensor-parallel-size={tensor_parallel_size}"
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    env['VLLM_USE_MODELSCOPE'] = 'true'

    # 保存 vLLM 启动命令到 sh 文件
    vllm_sh_file = LOGS_DIR / f"{model_name}_vllm.sh"
    with open(vllm_sh_file, 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"export CUDA_VISIBLE_DEVICES={gpu_idx}\n")
        f.write(f"export VLLM_USE_MODELSCOPE=true\n")
        f.write(f"{' '.join(vllm_cmd)}\n")

    # 重定向 vLLM 日志到文件
    with open(vllm_log_file, 'w') as vllm_log:
        vllm_process = subprocess.Popen(vllm_cmd, env=env, stdout=vllm_log, stderr=vllm_log)

    # 等待服务真正可用
    if not wait_for_service(port):
        vllm_process.terminate()
        vllm_process.wait()
        raise RuntimeError(f"vLLM 服务启动超时 (端口: {port})")

    try:
        # 运行评测
        eval_script = build_eval_script(model_path, eval_config, dataset_path, port, str(output_dir))
        tmp_script = LOGS_DIR / f"{model_name}_eval.py"
        with open(tmp_script, 'w') as f:
            f.write(eval_script)

        # 重定向 evalscope 日志到文件
        with open(eval_log_file, 'w') as eval_log:
            result = subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT, stdout=eval_log, stderr=eval_log)

        # 解析结果
        results = parse_evaluation_results(model_name, output_dir)
        return results

    finally:
        # 停止 vLLM 服务
        vllm_process.terminate()
        vllm_process.wait()


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


def save_summary_csv(all_results: List[Dict[str, Any]], output_path: str, dataset_name: str):
    """保存结果汇总 CSV（增量写入）"""
    import csv

    if not all_results:
        print("[SUMMARY] 没有结果需要保存")
        return

    # 添加测试集名称和时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 从结果中动态获取所有评测指标字段
    if all_results:
        # 固定字段
        fieldnames = ['dataset', 'timestamp', 'transform', 'quantization', 'model_path']
        # 动态添加评测指标字段（排除固定字段）
        first_result = all_results[0]
        for key in first_result.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    # 检查文件是否存在，确定是增量写入还是覆盖写入
    file_exists = Path(output_path).exists()
    
    # 如果文件存在，读取现有数据用于查找重复项
    existing_rows = {}
    if file_exists:
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 使用 transform + quantization + dataset 作为唯一键
                key = (row['transform'], row['quantization'], row['dataset'])
                existing_rows[key] = row

    # 写入数据
    write_mode = 'a' if file_exists else 'w'
    with open(output_path, write_mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果是新建文件，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入结果
        updated_count = 0
        for result in all_results:
            key = (result['transform'], result['quantization'], dataset_name)
            
            # 添加额外字段
            result['dataset'] = dataset_name
            result['timestamp'] = timestamp
            
            # 如果存在重复，更新（重新写入所有数据）
            if key in existing_rows:
                updated_count += 1
            
            writer.writerow(result)
    
    # 如果有更新，提示用户
    if updated_count > 0:
        logger.info(f"结果已保存到: {output_path} (更新了 {updated_count} 条记录)")
    else:
        logger.info(f"结果已保存到: {output_path} (新增 {len(all_results)} 条记录)")


def main():
    parser = argparse.ArgumentParser(description='量化评测自动化流程')
    parser.add_argument('--trans-config', type=str, default=None, nargs='+',
                        help='转换配置文件路径 (支持多个，默认使用 1-trans 目录下所有 YAML)')
    parser.add_argument('--quant-config', type=str, default=None, nargs='+',
                        help='量化配置文件路径 (支持多个，默认使用 2-quant 目录下所有 YAML)')
    parser.add_argument('--eval-config', type=str, default=None,
                        help='评测配置文件路径 (默认使用 3-eval 目录下的单个 YAML)')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='结果汇总 CSV 文件路径 (默认根据数据集名称自动生成)')

    args = parser.parse_args()

    # 加载配置
    trans_configs = []
    quant_configs = []

    # 加载评测配置
    if args.eval_config:
        eval_config = load_yaml_config(args.eval_config)
        logger.info(f"使用评测配置: {args.eval_config}")
    else:
        eval_dir = PIPELINE_DIR / "3-eval"
        eval_files = list(eval_dir.glob("*.yaml"))
        if len(eval_files) == 0:
            raise RuntimeError(f"评测配置目录为空: {eval_dir}")
        elif len(eval_files) > 1:
            raise RuntimeError(f"评测配置目录下有多个 YAML 文件: {[f.name for f in eval_files]}，请只保留一个")
        else:
            eval_config = load_yaml_config(eval_files[0])
            logger.info(f"使用评测配置: {eval_files[0].name}")

    # 根据 dataset_name 生成 CSV 文件名
    dataset_name = eval_config['dataset_name']
    if args.output_csv is None:
        args.output_csv = RESULTS_DIR / f"results_{dataset_name}.csv"
    else:
        args.output_csv = Path(args.output_csv)

    # 加载转换配置
    if args.trans_config:
        for cfg_path in args.trans_config:
            trans_configs.append(load_yaml_config(cfg_path))
    else:
        trans_dir = PIPELINE_DIR / "1-trans"
        for cfg_path in get_configs_from_dir(trans_dir):
            trans_configs.append(load_yaml_config(cfg_path))

    # 加载量化配置
    if args.quant_config:
        for cfg_path in args.quant_config:
            quant_configs.append(load_yaml_config(cfg_path))
    else:
        quant_dir = PIPELINE_DIR / "2-quant"
        for cfg_path in get_configs_from_dir(quant_dir):
            quant_configs.append(load_yaml_config(cfg_path))

    print(f"\n{'='*60}")
    print(f"配置: {len(trans_configs)} 转换 × {len(quant_configs)} 量化 = {len(trans_configs) * len(quant_configs)} 任务")
    print(f"  结果: {args.output_csv}")
    print(f"  日志: {LOGS_DIR}")
    print(f"{'='*60}\n")

    # 验证所有转换配置的 input_model 一致且存在
    base_model = None
    for i, trans_cfg in enumerate(trans_configs):
        if 'input_model' not in trans_cfg:
            raise RuntimeError(f"转换配置 {trans_cfg.get('name', i)} 缺少 input_model 字段")
        
        model_path = Path(trans_cfg['input_model'])
        if not model_path.exists():
            raise RuntimeError(f"基础模型路径不存在: {model_path}")
        
        # 检查是否配置一致
        if base_model is None:
            base_model = str(model_path)
        elif base_model != str(model_path):
            raise RuntimeError(f"转换配置的 input_model 不一致: {base_model} vs {model_path}")
    
    logger.info(f"基础模型路径: {base_model}")

    # 存储所有结果
    all_results = []

    # 准备评测数据集
    dataset_path = prepare_dataset(eval_config)

    # 阶段1: 顺序执行所有转换
    print(f"\n[PHASE 1] 执行转换 ({len(trans_configs)} 个配置)")

    trans_outputs = []
    for trans_cfg in trans_configs:
        logger.info(f"阶段1 - 转换: {trans_cfg['name']}")
        trans_output = run_transform(trans_cfg, base_model)
        trans_outputs.append((trans_cfg, trans_output))

    logger.info(f"阶段1完成 ({len(trans_outputs)} 个转换)")

    # 阶段2: 并行执行量化和评测
    total_tasks = len(trans_outputs) * len(quant_configs)
    max_workers = len(eval_config['gpu_resources']['devices']) // eval_config['gpu_resources']['tensor_parallel_size']
    logger.info(f"阶段2 - 并行启动 {total_tasks} 个任务 (并发数: {max_workers})")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_workers = len(eval_config['gpu_resources']['devices']) // eval_config['gpu_resources']['tensor_parallel_size']

    def process_task(task_idx, trans_cfg, trans_output, quant_cfg):
        """处理单个任务：量化 + 评测"""
        task_name = f"{trans_cfg['name']}-{quant_cfg['name']}"
        logger.info(f"[{task_idx + 1}/{len(trans_outputs) * len(quant_configs)}] 开始: {task_name}")

        try:
            # 执行量化
            quant_output = run_quantization(trans_output, quant_cfg)

            # 执行评测
            results = run_evaluation(quant_output, eval_config, dataset_path, task_idx, len(trans_outputs) * len(quant_configs))

            # 记录结果
            result = {
                'transform': trans_cfg['name'],
                'quantization': quant_cfg['name'],
                'model_path': quant_output,
                'mmlu': results.get('mmlu', 0.0),
                'gsm8k': results.get('gsm8k', 0.0),
            }
            logger.success(f"[{task_idx + 1}/{len(trans_outputs) * len(quant_configs)}] 完成: {task_name}")
            return result
        except Exception as e:
            logger.error(f"[{task_idx + 1}/{len(trans_outputs) * len(quant_configs)}] 失败: {task_name} - {e}")
            return None

    # 提交所有任务
    task_idx = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for trans_cfg, trans_output in trans_outputs:
            for quant_cfg in quant_configs:
                future = executor.submit(process_task, task_idx, trans_cfg, trans_output, quant_cfg)
                futures.append(future)
                task_idx += 1

        # 等待任务完成，每完成一个就保存结果
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)
                # 立即保存结果
                dataset_name = eval_config['dataset_name']
                save_summary_csv(all_results, args.output_csv, dataset_name)

    logger.success(f"所有任务完成! 结果: {args.output_csv}")

if __name__ == '__main__':
    main()