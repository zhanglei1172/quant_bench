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

# 添加 rich 进度显示
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from enum import Enum

class TaskStatus(Enum):
    PENDING = "⏳ 等待"
    RUNNING = "🔄 进行中"
    COMPLETED = "✅ 完成"
    SKIPPED = "⏭️ 跳过"
    FAILED = "❌ 失败"

class TaskTracker:
    """任务状态追踪器（线程安全）"""
    def __init__(self):
        self.tasks = {}  # task_id -> {name, status, gpu, message}
        self.lock = Lock()
        self.console = Console()
        self.live = None
        self.start_time = time.time()
    
    def add_task(self, task_id: str, name: str, task_type: str):
        """添加新任务"""
        with self.lock:
            self.tasks[task_id] = {
                'name': name,
                'type': task_type,
                'status': TaskStatus.PENDING,
                'gpu': None,
                'message': '',
                'start_time': None,
                'end_time': None,
            }
    
    def update_status(self, task_id: str, status: TaskStatus, gpu: List[int] = None, message: str = ''):
        """更新任务状态"""
        with self.lock:
            if task_id is not None and task_id in self.tasks:
                self.tasks[task_id]['status'] = status
                if gpu is not None:
                    self.tasks[task_id]['gpu'] = gpu
                if message:
                    self.tasks[task_id]['message'] = message
                
                if status == TaskStatus.RUNNING and self.tasks[task_id]['start_time'] is None:
                    self.tasks[task_id]['start_time'] = time.time()
                elif status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]:
                    self.tasks[task_id]['end_time'] = time.time()
        
        self.refresh()
    
    def generate_table(self) -> Table:
        """生成进度表格"""
        table = Table(title="🚀 量化评测流水线进度", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=4)
        table.add_column("类型", width=8)
        table.add_column("任务名称", width=40)
        table.add_column("状态", width=12)
        table.add_column("GPU", width=8)
        table.add_column("耗时", width=10)
        table.add_column("备注", width=30)
        
        with self.lock:
            # 按类型和ID排序
            sorted_tasks = sorted(
                self.tasks.items(),
                key=lambda x: (
                    0 if x[1]['type'] == '基线' else 1,
                    x[0]
                )
            )
            
            for task_id, info in sorted_tasks:
                # 状态样式
                status = info['status']
                if status == TaskStatus.COMPLETED:
                    status_text = Text(status.value, style="bold green")
                elif status == TaskStatus.RUNNING:
                    status_text = Text(status.value, style="bold yellow")
                elif status == TaskStatus.FAILED:
                    status_text = Text(status.value, style="bold red")
                elif status == TaskStatus.SKIPPED:
                    status_text = Text(status.value, style="bold cyan")
                else:
                    status_text = Text(status.value, style="dim")
                
                # GPU 显示
                gpu_text = ",".join(map(str, info['gpu'])) if info['gpu'] else "-"
                
                # 耗时计算
                if info['start_time']:
                    if info['end_time']:
                        elapsed = info['end_time'] - info['start_time']
                    else:
                        elapsed = time.time() - info['start_time']
                    elapsed_text = f"{elapsed:.1f}s"
                else:
                    elapsed_text = "-"
                
                # 任务名称截断
                name = info['name']
                if len(name) > 38:
                    name = name[:35] + "..."
                
                table.add_row(
                    task_id.split('-')[-1],  # 只显示数字ID
                    info['type'],
                    name,
                    status_text,
                    gpu_text,
                    elapsed_text,
                    info['message'][:28] + "..." if len(info['message']) > 30 else info['message']
                )
            
            # 添加统计信息
            stats = self.get_statistics()
            total_time = time.time() - self.start_time
            
            table.caption = (
                f"总任务: {stats['total']} | "
                f"完成: {stats['completed']} | "
                f"跳过: {stats['skipped']} | "
                f"进行中: {stats['running']} | "
                f"等待: {stats['pending']} | "
                f"失败: {stats['failed']} | "
                f"总耗时: {total_time:.1f}s"
            )
        
        return table
    
    def get_statistics(self) -> Dict[str, int]:
        """获取统计信息"""
        stats = {
            'total': len(self.tasks),
            'pending': 0,
            'running': 0,
            'completed': 0,
            'skipped': 0,
            'failed': 0,
        }
        
        for info in self.tasks.values():
            status = info['status']
            if status == TaskStatus.PENDING:
                stats['pending'] += 1
            elif status == TaskStatus.RUNNING:
                stats['running'] += 1
            elif status == TaskStatus.COMPLETED:
                stats['completed'] += 1
            elif status == TaskStatus.SKIPPED:
                stats['skipped'] += 1
            elif status == TaskStatus.FAILED:
                stats['failed'] += 1
        
        return stats
    
    def start_live_display(self):
        """启动实时显示"""
        self.live = Live(self.generate_table(), refresh_per_second=4, console=self.console)  # 提高刷新频率
        self.live.start()
    
    def stop_live_display(self):
        """停止实时显示"""
        if self.live:
            self.live.stop()
    
    def refresh(self):
        """刷新显示"""
        if self.live:
            try:
                self.live.update(self.generate_table())
            except Exception:
                # 忽略刷新过程中的异常（例如终端大小变化）
                pass

# 全局任务追踪器
task_tracker = TaskTracker()

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


# GPU 分配器（线程安全）
class GPUAllocator:
    def __init__(self, devices: List[int], pipeline_parallel_size: int = 1):
        self.devices = devices
        self.pipeline_parallel_size = pipeline_parallel_size
        self.lock = Lock()
        # 计算可用的 GPU 组
        self.num_groups = len(devices) // pipeline_parallel_size
        if self.num_groups == 0:
            raise ValueError(f"GPU 数量 ({len(devices)}) 不足以支持 pipeline_parallel_size={pipeline_parallel_size}")
        # 记录每个 GPU 组是否被占用
        self.group_in_use = [False] * self.num_groups
        self.condition = None  # 延迟初始化
    
    def _get_condition(self):
        """延迟初始化 Condition 对象"""
        if self.condition is None:
            from threading import Condition
            self.condition = Condition(self.lock)
        return self.condition
    
    def allocate(self, timeout: float = None) -> List[int]:
        """分配一组 GPU，如果没有可用的则等待"""
        import time
        condition = self._get_condition()
        
        start_time = time.time()
        with condition:
            while True:
                # 查找空闲的 GPU 组
                for group_idx in range(self.num_groups):
                    if not self.group_in_use[group_idx]:
                        self.group_in_use[group_idx] = True
                        start_gpu = group_idx * self.pipeline_parallel_size
                        gpus = self.devices[start_gpu:start_gpu + self.pipeline_parallel_size]
                        logger.debug(f"分配 GPU 组 {group_idx}: {gpus}")
                        return gpus
                
                # 没有空闲的 GPU 组，等待
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise TimeoutError("等待 GPU 分配超时")
                    condition.wait(remaining)
                else:
                    condition.wait()
    
    def release(self, gpus: List[int]):
        """释放 GPU 组"""
        condition = self._get_condition()
        
        with condition:
            # 根据 GPU 列表找到对应的组
            if len(gpus) > 0:
                first_gpu = gpus[0]
                try:
                    idx = self.devices.index(first_gpu)
                    group_idx = idx // self.pipeline_parallel_size
                    if 0 <= group_idx < self.num_groups:
                        self.group_in_use[group_idx] = False
                        logger.debug(f"释放 GPU 组 {group_idx}: {gpus}")
                        condition.notify_all()
                except ValueError:
                    pass

# 全局 GPU 分配器（在 main 中初始化）
gpu_allocator: GPUAllocator = None


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 检查配置是否为空，添加 dummy 键避免后续处理出错
    if config is None:
        config = {'_empty': True}
    
    # 从文件名提取 name 和 output_suffix
    # 如果配置文件中没有这些字段，则使用文件名
    filename = path.stem  # 不含扩展名的文件名
    if 'name' not in config:
        config['name'] = filename
    if 'output_suffix' not in config:
        config['output_suffix'] = f'-{filename}'
    
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
    """执行模型转换（本质上是第一步的量化）"""
    output_suffix = config.get('output_suffix')

    # 如果 output_suffix 为 None，直接使用基础模型路径（original 基线）
    if output_suffix is None:
        logger.info(f"使用原始模型: {config['name']}")
        return base_model

    # 拼接输出路径
    output_model = base_model + output_suffix

    # 检查模型是否已存在
    if Path(output_model).exists():
        logger.info(f"跳过 {config['name']} (模型已存在)")
        return output_model

    logger.info(f"转换 {config['name']}")

    # 复用量化脚本生成逻辑
    quant_script = build_quant_script(base_model, output_model, config)

    # 保存临时脚本到 logs 目录（不删除，用于 debug）
    model_name = Path(output_model).name
    tmp_script = LOGS_DIR / f"{model_name}_trans.py"
    with open(tmp_script, 'w') as f:
        f.write(quant_script)

    subprocess.run([sys.executable, str(tmp_script)], check=True, cwd=PROJECT_ROOT)
    
    # 成功后将脚本重命名为隐藏文件
    hidden_script = LOGS_DIR / f".{model_name}_trans.py"
    tmp_script.rename(hidden_script)
    
    return output_model


def run_quantization(input_model: str, quant_config: Dict[str, Any], gpu_devices: List[int] = None, task_id: str = None) -> str:
    """执行模型量化"""
    quant_name = quant_config['name']

    # 构建输出路径
    quant_suffix = quant_config.get('output_suffix', '')
    output_model = input_model + quant_suffix
    model_name = Path(output_model).name

    # 更新任务状态
    task_tracker.update_status(task_id, TaskStatus.RUNNING, gpu_devices, "正在量化...")

    # 检查模型是否已存在
    if Path(output_model).exists():
        logger.info(f"跳过 {quant_name} (模型已存在)")
        task_tracker.update_status(task_id, TaskStatus.SKIPPED, message="模型已存在")
        return output_model

    logger.info(f"量化 {quant_name} (GPU: {gpu_devices})")

    # 构建量化脚本
    quant_script = build_quant_script(input_model, output_model, quant_config)

    # 保存临时脚本到 logs 目录（不删除，用于 debug）
    tmp_script = LOGS_DIR / f"{model_name}_quant.py"
    with open(tmp_script, 'w') as f:
        f.write(quant_script)

    # 日志文件
    log_file = LOGS_DIR / f"{model_name}_llmcompressor.log"

    # 设置环境变量：限制量化只使用指定的GPU
    env = os.environ.copy()
    if gpu_devices:
        # 将GPU列表转换为逗号分隔的字符串
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_devices))

    with open(log_file, 'w') as log_f:
        result = subprocess.run([sys.executable, str(tmp_script)], check=False, cwd=PROJECT_ROOT, stdout=log_f, stderr=log_f, env=env)
    
    if result.returncode != 0:
        task_tracker.update_status(task_id, TaskStatus.FAILED, message="量化失败")
        raise RuntimeError(f"量化失败: {quant_name}")
    
    # 成功后将脚本和日志重命名为隐藏文件
    hidden_script = LOGS_DIR / f".{model_name}_quant.py"
    hidden_log = LOGS_DIR / f".{model_name}_llmcompressor.log"
    tmp_script.rename(hidden_script)
    log_file.rename(hidden_log)
    
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

    # 默认 save_compressed 为 False
    if save_compressed is None:
        save_compressed = False

    # 从配置中读取 dtype（默认 auto）
    dtype = config.get('dtype', 'auto')

    # 是否使用原始 config.json 替换
    original_config_json = config.get('original_config_json', False)

    script = f"""
import torch
import yaml
import json
import os
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
{'from datasets import load_dataset' if not is_data_free else ''}

MODEL_ID = "{input_model}"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="{dtype}")
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
    )
finally:
    os.unlink(recipe_path)
"""

    script += f"""
SAVE_DIR = "{output_model}"

# 当 save_compressed=False 时，使用权重移植方式保存（避免量化相关配置污染）
if {not save_compressed}:
    import gc
    print(">>> [1/3] Quantization complete, preparing weight transplant...")
    
    # 从原始路径加载纯净模型结构到 CPU，使用相同的 dtype
    clean_model = AutoModelForCausalLM.from_pretrained(
        "{input_model}",
        dtype="{dtype}",
        device_map="cpu",
    )
    
    print(">>> [2/3] Transplanting quantized weights to clean model...")
    
    # 复制权重（GPU 量化模型 -> CPU 纯净模型）
    quantized_state_dict = model.state_dict()
    clean_state_dict = clean_model.state_dict()
    
    for name in clean_state_dict.keys():
        if name in quantized_state_dict:
            try:
                if clean_state_dict[name].shape == quantized_state_dict[name].shape:
                    clean_state_dict[name].copy_(quantized_state_dict[name].to("cpu"))
            except Exception as e:
                print(f"Warning: Failed to copy {{name}}: {{e}}")
    
    # 释放原模型显存
    del model
    del quantized_state_dict
    gc.collect()
    
    print(">>> [3/3] Saving clean model...")
    model = clean_model
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
else:
    model.save_pretrained(SAVE_DIR, save_compressed={save_compressed})

tokenizer.save_pretrained(SAVE_DIR)

# 当 original_config_json=True 时，使用原始的 config.json
if {original_config_json}:
    import shutil
    original_config_path = os.path.join("{input_model}", "config.json")
    new_config_path = os.path.join(SAVE_DIR, "config.json")
    
    if os.path.exists(original_config_path):
        # 如果新配置已存在，先备份
        if os.path.exists(new_config_path):
            shutil.copy2(new_config_path, new_config_path + ".before_original")
        # 复制原始配置
        shutil.copy2(original_config_path, new_config_path)
        print(f"Replaced config.json with original from {{original_config_path}}")

if True:
    import json
    config_path = os.path.join(SAVE_DIR, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # 修改 quantization_config 中的 num_bits
        if 'quantization_config' in config_data:
            quant_config = config_data['quantization_config']
            
            # 遍历所有 config_groups
            if 'config_groups' in quant_config:
                for group_name, group_config in quant_config['config_groups'].items():
                    # 只修改包含 input_activations 的配置组（激活量化组）
                    if 'input_activations' in group_config and group_config['input_activations'] is not None:
                        if 'weights' in group_config and 'num_bits' in group_config['weights']:
                            old_bits = group_config['weights']['num_bits']
                            if old_bits == 4:
                                group_config['weights']['num_bits'] = 8
                                print(f"Modified {{group_name}} weights num_bits from 4 to 8 (W4A8 group)")
                    else:
                        print(f"Skipped {{group_name}} (weight-only quantization, no input_activations)")
            
            # 保存修改后的配置
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Updated config.json for w4a8 compatibility (stored as int8, loaded as w8a8)")
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


def wait_for_service(port: int, timeout: int = 400) -> bool:
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


def run_evaluation(model_path: str, eval_config: Dict[str, Any], dataset_path: str, task_idx: int, total_tasks: int, gpu_devices: List[int] = None, task_id: str = None) -> Dict[str, float]:
    """执行评测"""
    model_name = Path(model_path).name
    port = find_available_port()

    # 获取 GPU 资源配置
    gpu_resources = eval_config.get('gpu_resources', {})
    pipeline_parallel_size = gpu_resources.get('pipeline_parallel_size', 1)

    # 如果没有指定 gpu_devices，使用全局分配器分配
    allocated_gpus = None
    if gpu_devices is None:
        if gpu_allocator is not None:
            gpu_devices = gpu_allocator.allocate()
            allocated_gpus = gpu_devices  # 记录需要释放的 GPU
        else:
            # 降级：使用默认设备
            devices = gpu_resources.get('devices', [0])
            gpu_devices = devices[:pipeline_parallel_size]

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
        # 更新任务状态
        score = results.get('overall_score', 0)
        task_tracker.update_status(task_id, TaskStatus.SKIPPED, message=f"分数: {score}")
        # 释放 GPU
        if allocated_gpus is not None and gpu_allocator is not None:
            gpu_allocator.release(allocated_gpus)
        return results

    # 更新任务状态
    task_tracker.update_status(task_id, TaskStatus.RUNNING, gpu_devices, "启动vLLM服务...")

    logger.info(f"评测 {model_name} (GPU:{gpu_devices})")

    # 启动 vLLM 服务
    vllm_cmd = [
        "vllm", "serve", model_path,
        f"--port={port}",
        f"--pipeline-parallel-size={pipeline_parallel_size}"
    ]

    env = os.environ.copy()
    # 将GPU列表转换为逗号分隔的字符串
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_devices))

    # 保存 vLLM 启动命令到 sh 文件
    vllm_sh_file = LOGS_DIR / f"{model_name}_vllm.sh"
    with open(vllm_sh_file, 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"export CUDA_VISIBLE_DEVICES={','.join(map(str, gpu_devices))}\n")
        f.write(f"{' '.join(vllm_cmd)}\n")

    # 重定向 vLLM 日志到文件
    with open(vllm_log_file, 'w') as vllm_log:
        vllm_process = subprocess.Popen(vllm_cmd, env=env, stdout=vllm_log, stderr=vllm_log)

    # 等待服务真正可用
    logger.info(f"等待 vLLM 服务启动 (端口: {port})...")
    task_tracker.update_status(task_id, TaskStatus.RUNNING, gpu_devices, "等待vLLM启动...")
    
    service_ready = wait_for_service(port)
    if not service_ready:
        logger.error(f"vLLM 服务启动超时 (端口: {port})")
        vllm_process.terminate()
        vllm_process.wait()
        task_tracker.update_status(task_id, TaskStatus.FAILED, message="vLLM启动超时")
        # 释放 GPU
        if allocated_gpus is not None and gpu_allocator is not None:
            gpu_allocator.release(allocated_gpus)
        raise RuntimeError(f"vLLM 服务启动超时 (端口: {port})")
    logger.info(f"vLLM 服务启动成功 {model_name} (端口: {port}，GPU: {gpu_devices})")
    
    task_tracker.update_status(task_id, TaskStatus.RUNNING, gpu_devices, "正在评测...")

    try:
        # 运行评测
        eval_script = build_eval_script(model_path, eval_config, dataset_path, port, str(output_dir))
        tmp_script = LOGS_DIR / f"{model_name}_eval.py"
        with open(tmp_script, 'w') as f:
            f.write(eval_script)

        # 重定向 evalscope 日志到文件
        with open(eval_log_file, 'w') as eval_log:
            result = subprocess.run([sys.executable, str(tmp_script)], check=False, cwd=PROJECT_ROOT, stdout=eval_log, stderr=eval_log)

        if result.returncode != 0:
            task_tracker.update_status(task_id, TaskStatus.FAILED, message="评测失败")
            raise RuntimeError(f"评测失败: {model_name}")

        # 解析结果
        results = parse_evaluation_results(model_name, output_dir)
        
        # 更新任务状态
        overall_score = results.get('overall_score', 0)
        task_tracker.update_status(task_id, TaskStatus.COMPLETED, message=f"分数: {overall_score:.4f}")
        
        # 成功后将所有相关文件重命名为隐藏文件
        hidden_vllm_sh = LOGS_DIR / f".{model_name}_vllm.sh"
        hidden_vllm_log = LOGS_DIR / f".{model_name}_vllm.log"
        hidden_eval_script = LOGS_DIR / f".{model_name}_eval.py"
        hidden_eval_log = LOGS_DIR / f".{model_name}_evalscope.log"
        
        vllm_sh_file.rename(hidden_vllm_sh)
        vllm_log_file.rename(hidden_vllm_log)
        tmp_script.rename(hidden_eval_script)
        eval_log_file.rename(hidden_eval_log)
        
        return results

    except Exception as e:
        task_tracker.update_status(task_id, TaskStatus.FAILED, message=str(e)[:30])
        raise
    finally:
        # 停止 vLLM 服务
        vllm_process.terminate()
        vllm_process.wait()
        # 释放 GPU
        if allocated_gpus is not None and gpu_allocator is not None:
            gpu_allocator.release(allocated_gpus)


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
    
    # 1. 读取总体分数 (data_collection.json)
    data_collection_path = reports_dir / "data_collection.json"
    if data_collection_path.exists():
        with open(data_collection_path, 'r') as f:
            data = json.load(f)
            # 提取总体分数
            overall_score = data.get('score')
            if overall_score is not None:
                results['overall_score'] = overall_score
    
    # 2. 读取数据集级别分数 (collection_detailed_report.json)
    report_path = reports_dir / "collection_detailed_report.json"
    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
            # 从 dataset_level 中提取各数据集的 weighted_avg 分数
            dataset_level = report.get('dataset_level', [])
            for dataset in dataset_level:
                dataset_name = dataset.get('dataset_name', '')
                weighted_avg = dataset.get('weighted_avg.', 0.0)
                if dataset_name:
                    results[dataset_name] = weighted_avg

    return results


def save_summary_csv(all_results: List[Dict[str, Any]], output_path: str, dataset_name: str):
    """保存结果汇总 CSV（增量写入，新结果按 overall_score 降序排列）"""
    import csv

    if not all_results:
        print("[SUMMARY] 没有结果需要保存")
        return

    # 添加测试集名称和时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 从结果中动态获取所有评测指标字段
    if all_results:
        # 固定字段（调整顺序，将 overall_score 放在前面）
        fieldnames = ['dataset', 'timestamp', 'transform', 'quantization', 'overall_score', 'model_path']
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

    # 新结果按 overall_score 降序排序
    def get_overall_score(result):
        try:
            return float(result.get('overall_score', 0))
        except (ValueError, TypeError):
            return 0.0
    
    all_results.sort(key=get_overall_score, reverse=True)

    # 写入数据（追加模式）
    write_mode = 'a' if file_exists else 'w'
    with open(output_path, write_mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # 如果是新建文件，写入表头
        if not file_exists:
            writer.writeheader()
        
        # 写入结果
        for result in all_results:
            key = (result['transform'], result['quantization'], dataset_name)
            
            # 添加额外字段
            result['dataset'] = dataset_name
            result['timestamp'] = timestamp
            
            writer.writerow(result)
    
    logger.info(f"结果已保存到: {output_path} (新增 {len(all_results)} 条记录)")


def build_result(transform_name: str, quant_name: str, model_path: str, eval_results: Dict[str, float]) -> Dict[str, Any]:
    """构建结果字典，动态包含所有评测字段"""
    result = {
        'transform': transform_name,
        'quantization': quant_name,
        'model_path': model_path,
    }
    # 动态添加所有评测结果字段
    for key, value in eval_results.items():
        result[key] = value
    return result


def main():
    global gpu_allocator
    
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

    # 初始化全局 GPU 分配器
    gpu_devices = eval_config['gpu_resources']['devices']
    pipeline_parallel_size = eval_config['gpu_resources']['pipeline_parallel_size']
    gpu_allocator = GPUAllocator(gpu_devices, pipeline_parallel_size)
    max_workers = gpu_allocator.num_groups
    
    logger.info(f"GPU 配置: {len(gpu_devices)} 个 GPU, PP={pipeline_parallel_size}, 最大并发={max_workers}")

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

    # 从评测配置获取基础模型路径
    if 'base_model' not in eval_config:
        raise RuntimeError(f"评测配置缺少 base_model 字段")

    base_model = Path(eval_config['base_model'])
    if not base_model.exists():
        raise RuntimeError(f"基础模型路径不存在: {base_model}")

    base_model_str = str(base_model)
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
        trans_output = run_transform(trans_cfg, base_model_str)
        trans_outputs.append((trans_cfg, trans_output))

    logger.info(f"阶段1完成 ({len(trans_outputs)} 个转换)")

    # 阶段2: 并行执行量化和评测
    total_baseline = len(trans_outputs) + 1
    total_quant = len(trans_outputs) * len(quant_configs)
    total_tasks = total_baseline + total_quant
    logger.info(f"阶段2 - 并行启动 {total_tasks} 个任务 (基线: {total_baseline}, 量化: {total_quant}, 并发: {max_workers})")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 初始化任务追踪器
    task_idx = 0
    baseline_models = [('original', base_model_str)]
    baseline_models.extend([(cfg['name'], out) for cfg, out in trans_outputs])
    
    # 添加基线任务
    for model_name, model_path in baseline_models:
        task_id = f"baseline-{task_idx}"
        task_tracker.add_task(task_id, model_name, "基线")
        task_idx += 1
    
    # 添加量化任务
    for trans_cfg, trans_output in trans_outputs:
        for quant_cfg in quant_configs:
            task_id = f"quant-{task_idx}"
            task_name = f"{trans_cfg['name']}-{quant_cfg['name']}"
            task_tracker.add_task(task_id, task_name, "量化")
            task_idx += 1
    
    # 启动实时进度显示
    task_tracker.start_live_display()

    # 提交所有任务
    task_idx = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 第一步：提交基线模型评测任务
        for model_name, model_path in baseline_models:
            current_idx = task_idx
            task_id = f"baseline-{current_idx}"
            
            def eval_baseline(idx=current_idx, tid=task_id, name=model_name, path=model_path):
                try:
                    task_tracker.update_status(tid, TaskStatus.RUNNING, message="准备评测...")
                    # GPU 由 run_evaluation 内部自动分配和释放
                    results = run_evaluation(path, eval_config, dataset_path, idx, total_baseline, task_id=tid)
                    result = build_result(name, 'baseline', path, results)
                    logger.success(f"[基线 {idx + 1}/{total_baseline}] 完成: {name}")
                    return result
                except Exception as e:
                    logger.error(f"[基线 {idx + 1}/{total_baseline}] 失败: {name} - {e}")
                    task_tracker.update_status(tid, TaskStatus.FAILED, message=str(e)[:30])
                    return None
            futures.append(executor.submit(eval_baseline))
            task_idx += 1
        
        # 第二步：提交量化+评测任务
        for trans_cfg, trans_output in trans_outputs:
            for quant_cfg in quant_configs:
                current_idx = task_idx
                task_id = f"quant-{current_idx}"
                trans_name = trans_cfg['name']
                quant_name = quant_cfg['name']
                
                def process_quant(idx=current_idx, tid=task_id, t_name=trans_name, q_name=quant_name, t_out=trans_output, q_cfg=quant_cfg):
                    gpus = None
                    try:
                        task_name = f"{t_name}-{q_name}"
                        
                        # 分配 GPU
                        gpus = gpu_allocator.allocate()
                        task_tracker.update_status(tid, TaskStatus.RUNNING, gpus, "准备量化...")

                        quant_output = run_quantization(t_out, q_cfg, gpus, task_id=tid)
                        task_tracker.update_status(tid, TaskStatus.RUNNING, gpus, "准备评测...")
                        # 评测时传入已分配的 GPU，避免重复分配
                        results = run_evaluation(quant_output, eval_config, dataset_path, idx, total_tasks, gpus, task_id=tid)

                        result = build_result(t_name, q_name, quant_output, results)
                        logger.success(f"[量化 {idx - total_baseline + 1}/{total_quant}] 完成: {task_name}")
                        return result
                    except Exception as e:
                        logger.error(f"[量化 {idx - total_baseline + 1}/{total_quant}] 失败: {t_name}-{q_name} - {e}")
                        task_tracker.update_status(tid, TaskStatus.FAILED, message=str(e)[:30])
                        return None
                    finally:
                        # 确保 GPU 被释放
                        if gpus is not None:
                            gpu_allocator.release(gpus)

                futures.append(executor.submit(process_quant))
                task_idx += 1

        # 等待任务完成，收集结果
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_results.append(result)

        # 停止实时显示
        task_tracker.stop_live_display()
        
        # 显示最终结果表格
        print("\n" + "="*80)
        print("最终结果:")
        print("="*80)
        task_tracker.console.print(task_tracker.generate_table())

        # 所有任务完成后统一保存结果
        dataset_name = eval_config['dataset_name']
        save_summary_csv(all_results, args.output_csv, dataset_name)

    logger.success(f"所有任务完成! 结果: {args.output_csv}")

if __name__ == '__main__':
    main()