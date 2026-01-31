from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType

DATASET_PATH = 'test/qwen3_8b-text_mmlu_gsm8k_1024.jsonl'

task_kwargs_template = dict(
    model = 'Qwen/Qwen3-8B',
    api_url='http://127.0.0.1:8001/v1/chat/completions',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=['data_collection'],
    dataset_args={
        'data_collection': {
            'local_path': DATASET_PATH,
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 4096,
        'temperature': 0.0,
    },
    # limit=10
)

tasks = {
    # '0': TaskConfig(**{**task_kwargs_template, 
    #     'model': '/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2-gptq-W8-weightonly_cleaned',
    #     'api_url': 'http://127.0.0.1:8000/v1/chat/completions',
    # }),
    # '1': TaskConfig(**{**task_kwargs_template, 
    #     'model': '/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2-rtn-W8-weightonly_cleaned',
    #     'api_url': 'http://127.0.0.1:8001/v1/chat/completions',
    # }),
    # '2': TaskConfig(**{**task_kwargs_template, 
    #     'model': '/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2-R4-gptq-W8A8-dynamic',
    #     'api_url': 'http://127.0.0.1:8002/v1/chat/completions',
    # }),
    '3': TaskConfig(**{**task_kwargs_template, 
        'model': '/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2-gptq-ignore-down-W8A8-static',
        'api_url': 'http://127.0.0.1:8003/v1/chat/completions',
    }),
}

def prepare_dataset():
    from evalscope.collections import CollectionSchema, DatasetInfo, WeightedSampler
    from evalscope.utils.io_utils import dump_jsonl_data

    schema = CollectionSchema(name='Qwen3-8B-Text', datasets=[
        CollectionSchema(name='PureText', weight=1, datasets=[
            DatasetInfo(name='mmlu', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
            DatasetInfo(name='gsm8k', weight=1, task_type='math', tags=['en'], args={'few_shot_num': 0}),
        ]),
        # CollectionSchema(name='Vision', weight=2, datasets=[
        #     DatasetInfo(name='mmmu', weight=1, task_type='exam', tags=['en'], args={'few_shot_num': 0}),
        # ]),
    ])
    # get the mixed data
    mixed_data = WeightedSampler(schema).sample(1024)
    # dump the mixed data to a jsonl file
    dump_jsonl_data(mixed_data, DATASET_PATH)

if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    import os
    if not os.path.exists(DATASET_PATH):
        prepare_dataset()
    
    print(f'Starting evaluation for tasks: {list(tasks.keys())}')
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        executor.map(run_task, tasks.values())