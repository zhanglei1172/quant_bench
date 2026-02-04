#!/bin/bash

cd "$(dirname "$0")"

python run_pipeline.py \
  --trans-config pipline/1-trans/example/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/example/r4-gptq-w4a8-static.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml
