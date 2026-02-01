#!/bin/bash

cd "$(dirname "$0")"

python run_pipeline.py \
  --trans-config pipline/1-trans/example/spinquant-r1r2.yaml \
  --quant-config pipline/2-quant/example/gptq-w8-weightonly.yaml pipline/2-quant/example/gptq-w8a8-dynamic.yaml pipline/2-quant/example/gptq-w8a8-static.yaml pipline/2-quant/example/gptq-w8a8-static-ignore-down.yaml pipline/2-quant/example/rtn-w8-weightonly.yaml \
  --eval-config pipline/3-eval/example/mmlu_gsm8k.yaml

curl https://api.day.app/9bAGrDYKbZxfKaQNYMVYBi/QuantizeDone