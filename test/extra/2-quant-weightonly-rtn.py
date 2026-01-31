import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier, GPTQModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "/dataset/workspace/lim42/datasets/ultra_chat_200k/"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Confirm transfromed model generations look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# # Load dataset and preprocess.
# ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
# ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# def preprocess(example):
#     return {
#         "text": tokenizer.apply_chat_template(
#             example["messages"],
#             tokenize=False,
#         )
#     }

# ds = ds.map(preprocess)

# # Tokenize inputs.
# def tokenize(sample):
#     return tokenizer(
#         sample["text"],
#         padding=False,
#         max_length=MAX_SEQUENCE_LENGTH,
#         truncation=True,
#         add_special_tokens=False,
#     )


# ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = """
 quant_stage:
   quant_modifiers:
    QuantizationModifier:
      ignore: ["lm_head", "re:visual.*", "re:model.visual.*"]
    #   kv_cache_scheme: {num_bits: 8, type: int, symmetric: true, strategy: tensor, dynamic: false}
      config_groups:
        group_0:
          weights: {num_bits: 8, type: int, symmetric: true, strategy: channel}
        #   input_activations: {num_bits: 8, type: int, symmetric: true, strategy: tensor, dynamic: false}
          targets: [Linear]
"""

# Apply algorithms.
oneshot(
    model=model,
    recipe=recipe,
    pipeline="datafree"
)

# oneshot(
#     model=model,
#     dataset=ds,
#     recipe=recipe,
#     max_seq_length=MAX_SEQUENCE_LENGTH,
#     num_calibration_samples=NUM_CALIBRATION_SAMPLES,
#     save_compressed=False,
#     trust_remote_code_model=True,
# )

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk.
SAVE_DIR_QUANT = MODEL_ID + "-rtn-W8-weightonly"
model.save_pretrained(SAVE_DIR_QUANT, save_compressed=False)
tokenizer.save_pretrained(SAVE_DIR_QUANT)

# Remove quantization_config from config.json if it exists
import json
import os

config_path = os.path.join(SAVE_DIR_QUANT, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    if "quantization_config" in config:
        del config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Removed quantization_config from {config_path}")