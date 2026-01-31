from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/workspace/lim42@xiaopeng.com/github/quant_example/Qwen3-8B-R1R2-rtn-W8-weightonly"
MODEL_DEST_PATH = MODEL_PATH + "_cleaned"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_DEST_PATH)
tokenizer.save_pretrained(MODEL_DEST_PATH)