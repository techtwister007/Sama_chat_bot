# Kaggle notebook (GPU T4 x2). Paste each "# %%" block into a cell, or upload as a script.
# Input: a Kaggle dataset containing train.jsonl, one {"messages":[{"role":..,"content":..},...]} per line,
# and bench.jsonl kept OUT of training. Defensive-security Q&A you wrote or generated.

# %% install

# %% config
import os, json, glob, subprocess, sys
def sh(c): subprocess.run(c, shell=True, check=True)
def secret(name, default=None):
    """env var -> secrets.json in an attached private dataset -> Kaggle secrets."""
    if os.environ.get(name): return os.environ[name]
    for p in glob.glob("/kaggle/input/*/secrets.json"):
        d = json.load(open(p))
        if name in d: return d[name]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return default
sh("pip -q install -U transformers peft trl bitsandbytes datasets accelerate")
import glob, torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3-14B"   # ~9-10 GB in 4-bit; fits 2xT4 with room for activations
TRAIN = glob.glob("/kaggle/input/*/train.jsonl")[0]
OUT = "/kaggle/working/sama-lora"

# %% load
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16,  # T4: no bf16
                         bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")

# %% train
ds = load_dataset("json", data_files=TRAIN, split="train")
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
cfg = SFTConfig(output_dir=OUT, num_train_epochs=2, learning_rate=2e-4,
                per_device_train_batch_size=1, gradient_accumulation_steps=16,
                gradient_checkpointing=True, fp16=True, logging_steps=10,
                warmup_ratio=0.03, lr_scheduler_type="cosine", max_length=1536,
                save_strategy="epoch", report_to="none")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora, processing_class=tok)
trainer.train()
trainer.save_model(OUT)   # adapter only (small); download or attach as a dataset
