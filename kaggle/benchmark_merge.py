# Kaggle notebook (GPU T4 x2, Internet on). Blind-benchmarks base vs fine-tuned, then merges the adapter.
# bench.jsonl: {"question": "..."} per line, never used in training.
# Kaggle secrets: JUDGE_API_KEY (OpenAI-compatible key, e.g. OpenRouter); optional JUDGE_BASE_URL, JUDGE_MODEL.

# %% setup
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
sh("pip -q install -U transformers peft bitsandbytes accelerate")
import glob, json, random, re, torch, requests
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
KEY = secret("JUDGE_API_KEY")
JUDGE_URL = secret("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
JUDGE_MODEL = secret("JUDGE_MODEL", "deepseek/deepseek-chat")

BASE = "Qwen/Qwen3-14B"
ADAPTER = (glob.glob("/kaggle/input/*/sama-lora") or ["/kaggle/working/sama-lora"])[0]
BENCH = [json.loads(l)["question"] for l in open(glob.glob("/kaggle/input/*/bench.jsonl")[0])]
tok = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", quantization_config=bnb)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def gen(q):  # deterministic
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=600, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

# %% answers: adapter disabled = base, enabled = fine-tuned
base_ans, ft_ans = [], []
for q in BENCH:
    with model.disable_adapter():
        base_ans.append(gen(q))
    ft_ans.append(gen(q))

# %% blind paired judging (order randomized)
def _judge_prompt(q, a, b):
    return (f"Question:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\n"
            'Score each answer 1-5 for technical accuracy, completeness and clarity. Reply only with JSON: {"a": int, "b": int}')

def judge(q, a, b):
    prompt = _judge_prompt(q, a, b)
    if not KEY:  # free fallback: the local base model judges (self-preference bias; rough signal only)
        with model.disable_adapter():
            txt = gen(prompt)
        return json.loads(re.search(r"\{.*?\}", txt, re.S).group(0))
    r = requests.post(f"{JUDGE_URL}/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": JUDGE_MODEL, "temperature": 0,
                            "messages": [{"role": "user", "content": prompt}]}, timeout=120)
    return json.loads(re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S).group(0))

tot_base = tot_ft = wins = losses = ties = 0
for q, b, f in zip(BENCH, base_ans, ft_ans):
    swap = random.random() < 0.5
    s = judge(q, f, b) if swap else judge(q, b, f)
    sb, sf = (s["b"], s["a"]) if swap else (s["a"], s["b"])
    tot_base += sb; tot_ft += sf
    wins += sf > sb; losses += sf < sb; ties += sf == sb
n = len(BENCH)
print(f"base {tot_base/n:.3f}  fine-tuned {tot_ft/n:.3f}  ({(tot_ft-tot_base)/tot_base:+.2%})")
print(f"fine-tuned wins {wins}, losses {losses}, ties {ties}")

# %% merge adapter into full-precision weights (needs ~30 GB disk; /kaggle/working has ~20 GB,
# so save in fp16 to the working dir after freeing the adapter-run outputs, or merge on CPU)
del model; torch.cuda.empty_cache()
full = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(full, ADAPTER).merge_and_unload()
merged.save_pretrained("/kaggle/working/sama-merged", safe_serialization=True, max_shard_size="4GB")
tok.save_pretrained("/kaggle/working/sama-merged")
