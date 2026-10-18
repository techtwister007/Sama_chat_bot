# Kaggle GPU notebook/script: zero-API data generation. A local open model writes questions and answers.
# Output: /kaggle/working/train.jsonl and bench.jsonl (use as input to training via kernel_sources).
import json, glob, os, random, subprocess, torch
subprocess.run("pip -q install -U transformers accelerate bitsandbytes", shell=True, check=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GEN = "Qwen/Qwen3-14B"
PER_TOPIC = int(os.environ.get("PER_TOPIC", 25))   # 64 topics x 25 ~ 1,600 examples (keeps runtime ~hours)
TOPICS = [t.strip() for t in open(glob.glob("/kaggle/input/*/topics.txt")[0]) if t.strip()]

tok = AutoTokenizer.from_pretrained(GEN); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    GEN, device_map="auto", quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()

def gen(prompts, max_new=500, temp=0.7):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for p in prompts]
    ids = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.9)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

seen, items = set(), []
for t in TOPICS:
    out = gen([f"Write {PER_TOPIC + 2} distinct, technically precise questions a security engineer might ask about "
               f"'{t}', focused on defense, detection, hardening, incident response and secure design. "
               "One per line, no numbering."], max_new=900)[0]
    for l in out.splitlines():
        l = l.strip(" -*0123456789.")
        if len(l) > 20 and l not in seen:
            seen.add(l); items.append(l)

random.seed(0); random.shuffle(items)
bench, train_q = items[:60], items[60:]
answers = []
for i in range(0, len(train_q), 8):
    answers += gen(["Answer accurately and concisely, with practical steps:\n" + q for q in train_q[i:i + 8]], max_new=450, temp=0.4)
    print(i, "/", len(train_q), flush=True)

with open("/kaggle/working/train.jsonl", "w") as f:
    for q, a in zip(train_q, answers):
        f.write(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}) + "\n")
with open("/kaggle/working/bench.jsonl", "w") as f:
    for q in bench:
        f.write(json.dumps({"question": q}) + "\n")
