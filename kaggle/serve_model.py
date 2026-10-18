# Kaggle notebook (GPU T4 x2, Internet on). Serves base model + your LoRA adapter over a tunnel.
# Add Kaggle secret API_KEY (any string) and put the same value in MODEL_API_KEY locally.

# %% install

# %% serve
import os, json, glob, subprocess, sys
def sh(c): subprocess.run(c, shell=True, check=True)
def secret(name, default=None):
    """env var -> secrets.json in an attached private dataset -> Kaggle secrets."""
    if os.environ.get(name): return os.environ[name]
    for p in glob.glob("/kaggle/input/*/secrets.json"):
        d = json.load(open(p))
        if name in d: return d[name]
    try:
                return UserSecretsClient().get_secret(name)
    except Exception:
        return default
sh("pip -q install -U transformers peft bitsandbytes accelerate fastapi uvicorn")
sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")
import glob, re, subprocess, threading, time, torch, uvicorn
from fastapi import FastAPI, Header, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

API_KEY = secret("API_KEY")
BASE = "Qwen/Qwen3-14B"
ADAPTER = (glob.glob("/kaggle/input/*/sama-lora") or ["/kaggle/working/sama-lora"])[0]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16))
model = PeftModel.from_pretrained(model, ADAPTER).eval()

app = FastAPI()

class Req(BaseModel):
    messages: list
    max_new_tokens: int = 768
    temperature: float = 0.3

@app.post("/chat")
def chat(req: Req, x_api_key: str = Header("")):
    if x_api_key != API_KEY:
        raise HTTPException(401)
    prompt = tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=req.max_new_tokens, do_sample=req.temperature > 0,
                             temperature=max(req.temperature, 1e-3))
    return {"answer": tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)}

threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True).start()
time.sleep(3)
p = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8000"],
                     stderr=subprocess.PIPE, text=True)
for line in p.stderr:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        print("MODEL_API_URL =", m.group(0))
        break
while True:
    time.sleep(60)
