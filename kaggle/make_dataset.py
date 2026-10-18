"""Generate train.jsonl / bench.jsonl with a teacher model (OpenAI-compatible API).
Usage: JUDGE_API_KEY=... python make_dataset.py topics.txt  (one defensive-security topic per line)
Benchmark questions come from separate generation calls and are deduplicated against training."""
import json, os, sys, requests

URL = os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-chat")
KEY = os.environ["JUDGE_API_KEY"]

def ask(prompt):
    r = requests.post(f"{URL}/chat/completions", headers={"Authorization": f"Bearer {KEY}"},
                      json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, timeout=180)
    return r.json()["choices"][0]["message"]["content"]

def questions(topic, n):
    out = ask(f"Write {n} distinct, technically precise questions a security engineer or analyst might ask "
              f"about '{topic}', focused on defense, detection, hardening, incident response and secure design. "
              "One per line, no numbering.")
    return [l.strip() for l in out.splitlines() if len(l.strip()) > 20]

topics = [t.strip() for t in open(sys.argv[1]) if t.strip()]
seen, train, bench = set(), [], []
for t in topics:
    qs = [q for q in questions(t, 55) if q not in seen]
    seen.update(qs)
    bench += qs[:1]
    for q in qs[1:]:
        train.append({"messages": [{"role": "user", "content": q},
                                   {"role": "assistant", "content": ask(q)}]})
with open("train.jsonl", "w") as f: [f.write(json.dumps(x) + "\n") for x in train]
with open("bench.jsonl", "w") as f: [f.write(json.dumps({"question": q}) + "\n") for q in bench]
print(len(train), "train,", len(bench), "bench")
