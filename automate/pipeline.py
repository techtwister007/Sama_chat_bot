#!/usr/bin/env python3
"""One-command pipeline: data -> Kaggle dataset -> train -> benchmark -> serve.

Needs on your machine:  pip install kaggle requests
Env vars: KAGGLE_USERNAME, KAGGLE_KEY (kaggle.com -> Settings -> API -> Create New Token),
          JUDGE_API_KEY (OpenRouter), optional API_KEY (password for your model endpoint).
Usage: python automate/pipeline.py [--stage data|train|bench|serve|all] [--skip-data]
"""
import argparse, json, os, secrets, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "automate" / "_build"          # gitignored; contains secrets.json
USER = os.environ["KAGGLE_USERNAME"]
SLUGS = {"data": "sama-data", "train": "sama-train", "bench": "sama-bench", "serve": "sama-serve"}


def run(cmd, **kw):
    print("+", cmd)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def make_data():
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "train.jsonl").exists():
        run(f"{sys.executable} {ROOT/'kaggle/make_dataset.py'} {ROOT/'kaggle/topics.txt'}", cwd=WORK)
    kf = WORK / "api_key.txt"
    api_key = os.environ.get("API_KEY") or (kf.read_text() if kf.exists() else secrets.token_urlsafe(16))
    (WORK / "api_key.txt").write_text(api_key)
    (WORK / "secrets.json").write_text(json.dumps(
        {"JUDGE_API_KEY": os.environ["JUDGE_API_KEY"], "API_KEY": api_key}))
    meta = {"title": "sama data", "id": f"{USER}/{SLUGS['data']}", "licenses": [{"name": "CC0-1.0"}]}
    (WORK / "dataset-metadata.json").write_text(json.dumps(meta))
    # private by default; contains your keys, so keep it private
    exists = subprocess.run(f"kaggle datasets status {USER}/{SLUGS['data']}", shell=True,
                            capture_output=True).returncode == 0
    if exists:
        run("kaggle datasets version -p . -m update", cwd=WORK)
    else:
        run("kaggle datasets create -p .", cwd=WORK)
    wait(lambda: subprocess.run(f"kaggle datasets status {USER}/{SLUGS['data']}", shell=True,
                                capture_output=True, text=True).stdout.strip().lower() == "ready")


def push_kernel(stage, script, sources_kernels=(), gpu=True):
    d = WORK / f"k_{stage}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(ROOT / "kaggle" / script, d / script)
    meta = {"id": f"{USER}/{SLUGS[stage]}", "title": f"sama {stage}", "code_file": script,
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": gpu, "enable_internet": True,
            "dataset_sources": [f"{USER}/{SLUGS['data']}"],
            "kernel_sources": [f"{USER}/{SLUGS[k]}" for k in sources_kernels]}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    run("kaggle kernels push -p .", cwd=d)


def kernel_status(stage):
    out = subprocess.run(f"kaggle kernels status {USER}/{SLUGS[stage]}", shell=True,
                         capture_output=True, text=True).stdout.lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out:
            return s
    return "unknown"


def wait(cond, every=30, timeout=3600):
    t = time.time()
    while not cond():
        if time.time() - t > timeout:
            sys.exit("timed out waiting")
        time.sleep(every)


def wait_kernel(stage, timeout=12 * 3600):
    t = time.time()
    while True:
        s = kernel_status(stage)
        print(f"[{stage}] {s}")
        if s == "complete":
            return
        if s in ("error", "cancel"):
            run(f"kaggle kernels output {USER}/{SLUGS[stage]} -p {WORK/('out_'+stage)}")
            sys.exit(f"{stage} failed; see logs in {WORK/('out_'+stage)} and paste the error to Claude")
        if time.time() - t > timeout:
            sys.exit("timed out")
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["data", "train", "bench", "serve", "all"])
    a = ap.parse_args()
    st = a.stage
    if st in ("data", "all"):
        make_data()
    if st in ("train", "all"):
        push_kernel("train", "train_qlora.py")
        wait_kernel("train")
    if st in ("bench", "all"):
        push_kernel("bench", "benchmark_merge.py", sources_kernels=["train"])
        wait_kernel("bench")
        run(f"kaggle kernels output {USER}/{SLUGS['bench']} -p {WORK/'out_bench'}")
        print("Benchmark log saved in", WORK / "out_bench")
    if st in ("serve", "all"):
        push_kernel("serve", "serve_model.py", sources_kernels=["train"])
        key = (WORK / "api_key.txt").read_text()
        print(f"\nServe notebook started: https://www.kaggle.com/code/{USER}/{SLUGS['serve']}"
              "\nOpen it, view the log, and copy the line 'MODEL_API_URL = https://...'."
              f"\nThen run the GUI:  MODEL_API_KEY={key} chainlit run ui.py   and type  /model <url>")


if __name__ == "__main__":
    main()
