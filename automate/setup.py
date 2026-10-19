#!/usr/bin/env python3
"""One-time local setup: stores your Kaggle token outside the repo and checks it works.
Run:  python automate/setup.py   (it asks for the token; nothing is written inside this repo)"""
import getpass, os, subprocess, sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle", "requests"], check=True)
d = Path.home() / ".kaggle"
d.mkdir(exist_ok=True)
tok = os.environ.get("KAGGLE_API_TOKEN") or getpass.getpass("Paste your Kaggle API token (KGAT_...): ").strip()
f = d / "access_token"
f.write_text(tok)
f.chmod(0o600)
user = os.environ.get("KAGGLE_USERNAME") or input("Your Kaggle username (kaggle.com/<this part>): ").strip()
print(f"\nSaved token to {f}.\nNow set this in your terminal, then run the pipeline:")
print(f"  export KAGGLE_USERNAME={user}")
print("  python automate/pipeline.py --free        # zero API cost")
r = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
print("Auth check:", "OK" if r.returncode == 0 else "FAILED\n" + r.stderr[:300])
