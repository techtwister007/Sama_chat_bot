"""Client for the model served from Kaggle (see kaggle/serve_model.py)."""
import os

import requests

MODEL_API_URL = os.environ.get("MODEL_API_URL", "")  # e.g. https://xxxx.trycloudflare.com
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "")


def set_url(url):
    global MODEL_API_URL
    MODEL_API_URL = url.strip()


def chat(messages, max_new_tokens=768, temperature=0.3) -> str:
    if not MODEL_API_URL:
        raise RuntimeError("Set MODEL_API_URL to the tunnel URL printed by the Kaggle serve notebook.")
    resp = requests.post(
        f"{MODEL_API_URL.rstrip('/')}/chat",
        json={"messages": messages, "max_new_tokens": max_new_tokens, "temperature": temperature},
        headers={"x-api-key": MODEL_API_KEY},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["answer"]
