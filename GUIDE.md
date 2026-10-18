# Sama build guide (beginner, step by step)

Roles: **you are the architect** (decide goals, hold accounts and keys). **Claude is the engineer**
(writes and fixes all code in this repo). Anything marked 👤 only you can do. Anything marked 🤖 you can
ask Claude to do or fix; paste any error message into the chat and Claude will fix the code.

## Phase 0: accounts (👤, ~20 min)
1. Kaggle: sign in at kaggle.com with shanukumarofficialac@gmail.com. Verify your phone
   (Settings -> Phone verification). This unlocks GPUs and Internet in notebooks.
2. Teacher API key: create an account at openrouter.ai, add a few dollars of credit (~$3-5 is enough
   for ~3,000 Q&A), and create an API key. Keep it secret; never paste it into GitHub.
3. Optional: GitHub personal access token (only for private repos you want to chat with).

## Phase 1: make the training data (👤 run, 🤖 fixes)
1. On your computer, install Python 3.10+ and run: `git clone` this repo, `cd Sama_chat_bot`.
2. `pip install requests`
3. Set your key and run (Mac/Linux): `JUDGE_API_KEY=your_key python kaggle/make_dataset.py kaggle/topics.txt`
   (Windows PowerShell: `$env:JUDGE_API_KEY="your_key"; python kaggle/make_dataset.py kaggle/topics.txt`)
4. You get `train.jsonl` (~3,000 examples) and `bench.jsonl` (60 held-out test questions).
   Open both and read a few lines to sanity-check quality.
5. Kaggle -> Datasets -> New Dataset -> upload both files, name it `sama-data`.

## Phase 2: train (👤, ~2-4 hours of Kaggle time)
1. Kaggle -> Code -> New Notebook. Right side: Accelerator = **GPU T4 x2**, Internet = On.
2. Add Data -> select your `sama-data` dataset.
3. Copy `kaggle/train_qlora.py` into cells (each `# %%` line starts a new cell; run the `!pip` line
   by deleting the leading `# `).
4. Run all. When finished, `/kaggle/working/sama-lora` exists. Save Version -> Save & Run All so it
   persists, then in the notebook Output tab create a dataset from `sama-lora`.
   Free quota is ~30 GPU hours/week, with a 12-hour limit per session.
5. If it runs out of memory: tell Claude, which will lower `max_length` or switch to a 7-8B model.

## Phase 3: test quality (👤 run)
1. New notebook (GPU T4 x2, Internet On); add `sama-data` and the adapter dataset.
2. Add Kaggle secret `JUDGE_API_KEY` (Add-ons -> Secrets).
3. Run `kaggle/benchmark_merge.py`. Read the base vs fine-tuned scores. If the fine-tune is not
   better, tell Claude; the data or hyperparameters get adjusted.
4. The merge step is optional (the GUI works with the adapter alone).

## Phase 4: serve the model (👤)
1. New notebook, GPU T4 x2, Internet On; add the adapter dataset. Add secret `API_KEY` (invent a password).
2. Run `kaggle/serve_model.py`. After a few minutes it prints `MODEL_API_URL = https://...trycloudflare.com`.
   Keep the notebook running. The URL changes each session.

## Phase 5: the GUI on your computer (👤)
1. `pip install -r requirements-gui.txt`
2. Mac/Linux: `MODEL_API_URL=<url> MODEL_API_KEY=<API_KEY> chainlit run ui.py`
   (PowerShell: set `$env:MODEL_API_URL` and `$env:MODEL_API_KEY` first.)
3. Browser opens. Type `upload` for local files, `/github owner/repo` for a repo, then ask questions.

## When something breaks
Copy the full error text into the chat. Claude (engineer) edits the code, you re-run the failing phase.
Fix order of likelihood: out-of-memory -> tunnel URL changed -> missing secret -> package version.

## Automated route (replaces phases 1-4)
1. Kaggle -> Settings -> API -> Create New Token (gives username and key). Verify your phone first.
2. `pip install kaggle requests`
3. Set `KAGGLE_USERNAME`, `KAGGLE_KEY`, `JUDGE_API_KEY` (and optionally `API_KEY`) in your terminal.
4. `python automate/pipeline.py` runs: generate data -> upload private dataset -> train -> benchmark -> serve.
   Run one step at a time with `--stage data|train|bench|serve`.
5. Open the printed serve-notebook link, copy `MODEL_API_URL`, then `chainlit run ui.py` and type `/model <url>`.
Notes: the private `sama-data` dataset holds your keys in `secrets.json`; keep it private.
Pick the GPU (T4 x2) in the notebook's Settings -> Accelerator if Kaggle defaults to a single GPU.

## Free (no API cost) options
- **Fully free:** `python automate/pipeline.py --free`. Kaggle's GPU (Qwen3-14B) writes the questions and answers
  and the base model judges the benchmark. No `JUDGE_API_KEY` needed. Costs GPU hours only (~30 free/week),
  and the data is only as good as the 14B generator, so quality is below a big teacher.
- **Free teacher API:** get a free key from Groq (console.groq.com) or Google AI Studio, then
  `PROVIDER=groq JUDGE_API_KEY=<key> python automate/pipeline.py` (or `PROVIDER=gemini`,
  `PROVIDER=openrouter_free`). Free tiers rate-limit, so data generation is slow but automatic retries are built in.
  Model names change; override with `JUDGE_MODEL`.
