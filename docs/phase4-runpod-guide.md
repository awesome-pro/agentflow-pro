# Phase 4 — Train & Evaluate on RunPod

Complete guide for renting a cloud GPU and running the whole Phase 4 pipeline:
GPQA baseline → collect trajectories → train the PRM → DAPO-train the Planner →
re-evaluate. One focused session, **~8–10 GPU-hours, ~$5–15 total**.

---

## 0. Before you start (local, 5 min)

- Repo is pushed to GitHub (done).
- Have three keys ready: `HF_TOKEN`, `TAVILY_API_KEY`, `DEEPSEEK_API_KEY`.
- You can kill the local Mac GPQA run — GPQA will run on the pod instead.

---

## 1. Pick & launch the pod

Sign up at **runpod.io**, add **~$20** credit (Billing → pay as you go).

**Deploy → Pods → Deploy a Pod:**

| Setting | Choice | Why |
|---------|--------|-----|
| GPU | **A40 (48GB)** | ~$0.40/hr — barely more than an RTX 4090, but 48GB removes all OOM risk when DAPO colocates vLLM for rollouts. The single best value here. |
| Cloud | **Community Cloud** | ~40% cheaper than Secure; fine for this. |
| Pricing | **On-Demand** (not Spot) | Spot is ~50% cheaper but a 5-sec kill window will wreck a 4-hour run. ~$1 of insurance. |
| Template | `runpod/pytorch` — **CUDA 12.4–12.8, Python 3.11** | torch/transformers wheels target this. |
| Container disk | **40 GB** | OS + pip installs (ephemeral). |
| Volume / Network volume | **100 GB**, mounted at `/workspace` | Holds models, datasets, checkpoints. A **Network Volume** survives pod termination — create one if you'll spin pods up/down (~$7/mo parked). |
| Exposed ports | **22** (SSH), **8888** (Jupyter) | |

Cheaper alternative: **RTX 3090 (24GB) ~$0.22/hr** works but 24GB is tight with colocated vLLM — expect to babysit memory. A40 is worth the extra ~$0.20/hr.

---

## 2. Connect & set up the environment (~15–20 min)

Connect via the pod's **web terminal** (or SSH). Then:

```bash
# --- always work inside tmux so a disconnect doesn't kill a multi-hour run ---
tmux new -s af          # reattach later with:  tmux attach -t af

cd /workspace

# --- clone the repo (use a token if it is private) ---
git clone https://github.com/awesome-pro/agentflow-pro.git
cd agentflow-pro

# --- uv package manager ---
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# --- keep big downloads on the persistent volume ---
export HF_HOME=/workspace/hf
export OLLAMA_MODELS=/workspace/ollama

# --- Python deps (3.11) ---
# The rl extra is pinned to a validated set (transformers<5, trl, peft, torch,
# accelerate, datasets) — NO unsloth/bitsandbytes, so this resolves clean. The
# policy trains as bf16 + a LoRA adapter, which a 48GB card fits comfortably.
uv sync --extra rl --extra eval --python 3.11

# --- qwen3:8b — needed in two formats (same model, two channels): ---
#   GGUF via Ollama  -> agent's Planner inference (eval, collection, re-eval)
#   PyTorch via HF   -> training stack (transformers loads Qwen/Qwen3-8B)

# (a) GGUF side — Ollama
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /workspace/ollama.log 2>&1 &
ollama pull qwen3:8b
ollama ps          # confirm it loads on the GPU

# (b) HF side — pre-fetch so the training step doesn't sit downloading later
uv run python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-8B'); \
    snapshot_download('Qwen/Qwen3-0.6B')"   # 0.6B = the PRM backbone

# --- API keys ---
cat > .env <<'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
EOF
```

Tip: `export HF_HOME` / `export OLLAMA_MODELS` in `~/.bashrc` so they survive new shells.

---

## 3. GPQA baseline (~1–1.5 h)

```bash
uv run python -m eval.run --benchmark gpqa --max-steps 6
```
→ `runs/eval_gpqa_*.json` + the GPQA accuracy. This is the second baseline number
(alongside AIME24 = 33.3%).

## 4. Collect training trajectories (~1 h)

```bash
uv run python -m eval.run --benchmark aime_train --limit 150 --max-steps 6
```
→ `runs/eval_aime_train_*.json` — the trajectory corpus for the PRM and DAPO.
(Trains on AIME 1983–2023, never on the AIME24/GPQA test sets.)

## 5. Judge the steps → PRM labels (~10–20 min, DeepSeek API)

```bash
uv run python -m train.judge --runs "runs/eval_aime_train_*.json" --provider deepseek
```
→ `artifacts/prm_labels.jsonl`. DeepSeek API cost: well under $1.

## 6. Train the PRM (~20–40 min)

```bash
uv run python -m train.prm train --labels artifacts/prm_labels.jsonl
uv run python -m train.prm score   # sanity check: predicted vs judge label
```
→ `artifacts/prm` — the trained Process Reward Model.

## 7. DAPO-train the Planner (~3–5 h) — the headline run

```bash
uv run python -m train.dapo --prm artifacts/prm --runs "runs/eval_aime_train_*.json"
```
→ `artifacts/planner-dapo` (the LoRA adapter) **and** `artifacts/planner-dapo-merged`
(the LoRA merged into the base, bf16 — ready to convert). Watch the reward curve
climb in the logs. If you hit OOM: lower `--num-generations` (e.g. 4), or
`--lora-rank 16`.

## 8. Export to GGUF + load into Ollama

The merged bf16 model must become a GGUF so Ollama can serve it for the re-eval.
We use llama.cpp's converter (no unsloth):

```bash
# one-time: get llama.cpp's conversion script
git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
uv pip install -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# convert the merged model -> GGUF (q4_k_m), then register it with Ollama
uv run python /workspace/llama.cpp/convert_hf_to_gguf.py artifacts/planner-dapo-merged \
    --outfile artifacts/planner.gguf --outtype q4_k_m

# IMPORTANT: inherit qwen3:8b's chat template + stop params, only swap the weights.
# A bare `FROM planner.gguf` would drop the Qwen3 template and the agent would
# mis-parse every Planner JSON at re-eval time.
ollama show qwen3:8b --modelfile | grep -vE '^FROM ' > Modelfile.tmpl
printf 'FROM ./artifacts/planner.gguf\n' | cat - Modelfile.tmpl > Modelfile
ollama create agentflow-planner -f Modelfile
ollama run agentflow-planner "hi" --verbose   # smoke-test it loads + responds
```

## 9. Re-evaluate — the "after" numbers

```bash
uv run python -m eval.run --benchmark aime24 --model agentflow-planner --max-steps 6
uv run python -m eval.run --benchmark gpqa  --model agentflow-planner --max-steps 6
```
Compare to baseline: **AIME24 33.3% → ?** That delta is the portfolio result.

## 10. Save results & shut down

```bash
# copy the run JSONs somewhere safe before terminating
# from your Mac:  runpodctl receive ...   OR  scp over the exposed port 22
```
Then **Stop** the pod (halts GPU billing; volume kept) or **Terminate** it
(keep only the Network Volume). Per-second billing — stopping promptly matters.

---

## Cost summary

| Item | Estimate |
|------|----------|
| A40 GPU, ~9 h on-demand | ~$3.60 |
| DeepSeek judge API | <$1 |
| 100 GB network volume | ~$7/mo (prorated — a few $ for a few days) |
| **Total for a full run** | **~$8–15** |

Even a few repeat experiments stay under $30.

## Troubleshooting

- **OOM during DAPO** → lower `--num-generations` (e.g. 4), `--lora-rank 16`, or
  shorten `max_completion_length` in `train/dapo.py`. (bf16 8B + LoRA is ~18GB of
  weights; an A40's 48GB has wide margin, so OOM usually means too many rollouts.)
- **`transformers` resolves to 5.x and breaks `trl`/`peft`** → the `rl` extra pins
  `transformers<5`; if you installed packages by hand, force it:
  `uv pip install "transformers<5"`. (The 5.x line removed symbols `peft`/`trl` need.)
- **Ollama not on GPU** → `ollama ps` should show the GPU; restart `ollama serve`.
- **SSH/web terminal disconnects** → always run inside `tmux`; `tmux attach -t af`.
- **Re-download avoidance** → keep `HF_HOME` and `OLLAMA_MODELS` on `/workspace`
  (the volume); with a Network Volume they persist across pods.
