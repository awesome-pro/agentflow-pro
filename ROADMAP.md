# ROADMAP

The build plan for AgentFlow-Pro, in phases. Each phase has a single clear goal, an acceptance
criterion that says when it's "done", and the files it touches. The north star: **establish a
baseline → train the Planner with DAPO + PRM → re-evaluate → report the delta.**

Legend: ✅ done · 🔵 partly done · ⏳ next · 🟡 stretch

| # | Phase | Status |
|---|---|---|
| 0 | Scaffold | ✅ |
| 1 | Core solver loop | ✅ |
| 2 | Real tools (MCP + Tavily + sandboxed exec) | ✅ |
| 3 | Eval harness + **baseline numbers** | 🔵 (harness done; AIME24 baseline = 33.3%, GPQA pending) |
| 4 | **DAPO + PRM — RL training** | 🔵 (full pipeline built & committed; GPU run pending) |
| 5 | Episodic memory (Qdrant) | 🟡 |
| 6 | Report & polish | ⏳ |

---

## Phase 0 — Scaffold ✅
- **Goal**: `uv` project, dependency set, package layout.
- **Files**: `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, package `__init__.py`s.
- **Done when**: `uv sync` is clean; `uv run python -c "import core"` works.

## Phase 1 — Core solver loop ✅
- **Goal**: the Planner → Executor → Verifier → Memory loop running end-to-end against Ollama.
- **Files**: `core/types.py`, `core/memory.py`, `core/planner.py`, `core/executor.py`,
  `core/verifier.py`, `core/solver.py`, `main.py`.
- **Done when**: `uv run python main.py "What is 15% of 240, then doubled?"` returns `72`.

## Phase 2 — Real tools ✅
- **Goal**: replace stub tools with real ones; expose them over MCP.
- **Files**: `tools/mcp_server.py` (FastMCP), `tools/builtin/search.py` (Tavily web search),
  `tools/builtin/python_exec.py` (sandboxed exec), `core/executor.py` wiring.
- **Done when**: a `search` action returns real web results; a `code` action runs Python and returns
  stdout; the sandbox rejects `open(...)` / `__import__(...)`.

## Phase 3 — Eval harness + baseline 🔵
- **Goal**: measure untrained `qwen3:8b` on AIME24 and GPQA Diamond so Phase 4 has credible
  baselines to beat.
- **Files**: `eval/datasets.py`, `eval/scorer.py`, `eval/runner.py`, `eval/run.py` *(all built)*;
  output to `runs/eval_<benchmark>_<timestamp>.json` with accuracy, average steps, and elapsed time.
- **Commands**:
  ```bash
  uv sync --extra eval
  uv run python -m eval.run -b aime24 --limit 5 --max-steps 8   # gauge speed first
  uv run python -m eval.run -b aime24                            # full 30
  uv run python -m eval.run -b gpqa --limit 5 --max-steps 4      # needs HF_TOKEN in .env
  uv run python -m eval.run -b gpqa --max-steps 4                # full Diamond if runtime is acceptable
  ```
- **Done when**: `runs/eval_aime24_*.json` and `runs/eval_gpqa_*.json` exist and baseline accuracy
  (+ avg steps) is recorded in `docs/research.md` and the README table. If full GPQA is too slow on
  local Ollama, record a clearly labelled GPQA Diamond subset first and keep the full run queued.
- **Status**: AIME24 baseline **done — 33.3% (10/30)**, `qwen3:8b`, `max_steps 6`, `temp 0`, verified
  (no false positives). GPQA Diamond baseline queued for the GPU box. AIME *training* split loader
  (`load_aime_train`, 918 problems, Year ≤ 2023, de-duplicated vs AIME24) also shipped here.

## Phase 4 — DAPO + PRM (RL training) 🔵 **code done; GPU run pending**
- **Goal**: LoRA-train the Planner with **DAPO** (decoupled clip + dynamic sampling) and a **learned
  Process Reward Model** (step-level credit). Design: [docs/research.md](docs/research.md).
- **What shipped** (the actual layout — `train/`, not the originally-sketched `rl/`):
  - `train/data.py` — shared plumbing: `Step` / `PRMExample`, `load_steps()` (pull Planner steps out
    of trajectory JSON), and `build_prm_input()` — the **single source of truth** for the text the PRM
    scores (shared by labeler, trainer, reward fn; deliberately excludes the tool result).
  - `train/judge.py` — LLM-judge labeling. Scores each step 0–1 with **DeepSeek** (`deepseek-chat`,
    stronger than the 8B policy); free Ollama judge available for smoke tests. Writes
    `artifacts/prm_labels.jsonl`.
  - `train/prm.py` — the **PRM**: `Qwen3-0.6B` + a regression head (`num_labels=1`), MSE-trained on
    the judge scores; `PRM.score(texts) -> [0,1]` inference wrapper.
  - `train/reward.py` — `make_prm_reward()`: a TRL-style reward fn; malformed/unknown-action ⇒ 0.0,
    else the PRM score. The PRM (not DeepSeek) is the live training signal.
  - `train/dynamic_sampling.py` — `curate_prompts()` / `has_signal()`: the **one DAPO component TRL
    does not implement** — drops zero-variance prompt groups before training.
  - `train/dapo.py` — the trainer: `Qwen3-8B` bf16 + a PEFT LoRA adapter + TRL `GRPOTrainer`
    (`loss_type="dapo"`, `epsilon=0.2`, `epsilon_high=0.28`, `mask_truncated_completions=True`,
    `beta=0.0`) + soft overlong punishment + the dynamic-sampling curation pass. No
    unsloth/bitsandbytes — a 48GB card fits the 8B in bf16, which keeps deps reproducible.
- **Environment**: a rented ~24–48 GB GPU (A40 recommended, est. $8–15 total). `uv sync --extra rl
  --extra eval` on that box only; code must not assume local CUDA. Full infra walkthrough:
  [docs/phase4-runpod-guide.md](docs/phase4-runpod-guide.md).
- **Done when**: the GPU run completes (collect → judge → train PRM → DAPO → merge/GGUF), a trained
  Planner checkpoint is saved, and the reward curve climbs sanely.

## Phase 5 — Episodic memory (Qdrant) 🟡 stretch
- **Goal**: cross-episode memory behind the existing `Memory` interface — retrieve hints from past
  solves and inject them into the planner's context.
- **Files**: `core/memory.py` (add a Qdrant-backed implementation behind the same API), embedding via
  `sentence-transformers`; `--memory` toggle on the CLIs. Deps: `uv sync --extra memory`.
- **Done when**: retrieved past-episode hints visibly appear in the planner prompt and can be toggled
  off.

## Phase 6 — Report & polish ⏳
- **Goal**: close the loop — quantify what the training bought.
- **Tasks**: re-run the eval harness pointed at the trained checkpoint; add a base→trained comparison
  table (accuracy, avg steps, tool-call efficiency) to the README and a Results section to
  `docs/research.md`; record a short demo; tidy `pyproject.toml` extras and pin versions.
- **Done when**: README shows the benchmark table; `docs/research.md` has the results + a short
  discussion.

---

## Backlog / nice-to-haves
- Unit tests (`tests/`) — scorer edge cases, planner JSON-degrade path, sandbox blocks. (`--extra dev`.)
- More benchmarks (MATH-500, GSM8K) in `eval/datasets.py`.
- Fix the `search()` docstring in `tools/mcp_server.py` ("DuckDuckGo" → "Tavily").
- `--base-url` already lets you point at any OpenAI-compatible server (vLLM, hosted) — document it.
