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
| 3 | Eval harness + **baseline numbers** | 🔵 (harness done, baseline not yet recorded) |
| 4 | **DAPO + PRM — RL training** | ⏳ |
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
- **Goal**: measure untrained `qwen3.5:4b` on real benchmarks so Phase 4 has something to beat.
- **Files**: `eval/datasets.py`, `eval/scorer.py`, `eval/runner.py`, `eval/run.py` *(all built)*;
  output to `runs/eval_<benchmark>_<timestamp>.json`.
- **Commands**:
  ```bash
  uv sync --extra eval
  uv run python -m eval.run -b aime24 --limit 5 --max-steps 8   # gauge speed first
  uv run python -m eval.run -b aime24                            # full 30
  # uv run python -m eval.run -b gpqa                            # needs HF_TOKEN in .env
  ```
- **Done when**: `runs/eval_aime24_*.json` exists and the baseline accuracy (+ avg steps) is recorded
  in `docs/research.md` and the README table. GPQA Diamond deferred until `HF_TOKEN` is set.

## Phase 4 — DAPO + PRM (RL training) ⏳ **next**
- **Goal**: LoRA-train the Planner with **DAPO** (decoupled clip + dynamic sampling) and a
  **Process Reward Model** (step-level credit). Design: [docs/research.md](docs/research.md).
- **Files to create**:
  - `rl/__init__.py`
  - `rl/rewards.py` — `outcome_reward(task, answer)`; heuristic step rewards (tool-error penalty,
    repeated-action penalty, progress bonus).
  - `rl/prm.py` — `ProcessRewardModel.score(state, action)`; v1 rule/verifier-derived, v2 a small
    learned head.
  - `rl/dapo.py` — DAPO advantage computation + the four tricks (clip-higher, dynamic sampling,
    token-level PG loss, overlong-reward shaping), layered on TRL's GRPO/PPO trainer.
  - `rl/trainer.py` — training loop: roll out trajectories with `Solver` as the environment, collect
    `(state, action, reward)`, update the Planner LoRA.
  - `rl/dataset.py` — training task mix (math + agentic), reusing `eval/datasets.py` loaders.
  - `train/config.yaml` — model, LoRA rank, lr, batch, DAPO hyperparams, dataset paths.
  - `train/run.py` — `uv run python -m train.run --config train/config.yaml` (Typer CLI).
- **Environment**: needs a rented ~24 GB GPU (RunPod RTX 4090, est. $5–15 total). `uv sync --extra rl`
  on that box only; code must not assume local CUDA.
- **Done when**: a training run completes on the GPU, a LoRA checkpoint is saved, and the loss/reward
  curves look sane (logged to W&B or stdout).

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
