# AGENTS.md — working guide for AI agents & contributors

Read this first. It's the map of the project, the conventions, and what to do next.

## What this project is

AgentFlow-Pro is a **trainable multi-agent reasoning framework** — a clean rebuild of the AgentFlow
paper (ICLR 2026). A `Solver` runs a loop: a **Planner** (the only trainable module) picks the next
action, an **Executor** runs it (think / web search / sandboxed Python / answer), the result is
appended to **Memory**, and a **Verifier** decides whether there's enough to answer or the loop
should continue. The research contribution is to train the Planner with **DAPO** + a **learned
Process Reward Model** instead of the paper's outcome-only Flow-GRPO. The full training pipeline
(`train/`) is built and committed; only the GPU run itself (Phase 4) is outstanding.

```
              ┌───────────┐  next action   ┌───────────┐  result   ┌───────────┐  sufficient?
   query ───► │  Planner  │ ─────────────► │ Executor  │ ────────► │ Verifier  │ ──► answer ✔
              │(trainable)│                │  (tools)  │           │  (judge)  │ ──► loop ↺
              └─────▲─────┘                └───────────┘           └─────┬─────┘
                    │                                                   │
                    └───────────────  Memory (running state)  ◄──────────┘
                          loop runs up to `max_steps`, else fallback answer
```

## Repo layout — what every file does

```
core/
  llm.py        OllamaClient — hits Ollama's NATIVE /api/chat endpoint (httpx), NOT the OpenAI-compat
                /v1 endpoint. /v1 silently ignores `think: false`; native honors it (the 53x fix).
                `complete(messages, temperature, max_tokens, response_format)` returns text.
  types.py      Pydantic v2 models: Action(Enum: think|search|code|answer), PlannerOutput,
                ExecutorOutput, VerifierOutput, MemoryEntry, SolverResult. ALL data models live here.
  memory.py     Memory: add(step, plan, result) · to_context() -> str (LLM-readable history) ·
                entries (property) · clear(). In-task only; Qdrant backend is Phase 5.
  planner.py    Planner.plan(query, memory_context) -> PlannerOutput. OllamaClient call with a Pydantic
                JSON Schema as `format`, retry-once-then-degrade-to-THINK. Exports _strip_markdown /
                _strip_think_tags / _RETRY_MSG (verifier reuses them). `think` flag passed to OllamaClient.
  executor.py   Executor.execute(plan) -> ExecutorOutput. Routing: THINK/ANSWER echo the input;
                SEARCH → tools.builtin.search.web_search; CODE → tools.builtin.python_exec.run_python.
  verifier.py   Verifier.verify(query, memory_context, last_result) -> VerifierOutput. LLM call,
                strict (defaults to sufficient=False on parse failure so the loop keeps going).
  solver.py     Solver — the loop. _DEFAULT_MODEL = "qwen3:8b", base_url = "http://localhost:11434"
                (native endpoint). Early-exits on Planner ANSWER or Verifier sufficient.
  __init__.py   re-exports Solver + the public types.

tools/
  mcp_server.py        FastMCP("agentflow-tools") exposing search() and python_exec() tools.
                       Run with `uv run python -m tools.mcp_server`. (Nit: search() docstring still
                       says "DuckDuckGo" — it actually uses Tavily.)
  builtin/search.py    web_search(query, max_results=5) via TavilyClient + TAVILY_API_KEY.
  builtin/python_exec.py  run_python(code) — exec() with a stripped __builtins__ (blocks exec/eval/
                       compile/open/input/breakpoint/__import__) and a stdlib import whitelist.
                       Best-effort, NOT a real security boundary; never run on untrusted infra.

eval/
  datasets.py   Task(id, question, gold, kind: "math"|"mc"). load_aime24() (open, 30 problems),
                load_gpqa_diamond() (gated — needs HF_TOKEN). _patch_datasets_for_py314() works
                around a dill/datasets incompatibility.
  scorer.py     extract_final_answer() (\boxed{} → "Answer: X" → last int), score_math() (math-verify
                with int fallback), score_mc(), score(task, pred).
  runner.py     EvalResult / EvalReport models · run_eval(solver, tasks, benchmark, limit) — runs,
                scores, prints a Rich table, saves runs/eval_<benchmark>_<timestamp>.json.
  run.py        Typer CLI: `uv run python -m eval.run -b aime24|gpqa [-m model] [-l N] [-s steps] [-t temp]`.

main.py         Typer CLI: `uv run python main.py "<query>" [-m model] [-s max_steps] [--think]`.
pyproject.toml  deps: httpx openai pydantic rich typer python-dotenv tavily-python sympy numpy.
                extras: tools=[fastmcp] · eval=[datasets, math-verify] · memory=[qdrant-client,
                sentence-transformers] · rl=[transformers<5, trl, peft, torch, accelerate, datasets]
                · dev=[pytest, pytest-asyncio].
train/          DAPO + PRM training (BUILT + Phase 4 run complete). data.py (shared plumbing +
                build_prm_input single-source-of-truth) · judge.py (DeepSeek LLM-judge labeling) ·
                prm.py (Qwen3-0.6B regression PRM) · reward.py (PRM→TRL reward) · dynamic_sampling.py
                (the DAPO piece TRL lacks) · dapo.py (Qwen3-8B bf16 + PEFT LoRA + TRL GRPOTrainer).
runs/           eval reports (gitignored).
docs/           architecture.md (loop + tools + harness internals) · research.md (DAPO + PRM design) ·
                phase4-runpod-guide.md (the GPU run, step by step).
```

## Conventions

- **Python 3.11**. **`uv` only** — `uv sync`, `uv sync --extra eval`, `uv run python ...`. Never call
  bare `pip` or `python`.
- **No abstraction layers for the LLM.** Go through `core/llm.py`'s `OllamaClient`, which hits
  Ollama's **native `/api/chat`** endpoint via `httpx`. Do **not** use the OpenAI-compat `/v1`
  endpoint — it silently ignores `think: false` (Qwen3 keeps reasoning, burns the budget, returns
  empty content). No LiteLLM or any router. (The DeepSeek judge in `train/judge.py` is the one place
  that uses the `openai` SDK, pointed at the DeepSeek API.)
- **All data models are Pydantic v2 in `core/types.py`.** Don't scatter dataclasses/dicts.
- **Structured LLM output**: pass a Pydantic `model_json_schema()` to `OllamaClient`'s `response_format`
  (Ollama's `format` field — grammar-constrained), then parse with retry-once-then-degrade (copy the
  pattern in `Planner.plan` — try `json.loads`, on failure append `_RETRY_MSG` and retry once, then
  fall back to a safe default rather than crashing).
- **Qwen3 thinking**: pass `think=<bool>` to `OllamaClient` (the native endpoint honors it). Default
  **off** (it was burning >1 min per call). The `--think` CLI flag and the `think=` constructor args
  turn it back on when needed.
- **Match the surrounding code** — comment density, naming, the `match`/`case` style in `executor.py`.
- **macOS dev box**: there is **no `timeout` binary**. To bound a command, run it as a background
  process and poll, don't wrap it in `timeout`/`gtimeout`.

## How to run / test

```bash
uv sync                                            # base deps
uv run python main.py "What is 15% of 240, then doubled?"

uv sync --extra eval
uv run python -m eval.run -b aime24 --limit 5 --max-steps 8

uv run python -m tools.mcp_server                  # FastMCP server (needs --extra tools)
uv run pytest                                      # once tests exist (none yet)
```

Ollama must be running locally with `qwen3:8b` pulled. **You (the user) manage Ollama and model
installs** — code should never try to pull models or assume CUDA. Use `qwen3:14b` only if local
memory allows; **RL training needs a rented ~24 GB GPU** (e.g. RunPod RTX 4090).

## Status & what's next

Phases 0–2 (scaffold · core loop · real tools), the eval harness, and the **full DAPO + PRM training
pipeline** (`train/`) are **built and committed**, and the **Phase 4 GPU run is complete**. Both
baselines and trained-model numbers are recorded: **AIME24 33.3%→30.0%** (flat within noise, n=30) and
**GPQA 40.0%→45.0% (+5 pts, cross-domain)**. The trained Planner was served via Ollama at the same
Q4_K_M quant as the baseline; reports + analysis in **[results/](results/README.md)** and
**[docs/research.md](docs/research.md)**.

**Definition of done — met**: baseline eval numbers → trained the Planner (DAPO + PRM) → re-evaled →
base→trained delta reported in `docs/research.md`, `results/README.md`, and the README table. Optional
next: Phase 5 (Qdrant memory), more training (more steps / stronger PRM / vLLM) for larger gains.
