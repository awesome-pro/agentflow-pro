# AGENTS.md — working guide for AI agents & contributors

Read this first. It's the map of the project, the conventions, and what to do next.

## What this project is

AgentFlow-Pro is a **trainable multi-agent reasoning framework** — a clean rebuild of the AgentFlow
paper (ICLR 2026). A `Solver` runs a loop: a **Planner** (the only trainable module) picks the next
action, an **Executor** runs it (think / web search / sandboxed Python / answer), the result is
appended to **Memory**, and a **Verifier** decides whether there's enough to answer or the loop
should continue. The research contribution (Phase 4, not yet built) is to train the Planner with
**DAPO** + a **Process Reward Model** instead of the paper's outcome-only Flow-GRPO.

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
  types.py      Pydantic v2 models: Action(Enum: think|search|code|answer), PlannerOutput,
                ExecutorOutput, VerifierOutput, MemoryEntry, SolverResult. ALL data models live here.
  memory.py     Memory: add(step, plan, result) · to_context() -> str (LLM-readable history) ·
                entries (property) · clear(). In-task only; Qdrant backend is Phase 5.
  planner.py    Planner.plan(query, memory_context) -> PlannerOutput. LLM call in JSON mode,
                retry-once-then-degrade-to-THINK. Exports _strip_markdown / _strip_think_tags /
                _RETRY_MSG (verifier reuses them). `think` flag → extra_body={"think": ...}.
  executor.py   Executor.execute(plan) -> ExecutorOutput. Routing: THINK/ANSWER echo the input;
                SEARCH → tools.builtin.search.web_search; CODE → tools.builtin.python_exec.run_python.
  verifier.py   Verifier.verify(query, memory_context, last_result) -> VerifierOutput. LLM call,
                strict (defaults to sufficient=False on parse failure so the loop keeps going).
  solver.py     Solver — the loop. _DEFAULT_MODEL = "qwen3.5:4b", _OLLAMA_BASE_URL =
                "http://localhost:11434/v1". Early-exits on Planner ANSWER or Verifier sufficient.
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
pyproject.toml  deps: openai pydantic rich typer python-dotenv tavily-python.
                extras: tools=[fastmcp] · eval=[datasets, math-verify] · memory=[qdrant-client,
                sentence-transformers] · rl=[trl, transformers, torch, accelerate, datasets, unsloth]
                · dev=[pytest, pytest-asyncio].
rl/, train/     DAPO + PRM training. NOT BUILT YET — Phase 4. See docs/research.md and ROADMAP.md.
runs/           eval reports (gitignored).
docs/           architecture.md (loop + tools + harness internals) · research.md (DAPO + PRM design).
```

## Conventions

- **Python 3.11**. **`uv` only** — `uv sync`, `uv sync --extra eval`, `uv run python ...`. Never call
  bare `pip` or `python`.
- **No abstraction layers for the LLM.** Use the official `openai` SDK pointed at Ollama's
  OpenAI-compatible endpoint: `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`.
  Explicitly **not** LiteLLM or any router.
- **All data models are Pydantic v2 in `core/types.py`.** Don't scatter dataclasses/dicts.
- **Structured LLM output**: `response_format={"type": "json_object"}`, then parse with
  retry-once-then-degrade (copy the pattern in `Planner.plan` — try `json.loads`, on failure append
  `_RETRY_MSG` and retry once, then fall back to a safe default rather than crashing).
- **Qwen3.5 "thinking"**: pass `extra_body={"think": <bool>}` through the OpenAI SDK. Default **off**
  (it was burning >1 min per call). The `--think` CLI flag and the `think=` constructor args turn it
  back on when needed.
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

Ollama must be running locally with `qwen3.5:4b` pulled. **You (the user) manage Ollama and model
installs** — code should never try to pull models or assume CUDA. Default dev model fits 8 GB Apple
Silicon; **RL training needs a rented ~24 GB GPU** (e.g. RunPod RTX 4090).

## Status & what's next

Phases 0–2 (scaffold · core loop · real tools) + the eval harness are **done**. The immediate next
step is recording the **AIME24 baseline** for untrained `qwen3.5:4b`, then **Phase 4**: build `rl/`
(`rewards.py`, `prm.py`, `dapo.py`, `trainer.py`, `dataset.py`) and `train/` (`config.yaml`,
`run.py`). Full plan and acceptance criteria: **[ROADMAP.md](ROADMAP.md)**.

**Definition of done for the project**: baseline eval numbers → train the Planner (DAPO + PRM) →
re-eval → report the base→trained delta in `docs/research.md` and the README table.
