# AgentFlow-Pro

A **trainable, four-module agentic-reasoning framework** — a clean rebuild of
[AgentFlow](https://arxiv.org/abs/2510.05592) (ICLR 2026) on a modern stack, with an RL upgrade:
the planner is optimised with **DAPO** + a **Process Reward Model (PRM)** instead of the paper's
outcome-only Flow-GRPO.

```
              ┌───────────┐  next action   ┌───────────┐  result   ┌───────────┐  sufficient?
   query ───► │  Planner  │ ─────────────► │ Executor  │ ────────► │ Verifier  │ ──► answer ✔
              │(trainable)│                │  (tools)  │           │  (judge)  │ ──► loop ↺
              └─────▲─────┘                └───────────┘           └─────┬─────┘
                    │                                                   │
                    └───────────────  Memory (running state)  ◄──────────┘
```

## What's different from the paper

| | AgentFlow (paper) | AgentFlow-Pro |
|---|---|---|
| Backbone | Qwen2.5-7B | Qwen3-8B (local baseline; Qwen3-14B optional if memory allows) |
| RL algorithm | Flow-GRPO (outcome reward) | **DAPO** (decoupled clip + dynamic sampling) |
| Credit assignment | trajectory-level | **+ step-level PRM** (was *this* action a good move?) |
| Tool layer | bespoke | **MCP / FastMCP** server + sandboxed Python exec |
| LLM access | — | official `openai` SDK → Ollama `/v1` (no abstraction layers) |
| Memory | in-task | in-task now; Qdrant cross-episode planned (Phase 5) |

See **[docs/research.md](docs/research.md)** for the DAPO + PRM design and experimental protocol,
**[docs/architecture.md](docs/architecture.md)** for how the loop and tools work, and
**[ROADMAP.md](ROADMAP.md)** for the phased plan and status.

## Quickstart

Requires Python 3.11, [`uv`](https://docs.astral.sh/uv/), and a running [Ollama](https://ollama.com).

```bash
# 1. install deps
uv sync

# 2. pull the recommended local eval model — you manage Ollama/model installs yourself
ollama pull qwen3:8b

# 3. solve something
uv run python main.py "What is 15% of 240, then doubled?"
uv run python main.py "Explain how transformers work" --max-steps 3
uv run python main.py "..." --think          # enable Qwen3 reasoning tokens (slower, default off)
```

Web search uses [Tavily](https://tavily.com) — put `TAVILY_API_KEY=...` in `.env` (copy from
`.env.example`). The free tier is 1000 queries/month.

## Evaluation

```bash
uv sync --extra eval
uv run python -m eval.run -b aime24 --limit 5 --max-steps 8   # small subset first
uv run python -m eval.run -b aime24                            # full AIME24 (30 problems)
uv run python -m eval.run -b gpqa --limit 5 --max-steps 4      # GPQA Diamond subset
```

Reports are written to `runs/eval_<benchmark>_<timestamp>.json` with per-problem trajectories and an
overall accuracy, average steps, and per-task elapsed time. GPQA Diamond (`-b gpqa`) is gated on
Hugging Face — request access on the dataset page, then set `HF_TOKEN` in `.env`.

Baseline numbers (untrained `qwen3:8b`) and the post-training delta live in
[docs/research.md](docs/research.md).

## Repo map

| Path | What |
|---|---|
| `core/` | the inference engine — `types` (Pydantic models), `memory`, `planner`, `executor`, `verifier`, `solver` |
| `tools/` | `mcp_server.py` (FastMCP), `builtin/search.py` (Tavily), `builtin/python_exec.py` (sandboxed exec) |
| `eval/` | `datasets.py` (AIME24, GPQA loaders), `scorer.py` (math-verify), `runner.py`, `run.py` (CLI) |
| `main.py` | `solve` CLI |
| `rl/`, `train/` | DAPO + PRM training — **Phase 4, not yet built** |
| `runs/` | eval reports (gitignored) |

## Status

Phases 0–2 (scaffold · core loop · real tools) and the eval harness are **done**. Next: record the
AIME24 + GPQA Diamond baselines, then build the RL training (`rl/` + `train/`). Track it in
[ROADMAP.md](ROADMAP.md).

## License

Personal portfolio / research project. Built on the ideas of the AgentFlow paper; not affiliated with
its authors.
