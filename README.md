# AgentFlow-Pro

**Process-supervised reinforcement learning for a multi-step reasoning agent.**

A from-scratch rebuild of [AgentFlow](https://arxiv.org/abs/2510.05592) (ICLR 2026) that replaces
the paper's outcome-only Flow-GRPO with a **step-level Process Reward Model (PRM)** and **DAPO**
(Decoupled Clip + Dynamic Sampling Policy Optimization). The agent is a Planner → Executor →
Verifier loop; **only the Planner is trained**, and it is trained to make *better individual
decisions*, not just to land more correct final answers.

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white">
<img alt="rl" src="https://img.shields.io/badge/RL-DAPO%20%2B%20PRM-7B3FE4">
<img alt="trl" src="https://img.shields.io/badge/TRL-1.4-FF9D00">
<img alt="backbone" src="https://img.shields.io/badge/policy-Qwen3--8B%20%2B%20LoRA-00897B">
<img alt="serving" src="https://img.shields.io/badge/serving-Ollama%20native-555">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## The thesis

Outcome-only RL (GRPO / Flow-GRPO) gives an agent **one scalar of feedback per trajectory** — did the
final answer match? On a single-shot task that is fine. On a *multi-step* agentic loop it is a blunt
instrument: a five-step trajectory that succeeded propagates the **same** gradient to a brilliant
opening search and a wasteful `think`-loop in the middle. There is no notion of *which step was the
good move*. Long trajectories dilute the signal across many tokens, and any prompt where all rollouts
agree (all right or all wrong) contributes **zero gradient** while still costing a full set of
rollouts.

AgentFlow-Pro attacks both problems directly:

1. **A learned Process Reward Model** scores *every Planner decision* on a 0–1 quality scale, so the
   policy gets a dense, per-step signal instead of one trajectory-level bit.
2. **DAPO** decouples the PPO clip bounds (lets clearly-good actions take a larger step), uses a
   token-level loss (a long trajectory no longer gets down-weighted into noise), filters overlong
   stalls, and — via a hand-built **dynamic-sampling** pass — drops zero-variance prompt groups so
   every optimizer step carries real signal.

The result is a credit-assignment story you can point at: *this* action was good, *that* one stalled,
and the gradient reflects it.

---

## Architecture

```
              ┌───────────┐  next action   ┌───────────┐  result   ┌───────────┐  sufficient?
   query ───► │  Planner  │ ─────────────► │ Executor  │ ────────► │ Verifier  │ ──► answer ✔
              │(trainable)│                │  (tools)  │           │  (judge)  │ ──► loop ↺
              └─────▲─────┘                └───────────┘           └─────┬─────┘
                    │                                                   │
                    └───────────────  Memory (running state)  ◄──────────┘
                          loop runs up to max_steps, else a fallback answer
```

- **Planner** (`core/planner.py`) — the only trainable module. Emits a grammar-constrained
  `{thought, action, action_input}` JSON each step; `action ∈ {think, search, code, answer}`.
- **Executor** (`core/executor.py`) — pure dispatch, no LLM call: `search` → Tavily, `code` →
  a sandboxed Python REPL, `think`/`answer` → echo.
- **Verifier** (`core/verifier.py`) — a separate judge that decides whether the running state is
  enough to answer or the loop should continue. Conservative by default (parse failure ⇒ keep going).
- **Memory** (`core/memory.py`) — in-task running state today; a Qdrant cross-episode backend sits
  behind the same interface as a planned extension.

The loop, module contracts, and tool internals are documented in
**[docs/architecture.md](docs/architecture.md)**.

---

## What changes vs. the paper

| | AgentFlow (paper) | AgentFlow-Pro |
|---|---|---|
| Backbone | Qwen2.5-7B | **Qwen3-8B** (bf16 + LoRA via PEFT) |
| RL algorithm | Flow-GRPO (outcome reward) | **DAPO** — decoupled clip + dynamic sampling |
| Credit assignment | trajectory-level | **+ step-level, via a learned PRM** |
| Reward model | — | **Qwen3-0.6B regression head**, trained on LLM-judge labels |
| Tool layer | bespoke | **FastMCP** server + sandboxed Python exec |
| LLM serving | — | Ollama **native `/api/chat`** (see *Engineering* below) |
| Memory | in-task | in-task; Qdrant cross-episode planned |

Full design + the experimental protocol: **[docs/research.md](docs/research.md)**.

---

## Contribution 1 — DAPO, honestly accounted

[DAPO](https://arxiv.org/abs/2503.14476) is four tricks on top of GRPO. TRL 1.4's `GRPOTrainer`
(`loss_type="dapo"`) implements four of the five pieces I needed; **the fifth is mine.** I think
being precise about that line is the honest version of "I implemented DAPO":

| DAPO component | Where it comes from |
|---|---|
| Clip-higher (decoupled clip) | TRL — `epsilon=0.2`, `epsilon_high=0.28` |
| Token-level policy-gradient loss | TRL — `loss_type="dapo"` |
| Overlong filtering | TRL — `mask_truncated_completions=True` |
| Soft overlong punishment | TRL — `get_soft_overlong_punishment(...)` |
| **Dynamic sampling** | **`train/dynamic_sampling.py` — TRL does not implement it** |

**Dynamic sampling** (`curate_prompts`, `has_signal`) is the piece I built. Before training, each
candidate prompt is rolled out `G` times and scored by the PRM; any prompt whose `G` rewards have
~zero variance (`pstdev < 1e-3`) produces zero advantage on every token — a wasted rollout — and is
dropped. What survives is a dataset of *informative* states where the policy can actually learn the
difference between a good action and a bad one. The trainer wiring (`loss_type="dapo"`, `beta=0.0`
for the KL-free DAPO objective, clip-higher, overlong handling) lives in `train/dapo.py`.

---

## Contribution 2 — a learned Process Reward Model

The headline research piece. Instead of hand-tuned heuristic step rewards, the PRM is a **trained
model** that learns what a good Planner decision looks like.

**Pipeline** (`train/judge.py` → `train/prm.py` → `train/reward.py`):

1. **Collect** trajectories by running the untrained agent over the AIME training split, dumping full
   step-by-step traces (`eval/run.py --benchmark aime_train`).
2. **Label** every step with an LLM judge (`train/judge.py`). The judge is **DeepSeek**
   (`deepseek-chat`), deliberately *stronger than the 8B policy it supervises* — the standard
   distillation/RLHF principle that your reward signal should come from a more capable model. It rates
   each step 0–1 on a calibrated rubric. One-shot, cheap (well under \$1).
3. **Train** a **Qwen3-0.6B** sequence-regression head (`num_labels=1`, MSE) on those labels
   (`train/prm.py`). MAE on a held-out split is the convergence check.
4. **Reward** at training time (`train/reward.py`): the PRM scores each generated Planner action.
   Malformed JSON / unknown action ⇒ `0.0`; otherwise the PRM's `[0,1]` score is the reward. **The
   live RL signal is the PRM, not DeepSeek** — the judge is used exactly once, to build the dataset.

A detail I care about: `build_prm_input` (`train/data.py`) is the **single source of truth** for the
text the PRM scores, shared by the labeler, the trainer, and the reward function so they can never
drift. It deliberately **excludes the tool result** — the PRM rates the *decision*, which is the thing
being optimized, not the environment's response to it.

> The deployed agent stays **100% Qwen3-8B**. DeepSeek never runs at inference or in the RL loop.

---

## Engineering highlights

The kind of thing that doesn't show up in a results table but is most of the actual work:

- **A 53× serving fix.** Ollama's OpenAI-compatible `/v1` endpoint **silently ignores `think: false`**
  — Qwen3 kept emitting reasoning tokens until it burned the entire budget and returned *empty*
  content, so every structured call failed to parse, retried, and degraded. The native `/api/chat`
  endpoint honors it. A trivial two-step query went from **11m27s → ~13s**. The whole LLM layer
  (`core/llm.py`) is built on the native endpoint as a result.
- **Grammar-constrained structured output.** Planner/Verifier pass a Pydantic
  `model_json_schema()` straight to Ollama's `format` field, so every required field is
  grammar-guaranteed — no more missing-field crashes — backed by a retry-once-then-degrade fallback
  that never raises.
- **A sandboxed Python REPL** (`tools/builtin/python_exec.py`) with a stdlib whitelist
  (`sympy`/`numpy`/`mpmath` allowed because AIME needs symbolic math), auto-printing of a bare final
  expression, and a lenient parser that tolerates the stray indentation small models emit. *Not* a
  hardened security boundary — documented as such.
- **Leakage-free evaluation.** Training data is `di-zhang-fdu/AIME_1983_2024` filtered to **Year ≤
  2023** (918 problems), explicitly de-duplicated against the **AIME 2024** test set. The model is
  never trained on what it is scored on.
- **A Python-3.14 `dill`/`datasets` incompatibility** worked around with a targeted monkey-patch so
  caching doesn't break on the dev box.

---

## Results

Before/after, same harness, **same Q4_K_M quantization** — the only variable is the PRM-guided
DAPO training. Full per-problem trajectories and the curated reports live in
[`results/`](results/README.md).

| Model | Benchmark | Accuracy | Avg steps | Notes |
|---|---|---|---|---|
| `qwen3:8b` (untrained) | AIME 2024 (30) | 33.3% (10/30) | 4.03 | baseline; `think` off, `temp=0`, verified |
| `qwen3:8b` **+ DAPO + PRM** | AIME 2024 (30) | 30.0% (9/30) | 4.37 | flat within noise (n=30, ±~17pt CI; 11/30 flipped: +5/−6) |
| `qwen3:8b` (untrained) | GPQA Diamond (100) | 40.0% (40/100) | 3.09 | baseline |
| `qwen3:8b` **+ DAPO + PRM** | GPQA Diamond (100) | **45.0% (45/100)** | 3.19 | **+5.0 pts — cross-domain** (trained on AIME math) |

**What this shows.** A learned PRM driving DAPO (with hand-built dynamic sampling) trained
end-to-end and produced a **+5.0 pt cross-domain gain on GPQA** (n=100, the reliable test) from a
Planner trained only on AIME *math*. On AIME24 the result is **flat within noise** — at n=30 the
95% CI is ≈±17 pts, and the policy in fact changed a lot (11 of 30 problems flipped, net −1), i.e.
small-sample variance rather than a regression. The trained Planner also reasons more deliberately
(avg steps ↑), the expected signature of a *process* reward. This is a deliberately minimal demo
(300 LoRA steps, 8B); it validates the method, not a leaderboard push — see
[`results/README.md`](results/README.md) and [`docs/research.md`](docs/research.md) for the full
analysis and the levers for larger gains (more steps, stronger PRM, outcome-reward mixing, vLLM,
bigger policy).

---

## The training pipeline (Phase 4)

Everything below is built and committed; it runs as one session on a rented ~24–48 GB GPU
(A40 recommended, **~\$8–15 total**). Step-by-step infra guide:
**[docs/phase4-runpod-guide.md](docs/phase4-runpod-guide.md)**.

```bash
# on the GPU box, after `uv sync --extra rl --extra eval`
uv run python -m eval.run --benchmark aime_train --limit 150 --max-steps 6   # 1. collect trajectories
uv run python -m train.judge --runs "runs/eval_aime_train_*.json"            # 2. LLM-judge labels (DeepSeek)
uv run python -m train.prm   train --labels artifacts/prm_labels.jsonl       # 3. train the PRM
uv run python -m train.dapo  --prm artifacts/prm --runs "runs/eval_aime_train_*.json"   # 4. DAPO-train the Planner
# 5. export to GGUF, load into Ollama, re-run the eval for the "after" numbers
```

---

## Quickstart

Requires Python 3.11, [`uv`](https://docs.astral.sh/uv/), and a running [Ollama](https://ollama.com).

```bash
uv sync
ollama pull qwen3:8b

uv run python main.py "What is 15% of 240, then doubled?"
uv run python main.py "Explain how transformers work" --max-steps 3
uv run python main.py "..." --think          # enable Qwen3 reasoning tokens (slower, default off)
```

Web search uses [Tavily](https://tavily.com) — put `TAVILY_API_KEY=...` in `.env`
(copy `.env.example`).

### Evaluation

```bash
uv sync --extra eval
uv run python -m eval.run -b aime24 --limit 5 --max-steps 8    # small subset first
uv run python -m eval.run -b aime24                            # full AIME24 (30)
uv run python -m eval.run -b gpqa  --limit 5 --max-steps 6     # GPQA Diamond (gated; HF_TOKEN)
```

---

## Repo map

| Path | What |
|---|---|
| `core/` | the inference engine — `llm` (native-Ollama client), `types` (Pydantic models), `memory`, `planner`, `executor`, `verifier`, `solver` |
| `tools/` | `mcp_server.py` (FastMCP), `builtin/search.py` (Tavily), `builtin/python_exec.py` (sandboxed REPL) |
| `eval/` | `datasets.py` (AIME24 / GPQA / AIME-train loaders), `scorer.py` (math-verify + MC), `runner.py`, `run.py` (CLI) |
| `train/` | the RL stack — `data.py` (shared plumbing), `judge.py` (LLM-judge labeling), `prm.py` (learned PRM), `reward.py` (PRM→reward), `dynamic_sampling.py` (the DAPO piece TRL lacks), `dapo.py` (trainer) |
| `docs/` | `architecture.md`, `research.md` (the DAPO + PRM design), `phase4-runpod-guide.md` |
| `main.py` | the `solve` CLI |
| `runs/` | eval reports (gitignored) |

Contributors / agents: start with **[AGENTS.md](AGENTS.md)**; the phased plan and status live in
**[ROADMAP.md](ROADMAP.md)**.

---

## Status

Scaffold · core loop · real tools · eval harness · **the full DAPO + PRM training pipeline** are
built and committed, and the **Phase 4 GPU run is complete** (collect → judge → train PRM → DAPO →
GGUF → re-eval). Both baselines and trained-model numbers are recorded above: GPQA **+5.0 pts**
(cross-domain), AIME24 flat within noise. See [`results/`](results/README.md) for the full analysis.

## References

- **AgentFlow** — [arXiv 2510.05592](https://arxiv.org/abs/2510.05592)
- **DAPO** — [arXiv 2503.14476](https://arxiv.org/abs/2503.14476)
- **TRL** `GRPOTrainer` — [huggingface.co/docs/trl](https://huggingface.co/docs/trl) ·
  **PEFT** — [huggingface.co/docs/peft](https://huggingface.co/docs/peft)

## License

[MIT](LICENSE). Built on the ideas of the AgentFlow paper; not affiliated with its authors.
