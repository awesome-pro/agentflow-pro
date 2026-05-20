# Architecture

Internal design of AgentFlow-Pro — how the solver loop works, what each module contracts, how tools
are wired up, and where Phase 4 (RL) and Phase 5 (Qdrant) plug in.

## Solver loop — step lifecycle

`Solver.solve(query)` runs up to `max_steps` iterations. Each iteration:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Planner.plan(query, memory.to_context())                                │
│    → PlannerOutput(thought, action, action_input)                        │
│                                                                          │
│  if action == ANSWER:                                                    │
│      return SolverResult immediately  ← early exit, no executor call    │
│                                                                          │
│  Executor.execute(plan)                                                  │
│    → ExecutorOutput(tool, result, success)                               │
│                                                                          │
│  memory.add(step, plan, executor_result)                                 │
│                                                                          │
│  Verifier.verify(query, memory.to_context(), executor_result.result)    │
│    → VerifierOutput(sufficient, reason, answer)                          │
│                                                                          │
│  if sufficient and answer is not None:                                   │
│      return SolverResult immediately  ← early exit                      │
└──────────────────────────────────────────────────────────────────────────┘
                       (loop repeats)
 if max_steps exhausted → SolverResult(answer="Max steps reached…")
```

Sequence diagram (time flows down):

```
 main.py          Solver         Planner      Executor     Verifier     Memory
    │                │               │            │            │           │
    │── solve(q) ──► │               │            │            │           │
    │                │── plan(q,ctx)►│            │            │           │
    │                │◄── PlanOut ───│            │            │           │
    │            [ANSWER?] ──────────────────────────────────────────────► return
    │                │── execute ──────────────► │            │           │
    │                │◄── ExecOut ───────────────│            │           │
    │                │── add(step, plan, exec) ───────────────────────────►│
    │                │── verify(q,ctx,res) ────────────────── ►│           │
    │                │◄── VerifyOut ──────────────────────────│           │
    │            [sufficient?] ──────────────────────────────────────────► return
    │                │        (loop ↑)
```

## Module contracts

### `core/types.py` — all data models

| Type | Fields |
|---|---|
| `Action` | `Enum str`: `think` `search` `code` `answer` |
| `PlannerOutput` | `thought: str`, `action: Action`, `action_input: str` |
| `ExecutorOutput` | `tool: str`, `result: str`, `success: bool` |
| `VerifierOutput` | `sufficient: bool`, `reason: str`, `answer: str \| None` |
| `MemoryEntry` | `step: int`, `thought: str`, `action: str`, `action_input: str`, `result: str` |
| `SolverResult` | `answer: str`, `steps_taken: int`, `trajectory: list[MemoryEntry]` |

### `core/planner.py` — Planner

- **Input**: `query: str`, `memory_context: str` (from `Memory.to_context()`).
- **Output**: `PlannerOutput`.
- **LLM call**: JSON Schema `response_format`, `max_tokens=512`, `temperature=0.7`.
- **Failure handling**: retry once with `_RETRY_MSG`; if still invalid JSON → degrade to `Action.THINK`
  with a safe input, never raises.
- **Think toggle**: `extra_body={"think": self._think}` (default `False`).

### `core/verifier.py` — Verifier

- **Input**: `query`, `memory_context`, `last_result: str`.
- **Output**: `VerifierOutput`.
- **Conservative by default**: parse failure → `sufficient=False` so the loop continues.
- `temperature=0.2` for deterministic judgement. `max_tokens=256`.
- **Think toggle**: `extra_body={"think": self._think}` (default `False`), matching the Planner.

### `core/memory.py` — Memory

- In-task only (cleared by `Solver.solve` at the start of each call).
- `to_context()` serialises entries as a plain-text list of `Step N: Thought / Action / Result` for
  the LLM prompts.
- Qdrant cross-episode backend is Phase 5 — it will sit behind the same `add` / `to_context` API.

### `core/executor.py` — Executor

Routing table (no LLM call; pure dispatch):

| Action | What happens |
|---|---|
| `think` | Echoes `action_input` as the result. Internal reasoning, no tool. |
| `answer` | Echoes `action_input`. (Solver intercepts before reaching Executor in practice.) |
| `search` | Calls `tools.builtin.search.web_search(action_input)`. Returns formatted Tavily results. |
| `code` | Calls `tools.builtin.python_exec.run_python(action_input)`. Returns stdout or error string. |
| unknown | Returns `ExecutorOutput(tool="unknown", result="Unrecognised action.", success=False)`. |

## Tool layer

### `tools/builtin/search.py` — Tavily web search

`web_search(query, max_results=5)` — calls `TavilyClient.search`, formats the `answer` + per-result
`title / url / content` into a single string. Returns a "Search unavailable" message if
`TAVILY_API_KEY` is unset (graceful degradation).

### `tools/builtin/python_exec.py` — sandboxed Python exec

`run_python(code)` — uses `exec()` with a stripped `__builtins__` dict:

- **Blocked builtins**: `exec`, `eval`, `compile`, `open`, `input`, `breakpoint`, `__import__`.
- **Import whitelist**: only pure-stdlib math/logic/data modules (`math`, `statistics`, `random`,
  `json`, `re`, `datetime`, `itertools`, `collections`, …). No `os`, `sys`, `subprocess`, `socket`.
- Captures stdout via `StringIO`. Returns the captured output, or the exception as a string.
- **Not a real security boundary** — `exec()` has known escapes. Do not run on untrusted infra.

### `tools/mcp_server.py` — FastMCP server

`FastMCP("agentflow-tools")` exposing `search()` and `python_exec()` as MCP tools — thin wrappers
around `web_search` and `run_python`. Run with:
```bash
uv sync --extra tools
uv run python -m tools.mcp_server
```
The Executor uses the builtin tools directly (no MCP round-trip); the MCP server exists so external
MCP clients (Claude Desktop, etc.) can call the same tools.

## Eval harness

### Dataset loaders (`eval/datasets.py`)

`Task(id, question, gold, kind: "math"|"mc")`. Both loaders append a per-kind instruction
(`\boxed{...}` for math, `Answer: <letter>` for MC) to the question text so the scorer has a
consistent extraction target.

- `load_aime24()`: `Maxwell-Jia/AIME_2024` (open, 30 problems). Math kind.
- `load_gpqa_diamond()`: `Idavidrein/gpqa` / `gpqa_diamond` (gated — needs `HF_TOKEN`). MC kind;
  options shuffled per-problem with a fixed seed so runs are reproducible.

### Scorer (`eval/scorer.py`)

1. `extract_final_answer(text)` — tries `\boxed{}` → `Answer: X` → last integer.
2. `score_math(pred, gold)` — `math_verify.verify(parse(gold), parse(pred))`; integer fallback; string
   fallback.
3. `score_mc(pred, gold_letter)` — regex for `Answer: X`; trailing capital letter fallback.

### Runner (`eval/runner.py`)

`run_eval(solver, tasks, benchmark, limit)`:
- Iterates tasks (Rich progress bar), calls `solver.solve(task.question)`, scores, accumulates
  `EvalResult` objects.
- Saves `runs/eval_<benchmark>_<timestamp>.json` (full `EvalReport` with trajectories and timing).
- Prints per-task start/end timing plus a Rich summary table.

Report JSON shape:
```json
{
  "benchmark": "aime24",
  "model": "qwen3:8b",
  "max_steps": 8,
  "temperature": 0.0,
  "think": false,
  "n": 30,
  "accuracy": 0.067,
  "avg_steps": 5.2,
  "avg_elapsed_seconds": 82.1,
  "timestamp": "20250512T...",
  "results": [
    { "id": "aime24_0", "correct": false, "steps_taken": 8,
      "predicted": "...", "gold": "...", "trajectory": [...] }
  ]
}
```

## Extension points

### Phase 4 — RL training hooks

The `Solver` is the **environment**. The training loop in `rl/trainer.py` will:
1. Call `Solver.solve(task.question)` with `verbose=False`.
2. After each step, intercept `(memory_context, PlannerOutput)` as `(state, action)`.
3. Score the completed trajectory with `rl/rewards.py` + `rl/prm.py` to get per-step rewards.
4. Run a DAPO gradient update on the Planner's LoRA weights.

No changes to `Solver`, `Planner`, or `Executor` are needed — the trainer wraps them as a black box
and reads `SolverResult.trajectory` for the gradient signal.

### Phase 5 — Qdrant episodic memory

`Memory.to_context()` currently returns only the current episode. To add cross-episode retrieval:
- Add a `QdrantMemory` subclass (or `backend=` param on the existing class).
- On `add()`: embed the `MemoryEntry` and upsert into a Qdrant collection keyed by task type.
- On `to_context()`: retrieve top-k similar past entries and prepend them to the current-episode context.
- Toggle via `--memory` flag on the CLIs; default off so the baseline is not contaminated.
