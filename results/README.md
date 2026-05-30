# Results

Evaluation reports for AgentFlow-Pro — the full `EvalReport` JSONs (per-problem
predictions, gold answers, and complete step-by-step trajectories) kept here as
reproducible reference artifacts. (The raw `runs/` directory is gitignored; these
are the curated, citable runs.)

## Baselines — untrained `qwen3:8b` (the "before")

| Benchmark | n | Accuracy | Avg steps | Settings | Report |
|---|---|---|---|---|---|
| AIME 2024 | 30 | **33.3%** (10/30) | 4.03 | `max_steps=6`, `temp=0`, think off | [`eval_aime24_20260520T103641Z.json`](eval_aime24_20260520T103641Z.json) |
| GPQA Diamond | 100 | **40.0%** (40/100) | 3.09 | `max_steps=6`, `temp=0`, think off | [`eval_gpqa_20260530T103820Z.json`](eval_gpqa_20260530T103820Z.json) |

Both are honestly verified (no false positives). These are the pre-training numbers
the DAPO + PRM run has to beat.

## After DAPO + PRM (the "after") — _pending the training run_

| Benchmark | n | Accuracy | Δ vs baseline | Report |
|---|---|---|---|---|
| AIME 2024 | 30 | _pending_ | _pending_ | — |
| GPQA Diamond | 100 | _pending_ | _pending_ | — |

The trained Planner is produced by the pipeline in [`../docs/phase4-runpod-guide.md`](../docs/phase4-runpod-guide.md)
(collect → judge → PRM → DAPO → GGUF), then re-evaluated with the same settings.

## Notes

- **Train/test separation:** training uses AIME 1983–2023 (de-duplicated against the
  AIME 2024 test set in `eval/datasets.py:load_aime_train`). The model is never trained
  on what it is scored on.
- **Reproduce:**
  ```bash
  uv run python -m eval.run --benchmark aime24            --max-steps 6   # AIME24 (30)
  uv run python -m eval.run --benchmark gpqa  --limit 100 --max-steps 6   # GPQA Diamond (100)
  ```
- **A note on timing:** the AIME24 baseline was run on a local Mac (Apple M-series via
  Ollama); the GPQA baseline on a cloud A40. So the `avg_elapsed_seconds` in the two
  JSONs reflect different hardware and are **not** comparable — accuracy and step counts
  are the metrics to compare.
- Report schema is documented in [`../docs/architecture.md`](../docs/architecture.md).
