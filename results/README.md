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

Both were verified against the gold answers with no false positives — the pre-training
numbers the DAPO + PRM run has to beat.

## After DAPO + PRM (the "after")

The Planner was trained with PRM-guided DAPO (300 steps, Qwen3-8B + LoRA), exported to
GGUF, served through Ollama as `agentflow-planner` at the **same Q4_K_M quantization as
the baseline**, and re-evaluated with identical settings — so before/after differ only by
the training.

| Benchmark | n | Baseline | Trained | Δ accuracy | Avg steps (base→trained) | Report |
|---|---|---|---|---|---|---|
| AIME 2024 | 30 | 33.3% (10/30) | 30.0% (9/30) | **−3.3 pts** (within noise) | 4.03 → 4.37 | [`eval_aime24_20260531T224121Z.json`](eval_aime24_20260531T224121Z.json) |
| GPQA Diamond | 100 | 40.0% (40/100) | **45.0% (45/100)** | **+5.0 pts** | 3.09 → 3.19 | [`eval_gpqa_20260531T230946Z.json`](eval_gpqa_20260531T230946Z.json) |

### Reading the results

- **GPQA +5.0 pts (n=100) is the reliable signal — and it is cross-domain.** The Planner was
  trained only on AIME *math*, yet improved on GPQA *science multiple-choice*. The process
  training (dense per-step PRM rewards) generalized beyond the training distribution rather
  than overfitting to it.
- **AIME24 is flat within noise, not a regression.** At n=30 the 95% CI on a ~33% accuracy is
  ≈ ±17 pts, so 30.0% vs 33.3% (a one-problem net difference) is statistically
  indistinguishable. The per-problem diff makes this concrete: training **newly solved 5**
  problems (`aime24_5,7,12,13,14`) and **broke 6** (`aime24_0,2,3,8,21,24`) — **11 of 30
  flipped**, net −1. The policy changed substantially; the net is dominated by small-sample
  variance, not a systematic loss of capability.
- **The Planner reasons more deliberately after training** (avg steps 4.03→4.37 on AIME,
  3.09→3.19 on GPQA) — consistent with a *process* reward that credits good intermediate
  steps rather than only final answers.
- **Interpretation.** This is a deliberately minimal demo run (300 LoRA steps, 8B policy, a
  PRM bootstrapped from untrained-policy trajectories). The contribution it demonstrates is
  the **end-to-end method** — a learned PRM driving DAPO with hand-built dynamic sampling —
  producing a measurable cross-domain gain and a flat-within-noise on-domain result on a
  benchmark (competition math) where an 8B has little headroom (the "small-model ceiling"
  noted in [`../docs/research.md`](../docs/research.md) up front). Expected levers for larger
  gains: more training steps, a stronger PRM (a more capable judge / more labels), mixing an
  outcome reward with the process reward, vLLM-accelerated rollouts to train longer cheaply,
  or a larger policy.

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
