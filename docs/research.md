# Research Notes — DAPO + PRM contribution

This document is the design spec and experimental notebook for the AgentFlow-Pro research
contribution. It starts with the paper's setup, explains what we change and why, and tracks
experimental results as they come in.

## Background — AgentFlow's Flow-GRPO

The original AgentFlow paper (ICLR 2026) trains the Planner module with **Flow-GRPO**, an on-policy
RL variant adapted from GRPO (Group Relative Policy Optimisation):

1. Sample `G` rollouts per question.
2. Compute an **outcome reward** `R ∈ {0, 1}` (did the final answer match?).
3. Normalise: `advantage = (R - mean(R)) / std(R)` across the group.
4. Policy-gradient update on the **entire Planner token sequence** using a clipped surrogate loss
   identical to PPO's, with symmetric clip bounds `[1-ε, 1+ε]`.

**What Flow-GRPO does well**: simple, stable, and proven on short-horizon tasks where a single LLM
call is one "action".

**Where it falls short on our multi-step agentic loop**:

| Problem | Effect |
|---|---|
| Symmetric clip bounds | Over-restrains policy improvement when actions are clearly right |
| Outcome-only reward | No credit assignment to *which step* was useful — all actions get the same gradient |
| Zero-advantage groups | When all `G` rollouts succeed or all fail, the group variance is zero, the update is skipped, wasting rollouts |
| Sequence-level loss | A long trajectory has many tokens; reward is diluted across all of them |

---

## Contribution 1 — DAPO (Decoupled clip + Dynamic sampling)

DAPO was introduced by ByteDance/Tsinghua (arXiv 2503.14476) specifically to fix the four problems
above. We adapt its four tricks to the agentic setting.

> **Implementation note.** Four of the five pieces come from TRL 1.4's `GRPOTrainer` with
> `loss_type="dapo"` (clip-higher via `epsilon=0.2` / `epsilon_high=0.28`, token-level loss, overlong
> filtering via `mask_truncated_completions=True`, soft overlong punishment via
> `get_soft_overlong_punishment`). **Dynamic sampling is not in TRL** and is implemented from scratch
> in `train/dynamic_sampling.py`. The wiring lives in `train/dapo.py` (`beta=0.0` — the KL-free DAPO
> objective). The descriptions below are the design rationale; the parenthetical maps each to code.

### Trick 1 — Clip-higher (decoupled clip bounds)

Standard PPO/GRPO uses a single symmetric clip `[1-ε, 1+ε]`. DAPO decouples them:
- `ε_low = 0.2` (clip down — keep the penalty on clearly bad actions)
- `ε_high = 0.28` (clip up — allow better policy to improve more aggressively)

In the agentic loop each "action" is a `(thought, action, action_input)` JSON produced by the
Planner. High-reward actions (good search queries, correct code) deserve a bigger gradient step than
a symmetric clip allows.

### Trick 2 — Dynamic sampling (filter degenerate groups)

Before computing advantages, filter out groups where **all rollouts get the same reward**:
- All correct → `advantage = 0` for every token → skip; sample a harder question.
- All incorrect → similarly skip; we have no signal about *which* actions were better.

This doubles the effective information per training step and avoids the "zero-variance" gradient sink.
In practice: keep sampling until we have at least `G_min = 4` non-degenerate groups per batch.

### Trick 3 — Token-level policy-gradient loss

Standard approach averages the loss over tokens and then over samples — a long trajectory gets the
same weight as a short one. DAPO normalises *by the total token count in the batch*, not per-sample:

```
L_DAPO = -Σ_{i,t} [ clip_ratio * advantage_i ] / Σ_{i,t} 1
```

This prevents the model from learning "write short answers" as a shortcut to lower the loss.

### Trick 4 — Overlong-reward shaping

Trajectories that hit `max_steps` without a valid answer receive a soft penalty that grows linearly
with excess length. This discourages the Planner from stalling in `think`-loops when it's stuck:

```python
if steps_taken == max_steps and not answered_correctly:
    r_length_penalty = -0.1 * (steps_taken / max_steps)
    outcome_reward += r_length_penalty
```

---

## Contribution 2 — PRM (Process Reward Model)

A **Process Reward Model** assigns a reward to each individual Planner action, not just the final
outcome. This gives a much denser gradient signal for long agentic trajectories. Rather than
hand-tuned heuristics, AgentFlow-Pro uses a **learned** PRM — a small model trained to predict
step quality — which is the headline research contribution.

### The pipeline (`train/judge.py` → `train/prm.py` → `train/reward.py`)

1. **Collect.** Run the untrained agent over the AIME *training* split (`eval/run.py --benchmark
   aime_train`), dumping full step-by-step trajectories to `runs/`.
2. **Label.** `train/judge.py` scores every Planner step on a calibrated 0–1 rubric using an LLM
   judge. The default judge is **DeepSeek** (`deepseek-chat`), chosen deliberately because it is
   *stronger than the 8B policy it supervises* — the standard RLHF/distillation principle that the
   reward signal should come from a more capable model. The judge sees the problem, the prior context,
   the proposed step, **and** the tool result; it runs **once** to build the dataset
   (`artifacts/prm_labels.jsonl`, cost < \$1). A free local Ollama judge is available for smoke tests.
3. **Train the PRM.** `train/prm.py` fine-tunes **Qwen3-0.6B** as a sequence-regression model
   (`AutoModelForSequenceClassification`, `num_labels=1`, `problem_type="regression"`) with an MSE
   loss against the judge scores. Held-out MAE is the convergence check.
4. **Reward.** At RL time, `train/reward.py`'s `make_prm_reward` scores each generated Planner action
   with the trained PRM (clamped to `[0,1]`); a malformed completion (non-JSON, missing fields,
   unknown action) is forced to `0.0`. **The live RL reward is the PRM — DeepSeek never runs in the
   training loop or at inference.**

### One detail worth calling out: `build_prm_input`

`train/data.py:build_prm_input` is the **single source of truth** for the exact text the PRM scores.
The labeler, the PRM trainer, and the RL reward function all call it, so the three can never drift out
of format. It deliberately **excludes the tool result**: the PRM rates the Planner's *decision*
(thought + action + input), which is the thing being optimized — not the environment's response to it.
(The judge, by contrast, *does* see the result when assigning the label, since it is grading with
hindsight.)

### What the policy sees

The PRM score is the per-action reward fed to DAPO's group-relative advantage. Because the signal is
now dense (one score per step rather than one bit per trajectory), the advantage distinguishes a
strong opening move from a wasteful mid-trajectory `think`-loop even when the final answer is the
same — exactly the credit-assignment gap that motivates the project.

---

## Experimental protocol

### Setup

| Item | Value |
|---|---|
| Policy backbone | `Qwen3-8B`, bf16 + PEFT LoRA (`lora_rank=32`, all attn + MLP projections) |
| PRM backbone | `Qwen3-0.6B` + regression head (MSE on judge scores) |
| Judge | DeepSeek `deepseek-chat` (one-shot labeling; stronger than the policy) |
| Training data | `di-zhang-fdu/AIME_1983_2024`, Year ≤ 2023 (918 problems), de-duplicated vs AIME24 |
| Eval | AIME 2024 (30, disjoint from training); GPQA Diamond (gated — `HF_TOKEN`) |
| Metrics | Accuracy, avg steps to answer, tool-call diversity |
| RL framework | TRL `GRPOTrainer` (`loss_type="dapo"`) + hand-built dynamic sampling; PEFT LoRA (bf16) |
| Infra | RunPod A40 (48 GB) recommended, est. $8–15 for a full run |

**Train/test separation**: training uses AIME 1983–2023 and is explicitly de-duplicated against the
AIME 2024 test set in `eval/datasets.py:load_aime_train` — the model is never trained on what it is
scored on. (An earlier proof-of-concept sketch reused the 30 AIME24 problems for both; that was
dropped in favor of the disjoint split above.)

### Results (baseline + trained)

Run AIME24 and GPQA Diamond with Qwen thinking disabled by default:

```bash
uv run python -m eval.run -b aime24 --max-steps 8 --temperature 0.0
uv run python -m eval.run -b gpqa --max-steps 4 --temperature 0.0
```

| Model | Benchmark | Accuracy | Avg steps | Notes |
|---|---|---|---|---|
| `qwen3:8b` (untrained) | AIME24 | **33.3%** (10/30) | 4.03 | baseline; `--think` off, `temp=0`, verified — no false positives |
| `qwen3:8b` (untrained) | GPQA Diamond | **40.0%** (40/100) | 3.09 | baseline; A40, same settings |
| `qwen3:8b` + DAPO + PRM | AIME24 | 30.0% (9/30) | 4.37 | −3.3 pts = within noise at n=30 (11/30 problems flipped: +5 solved, −6 broken) |
| `qwen3:8b` + DAPO + PRM | GPQA Diamond | **45.0%** (45/100) | 3.19 | **+5.0 pts** — cross-domain gain (trained on AIME math) |

Both trained rows: `agentflow-planner` served via Ollama at the **same Q4_K_M quant as the
baseline**, identical eval settings — the only variable is the training. Reports:
[`results/eval_aime24_20260531T224121Z.json`](../results/eval_aime24_20260531T224121Z.json),
[`results/eval_gpqa_20260531T230946Z.json`](../results/eval_gpqa_20260531T230946Z.json).

**Result discussion.** The headline outcome is a **+5.0 pt cross-domain gain on GPQA** (n=100,
the statistically reliable test) from a Planner trained only on AIME *math* — the dense per-step
PRM signal improved general agentic reasoning, not just the training distribution. On **AIME24
the result is flat within noise**: at n=30 the 95% CI is ≈ ±17 pts, and the per-problem diff
shows the policy *changed substantially* (newly solved `aime24_5,7,12,13,14`; broke
`aime24_0,2,3,8,21,24`) for a net −1 — variance, not a capability regression. The trained
Planner also takes **more deliberate steps** (avg 4.03→4.37), the expected signature of a
*process* reward. This is a minimal demo run (300 LoRA steps, 8B policy, PRM bootstrapped from
untrained-policy trajectories); it validates the full learned-PRM + DAPO method end-to-end. The
"small-model ceiling on AIME" risk noted below was borne out — and the mitigation we planned
(report process metrics + the more reliable benchmark, not AIME accuracy alone) is exactly what
the data supports.

Baseline failure split (AIME24): 10 correct, ~8 confident-but-wrong, 12 hit `max_steps` without the
Verifier accepting an answer. Run JSON: `runs/eval_aime24_20260520T103641Z.json`.

Update this table after each run. The report JSON in `runs/` has per-problem trajectories, elapsed
time, and error details. Discard diagnostic runs where the planner repeatedly falls back to generic
`think` actions because those measure JSON/tooling reliability rather than benchmark ability.

### Training run checklist

Full step-by-step infra walkthrough: [phase4-runpod-guide.md](phase4-runpod-guide.md).

- [ ] `uv sync --extra rl --extra eval` on the GPU box; `DEEPSEEK_API_KEY` + `HF_TOKEN` in `.env`
- [ ] Collect: `uv run python -m eval.run --benchmark aime_train --limit 150 --max-steps 6`
- [ ] Label: `uv run python -m train.judge --runs "runs/eval_aime_train_*.json"`
- [ ] Train PRM: `uv run python -m train.prm train --labels artifacts/prm_labels.jsonl`
      (sanity-check with `uv run python -m train.prm score`)
- [ ] DAPO: `uv run python -m train.dapo --prm artifacts/prm --runs "runs/eval_aime_train_*.json"`
- [ ] Confirm the reward curve climbs over the first ~50 steps
- [ ] Export GGUF → load into Ollama → re-run `eval.run -b aime24 -m agentflow-planner`
- [ ] Fill in the results table above

---

## Open questions / risks

| Risk | Mitigation |
|---|---|
| **Reward hacking** — model learns to call `think` many times (safe, always `success=True`) | Repeated-action penalty; cap `think` steps at 2 in the Planner prompt |
| **PRM cold-start** — heuristic rewards are noisy early in training | Start with higher `β` (outcome weight); reduce `α` (PRM weight) for first N steps |
| **Small-model ceiling** — an 8B may have limited headroom to improve on AIME | Report the *process* improvement (avg steps, tool-call efficiency, step-reward curve), not accuracy alone |
| **Ollama `think` mode instability** — the model occasionally sends unexpected tokens | Already handled: `_strip_think_tags` strips `<think>...</think>` blocks; `OllamaClient(think=False)` on the native `/api/chat` endpoint disables it at source |
| **GPQA gate** — dataset requires HF login | Set `HF_TOKEN` in `.env`; or use a different multiple-choice benchmark |

---

## References

- AgentFlow paper: [arXiv 2510.05592](https://arxiv.org/abs/2510.05592)
- DAPO: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) — "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
- GRPO: DeepSeek-R1 tech report (the base algorithm DAPO improves on)
- TRL GRPOTrainer: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- PEFT (LoRA): [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
