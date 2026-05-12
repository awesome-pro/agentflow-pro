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
outcome. This gives a much denser gradient signal for long agentic trajectories.

### What we're rewarding at the step level

For each step `t` with `action_t`:

| Signal | Implementation |
|---|---|
| **Verifier verdict** | `+0.3` if `Verifier.verify(...)` returns `sufficient=True` after this step |
| **Tool success** | `+0.1` if `ExecutorOutput.success=True` and the result is non-trivial (not an error string) |
| **Repeated action** | `-0.2` if `action_t == action_{t-1}` with the same input (stuck in a loop) |
| **Tool error** | `-0.2` if `ExecutorOutput.success=False` (search fail, code error) |
| **Progress bonus** | `+0.05` per unique action type used so far (diversity bonus) |

These are heuristics for **v1 PRM**. A **v2** learned PRM head (a small MLP on top of a frozen
Planner hidden state) can be trained offline on labelled trajectories once we have enough rollouts.

### Combining PRM + outcome reward into DAPO advantage

The final per-step reward used for the policy update:

```
r_t = α · PRM_score(state_t, action_t) + β · R_outcome · γ^(T - t)
```

Where:
- `α = 0.4`, `β = 0.6` — weight balance (tunable).
- `γ = 0.95` — discount; later steps get slightly less credit for the outcome.
- `R_outcome ∈ {0, 1}` — did the final answer match? (from `eval/scorer.py`).
- `T` = total steps in the trajectory.

The group advantage is then computed over the combined `r_t` values, not just `R_outcome`.

---

## Experimental protocol

### Setup

| Item | Value |
|---|---|
| Backbone | `qwen3.5:4b` |
| Fine-tuning | LoRA (`r=16`, `alpha=32`, target: `q_proj`, `v_proj`, `o_proj`) |
| Training data | AIME 2024 (30 problems) + additional math mix (MATH-500 sample, ~500 problems) |
| Eval | AIME 2024 (held-out — same 30 for fair comparison); GPQA Diamond (once HF_TOKEN set) |
| Metrics | Accuracy, avg steps to answer, tool-call diversity |
| Training infra | RunPod RTX 4090 (24 GB), est. $5–15 for a full run |
| Framework | TRL `GRPOTrainer` + custom DAPO advantage fn; Unsloth for LoRA efficiency |

**Important**: the 30 AIME 2024 problems serve as both a small training seed AND the eval set in the
first experiments (proof-of-concept). When a larger training corpus is available, move to a proper
train/eval split.

### Baseline (Phase 3 — to be filled)

Run: `uv run python -m eval.run -b aime24 --max-steps 8 --temperature 0.0`

| Model | AIME24 accuracy | Avg steps | Notes |
|---|---|---|---|
| `qwen3.5:4b` (untrained) | TBD | TBD | baseline; `--think` off |
| `qwen3.5:4b` + DAPO + PRM | TBD | TBD | Phase 4 result |

Update this table after each run. The report JSON in `runs/` has per-problem breakdowns.

### Training run checklist

- [ ] `uv sync --extra rl` on the GPU box
- [ ] Set `WANDB_API_KEY` in `.env` (optional; fallback to stdout logging)
- [ ] `uv run python -m train.run --config train/config.yaml`
- [ ] Confirm loss decreasing, reward increasing over first 50 steps
- [ ] Save checkpoint to `checkpoints/dapo_prm_<run_id>/`
- [ ] Re-run eval: `uv run python -m eval.run -b aime24 -m checkpoints/dapo_prm_<run_id>/`
- [ ] Fill in the results table above

---

## Open questions / risks

| Risk | Mitigation |
|---|---|
| **Reward hacking** — model learns to call `think` many times (safe, always `success=True`) | Repeated-action penalty; cap `think` steps at 2 in the Planner prompt |
| **PRM cold-start** — heuristic rewards are noisy early in training | Start with higher `β` (outcome weight); reduce `α` (PRM weight) for first N steps |
| **Small-model ceiling** — 4B may not have the capacity to improve on AIME significantly | That's fine for a portfolio project; report the trajectory improvement, not just accuracy |
| **Ollama `think` mode instability** — the model occasionally sends unexpected tokens | Already handled: `_strip_think_tags` strips `<think>...</think>` blocks; `extra_body={"think": False}` disables at source |
| **GPQA gate** — dataset requires HF login | Set `HF_TOKEN` in `.env`; or use a different multiple-choice benchmark |

---

## References

- AgentFlow paper: [arXiv 2510.05592](https://arxiv.org/abs/2510.05592)
- DAPO: [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) — "DAPO: An Open-Source LLM Reinforcement Learning System at Scale"
- GRPO: DeepSeek-R1 tech report (the base algorithm DAPO improves on)
- TRL GRPOTrainer: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- Unsloth LoRA: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
