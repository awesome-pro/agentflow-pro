"""DAPO training of the Planner — the headline contribution.

Step-level DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization),
PRM-guided. The policy is Qwen3-8B (4-bit QLoRA via unsloth); the reward is the
trained PRM. TRL 1.4's GRPOTrainer with `loss_type="dapo"` provides clip-higher,
token-level loss, and overlong filtering; dynamic sampling is our own
(train/dynamic_sampling.py — TRL does not implement it).

Runs on the GPU box (the `rl` extra: torch + transformers + trl + unsloth).
Heavy imports are deferred into `main` so the module imports without them.

    uv run python -m train.dapo --prm artifacts/prm --runs "runs/eval_aime_train_*.json"
"""
import typer

from core.planner import _SYSTEM as PLANNER_SYSTEM
from train.data import build_prm_input, load_steps

app = typer.Typer(help="DAPO — PRM-guided RL fine-tuning of the Planner")


def _planner_user_msg(problem: str, context: str) -> str:
    """Mirror core.planner.Planner.plan()'s user message so training prompts
    match what the Planner sees at inference."""
    return (
        f"Question: {problem}\n\n"
        f"Steps taken so far:\n{context or 'No previous steps.'}\n\n"
        "What is the next best action?"
    )


def build_prompt_dataset(runs_glob: str) -> list[dict]:
    """Unique Planner states (problem + context) from collected trajectories.

    Each row: a conversational `prompt` (system + user messages) plus the
    `problem` / `context` columns the reward function reads.
    """
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for s in load_steps(runs_glob):
        key = (s.problem, s.context)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "prompt": [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": _planner_user_msg(s.problem, s.context)},
            ],
            "problem": s.problem,
            "context": s.context,
        })
    return rows


@app.command()
def main(
    prm: str = typer.Option("artifacts/prm", "--prm", help="trained PRM directory"),
    runs: str = typer.Option("runs/eval_aime_train_*.json", "--runs"),
    base_model: str = typer.Option("Qwen/Qwen3-8B", "--base-model"),
    out: str = typer.Option("artifacts/planner-dapo", "--out"),
    num_generations: int = typer.Option(6, "--num-generations"),
    train_steps: int = typer.Option(300, "--train-steps", help="optimizer steps"),
    lora_rank: int = typer.Option(32, "--lora-rank"),
    dynamic_sampling: bool = typer.Option(True, "--dynamic-sampling/--no-dynamic-sampling"),
):
    import torch
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import get_soft_overlong_punishment
    from unsloth import FastLanguageModel

    from train.dynamic_sampling import curate_prompts
    from train.prm import PRM
    from train.reward import make_prm_reward

    rows = build_prompt_dataset(runs)
    if not rows:
        raise SystemExit(f"No prompts from {runs!r} — run the trajectory collection first.")
    print(f"{len(rows)} unique Planner states")

    # --- policy: Qwen3-8B, 4-bit QLoRA (unsloth) ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,        # vLLM-backed rollouts, colocated
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,  # shares the 24GB with training; lower on OOM
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
    )

    # --- reward: the trained PRM (process reward) + DAPO soft overlong punishment ---
    prm_reward = make_prm_reward(PRM(prm), build_prm_input)
    reward_funcs = [
        prm_reward,
        get_soft_overlong_punishment(max_completion_len=1024, soft_punish_cache=256),
    ]

    # --- DAPO dynamic sampling — our own pre-training curation pass ---
    if dynamic_sampling:
        def generate(row: dict, n: int) -> list[str]:
            text = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True)
            enc = tokenizer([text] * n, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=512, do_sample=True,
                                     temperature=1.0, top_p=0.95)
            gen = out[:, enc["input_ids"].shape[1]:]
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

        before = len(rows)
        rows = curate_prompts(rows, generate, prm_reward, num_generations=num_generations)
        print(f"dynamic sampling: kept {len(rows)}/{before} informative prompts")
        if not rows:
            raise SystemExit("Dynamic sampling dropped every prompt — check the PRM/reward.")

    dataset = Dataset.from_list(rows)

    # --- DAPO config: clip-higher + token-level loss + overlong filtering ---
    args = GRPOConfig(
        output_dir=out,
        loss_type="dapo",                  # token-level policy-gradient loss
        epsilon=0.2, epsilon_high=0.28,    # clip-higher (decoupled clipping)
        mask_truncated_completions=True,   # overlong filtering
        beta=0.0,                          # no KL term (DAPO)
        num_generations=num_generations,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=4,
        max_prompt_length=1024,
        max_completion_length=1024,
        learning_rate=1e-6,
        max_steps=train_steps,
        logging_steps=5,
        save_steps=train_steps,
        optim="paged_adamw_8bit",
        bf16=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()

    # Merge LoRA and export GGUF so the trained Planner loads straight into Ollama
    # for the after-training eval. (unsloth API — verify on the GPU box.)
    model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    print(f"trained Planner saved -> {out}")


if __name__ == "__main__":
    app()
