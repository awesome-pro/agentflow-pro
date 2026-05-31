"""DAPO training of the Planner — the headline contribution.

Step-level DAPO (Decoupled Clip + Dynamic Sampling Policy Optimization),
PRM-guided. The policy is Qwen3-8B in bf16 + a LoRA adapter (PEFT); the reward is
the trained PRM. TRL's GRPOTrainer with `loss_type="dapo"` provides clip-higher,
token-level loss, and overlong filtering; dynamic sampling is our own
(train/dynamic_sampling.py — TRL does not implement it).

Runs on the GPU box (the `rl` extra: torch + transformers + trl + peft). A 48GB
card fits the 8B in bf16 + LoRA without quantization. Heavy imports are deferred
into `main` so the module imports without them.

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
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.rewards import get_soft_overlong_punishment

    from train.dynamic_sampling import curate_prompts
    from train.prm import PRM
    from train.reward import make_prm_reward

    rows = build_prompt_dataset(runs)
    if not rows:
        raise SystemExit(f"No prompts from {runs!r} — run the trajectory collection first.")
    print(f"{len(rows)} unique Planner states")

    # --- policy: Qwen3-8B in bf16 + a LoRA adapter (PEFT) ---
    # A 48GB card (A40) fits the 8B in bf16 plus LoRA without quantization, so no
    # bitsandbytes/unsloth — fewer, more reproducible deps. TRL wraps the model
    # with `peft_config` below; we keep a handle for the dynamic-sampling pass.
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        # sdpa (PyTorch's fused scaled-dot-product attention) is 3-4x faster than
        # eager for both the rollout generation and training — eager materialises
        # the full attention matrix and made the dynamic-sampling pass crawl. `dtype`
        # replaces the now-deprecated `torch_dtype` kwarg.
        base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
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
                # 256 is plenty for curation: a valid Planner JSON action is short
                # and emits EOS well before this; the cap only truncates runaway
                # rambling (which would be invalid JSON and dropped anyway), so it
                # speeds up the pass with negligible effect on the signal estimate.
                out = model.generate(**enc, max_new_tokens=256, do_sample=True,
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
        # NB: TRL 1.5.1's GRPOConfig dropped `max_prompt_length` — only
        # `max_completion_length` remains. Our prompts are bounded already
        # (train.data._format_context caps each prior step at 200 chars), so
        # there is nothing to truncate on the prompt side.
        max_completion_length=1024,
        learning_rate=1e-6,
        max_steps=train_steps,
        logging_steps=5,
        save_steps=train_steps,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,   # TRL wraps the policy as a LoRA model
    )
    trainer.train()

    # Save the LoRA adapter, then merge it into the base weights (bf16) and save a
    # standalone model so it can be converted to GGUF for the after-training eval.
    trainer.save_model(out)                                    # the LoRA adapter
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(f"{out}-merged", safe_serialization=True)
    tokenizer.save_pretrained(f"{out}-merged")
    print(f"adapter -> {out}  |  merged bf16 model -> {out}-merged")


if __name__ == "__main__":
    app()
