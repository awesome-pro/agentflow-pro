"""DAPO Dynamic Sampling — the one DAPO component TRL does not implement.

TRL's GRPOTrainer keeps prompt groups whose completions all get the same reward
(zero std -> zero advantage -> zero gradient) — wasted compute. DAPO resamples
to keep batches full of *informative* groups.

Here it is applied as a curation pass before training: for each candidate
prompt, sample G completions, score them, and keep only prompts whose rewards
show variance. (The fully-online per-batch variant is a documented extension.)
"""
import statistics
from typing import Callable


def has_signal(group_rewards: list[float], min_std: float = 1e-3) -> bool:
    """True if a prompt's G completion rewards vary enough to give a gradient."""
    if len(group_rewards) < 2:
        return False
    return statistics.pstdev(group_rewards) > min_std


def curate_prompts(
    prompts: list[dict],
    generate: Callable[[dict, int], list[str]],
    reward_fn: Callable[..., list[float]],
    num_generations: int = 6,
    min_std: float = 1e-3,
    keep: int | None = None,
) -> list[dict]:
    """Filter `prompts` to those that produce reward variance.

    prompts    — prompt rows; each carries the columns `reward_fn` needs
                 (everything except the `prompt` key is forwarded as a column).
    generate   — callable(prompt_row, n) -> n sampled completion strings.
    reward_fn  — a TRL-style reward function: (completions, **columns) -> list[float].
    keep       — optional cap; stop once this many informative prompts are found.

    Returns the informative subset, in order.
    """
    kept: list[dict] = []
    for row in prompts:
        completions = generate(row, num_generations)
        columns = {k: [v] * len(completions) for k, v in row.items() if k != "prompt"}
        rewards = reward_fn(completions=completions, **columns)
        if has_signal(rewards, min_std):
            kept.append(row)
            if keep is not None and len(kept) >= keep:
                break
    return kept
