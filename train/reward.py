"""DAPO reward function — wraps the trained PRM as the training signal.

TRL's GRPOTrainer calls each reward function with `completions` plus every
dataset column (here `problem` and `context`) and expects a `list[float]`.
The reward is the PRM's score of the Planner's generated action; a malformed
(non-JSON, missing fields, unknown action) completion gets 0.0.
"""
import json
from typing import Callable

_VALID_ACTIONS = {"think", "search", "code", "answer"}


def _completion_text(completion) -> str:
    """TRL gives conversational completions as a list of message dicts and
    standard ones as a plain string — normalize to text."""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    return completion or ""


def _parse_action(text: str) -> dict | None:
    """Parse a generated Planner completion into {thought, action, action_input}."""
    try:
        data = json.loads(text)
    except Exception:
        return None
    if not all(k in data for k in ("thought", "action", "action_input")):
        return None
    if str(data["action"]).strip().lower() not in _VALID_ACTIONS:
        return None
    return data


def make_prm_reward(prm, build_input: Callable) -> Callable:
    """Build a TRL reward function from a trained PRM.

    `prm`         — anything with `.score(list[str]) -> list[float]` (train.prm.PRM).
    `build_input` — `train.data.build_prm_input`.
    """
    def prm_reward(completions, problem=None, context=None, **kwargs) -> list[float]:
        n = len(completions)
        problems = problem if problem is not None else [""] * n
        contexts = context if context is not None else [""] * n

        # Only score the VALID completions. Malformed ones (non-JSON, missing
        # fields, unknown action) get a forced 0.0 and are never sent to the PRM.
        # Crucial: feeding empty placeholder strings to the PRM tokenizer yields
        # input_ids of shape [k, 0] when an entire group is malformed (common
        # early in training / during dynamic sampling, when the untrained policy
        # rarely emits valid Planner JSON), which crashes the model's forward.
        valid_texts: list[str] = []
        valid_pos: list[int] = []
        for i, (comp, prob, ctx) in enumerate(zip(completions, problems, contexts)):
            data = _parse_action(_completion_text(comp))
            if data is not None:
                valid_texts.append(build_input(prob, ctx, data["thought"],
                                               data["action"], data["action_input"]))
                valid_pos.append(i)

        rewards = [0.0] * n
        if valid_texts:
            scores = prm.score(valid_texts)
            for i, s in zip(valid_pos, scores):
                rewards[i] = s
        return rewards

    prm_reward.__name__ = "prm_reward"
    return prm_reward
