"""LLM-judge labeling for the PRM.

Scores every collected Planner step 0–1 to build PRM training data. Default
judge is DeepSeek (`deepseek-chat`) — stronger than the 8B policy and cheap.
A free local Ollama judge is available (`--provider ollama`) for testing.

    uv run python -m train.judge --runs "runs/eval_aime_train_*.json"
    uv run python -m train.judge --provider ollama --limit 5   # free smoke test
"""
import json
import os
import statistics
import time
from pathlib import Path

import typer
from dotenv import load_dotenv

from core.llm import OllamaClient
from train.data import Step, PRMExample, build_prm_input, load_steps

load_dotenv()
app = typer.Typer(help="LLM-judge — label collected steps for PRM training")

_SYSTEM = """You are an expert grader of mathematical problem-solving agents.
You see a problem, the agent's progress so far, and its proposed next step.
Rate ONLY that next step as a move toward correctly solving the problem.

Scoring guide:
  1.0  excellent — correct and makes clear progress
  0.7  good — reasonable and helpful
  0.5  mediocre — not wrong but weak or redundant
  0.2  poor — confused, repeats work, or a likely dead end
  0.0  bad — incorrect, broken, or actively harmful

Respond with JSON only: {"score": <number 0.0-1.0>, "reason": "<one sentence>"}"""


def _prompt(step: Step) -> str:
    return (
        f"Problem:\n{step.problem}\n\n"
        f"Progress so far:\n{step.context or 'No previous steps.'}\n\n"
        f"Proposed next step:\n"
        f"Thought: {step.thought}\n"
        f"Action: {step.action}\n"
        f"Action input: {step.action_input}\n"
        f"Tool result: {step.result}\n\n"
        f"Rate this step."
    )


def _parse_score(raw: str) -> float | None:
    try:
        return max(0.0, min(1.0, float(json.loads(raw)["score"])))
    except Exception:
        return None


class Judge:
    """Backend-agnostic judge — wraps a `complete(messages) -> str` callable."""

    def __init__(self, complete):
        self._complete = complete

    def score(self, step: Step) -> float:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _prompt(step)},
        ]
        for attempt in range(3):
            try:
                s = _parse_score(self._complete(messages))
                if s is not None:
                    return s
            except Exception:
                time.sleep(2 * (attempt + 1))
        return 0.5  # neutral fallback if the judge keeps failing


def _deepseek_complete(model: str):
    from openai import OpenAI

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError("Set DEEPSEEK_API_KEY in .env to use the DeepSeek judge.")
    client = OpenAI(base_url="https://api.deepseek.com", api_key=key)

    def complete(messages: list[dict]) -> str:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=200,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    return complete


def _ollama_complete(model: str, base_url: str):
    client = OllamaClient(base_url=base_url, model=model, think=False)

    def complete(messages: list[dict]) -> str:
        return client.complete(messages, temperature=0.0, max_tokens=200)

    return complete


@app.command()
def main(
    runs: str = typer.Option("runs/eval_aime_train_*.json", "--runs", help="glob for trajectory JSON"),
    out: str = typer.Option("artifacts/prm_labels.jsonl", "--out"),
    provider: str = typer.Option("deepseek", "--provider", help="deepseek (recommended) or ollama"),
    model: str = typer.Option("", "--model", help="judge model override"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
    limit: int | None = typer.Option(None, "--limit", help="label only the first N steps"),
):
    steps = load_steps(runs)
    if not steps:
        typer.echo(f"No steps found for {runs!r} — run the collection first.")
        raise typer.Exit(1)
    if limit:
        steps = steps[:limit]

    if provider == "deepseek":
        complete = _deepseek_complete(model or "deepseek-chat")
    elif provider == "ollama":
        complete = _ollama_complete(model or "qwen3:8b", base_url)
    else:
        typer.echo(f"Unknown provider {provider!r} — use 'deepseek' or 'ollama'.")
        raise typer.Exit(1)
    judge = Judge(complete)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    scores: list[float] = []
    with open(out, "w") as f:
        for i, step in enumerate(steps, 1):
            score = judge.score(step)
            ex = PRMExample(
                text=build_prm_input(step.problem, step.context, step.thought,
                                     step.action, step.action_input),
                score=score,
                meta={"action": step.action, "outcome_correct": step.outcome_correct},
            )
            f.write(ex.model_dump_json() + "\n")
            scores.append(score)
            print(f"[{i}/{len(steps)}] {step.action:7s} score={score:.2f}")

    print(f"\nWrote {len(scores)} PRM labels -> {out}")
    print(f"score: mean={statistics.mean(scores):.3f} min={min(scores):.2f} max={max(scores):.2f}")


if __name__ == "__main__":
    app()
