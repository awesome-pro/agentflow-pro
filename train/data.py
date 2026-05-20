"""Shared data plumbing for Step 4 training.

One source of truth for: pulling Planner steps out of collected trajectories,
and the exact text format the PRM scores — used by judge.py to label, by prm.py
to train, and by the DAPO reward function to score generated actions.
"""
import json
from pathlib import Path

from pydantic import BaseModel


class Step(BaseModel):
    """One Planner decision extracted from a collected trajectory."""
    problem: str
    context: str            # formatted summary of the steps before this one
    thought: str
    action: str
    action_input: str
    result: str             # executor output — the judge sees this, the PRM does not
    outcome_correct: bool    # did the trajectory this step belongs to end correct?


class PRMExample(BaseModel):
    """A formatted PRM input plus its judge label — one training row."""
    text: str
    score: float
    meta: dict = {}


def build_prm_input(problem: str, context: str, thought: str, action: str, action_input: str) -> str:
    """The exact text the PRM scores — single source of truth.

    judge.py labels this, prm.py trains on it, and the DAPO reward function
    scores generated actions with it. Deliberately excludes the tool result:
    the PRM rates the Planner's *decision*, which is what gets trained.
    """
    return (
        f"Problem:\n{problem}\n\n"
        f"Progress so far:\n{context or 'No previous steps.'}\n\n"
        f"Proposed next step:\n"
        f"Thought: {thought}\n"
        f"Action: {action}\n"
        f"Action input: {action_input}"
    )


def _format_context(prior: list[dict]) -> str:
    if not prior:
        return ""
    lines: list[str] = []
    for e in prior:
        ai = (e.get("action_input") or "")[:200]
        res = (e.get("result") or "")[:200]
        lines.append(f"Step {e.get('step')}: {e.get('action')}({ai}) -> {res}")
    return "\n".join(lines)


def load_steps(runs_glob: str) -> list[Step]:
    """Pull every Planner step out of eval-runner trajectory JSON files."""
    steps: list[Step] = []
    for path in sorted(Path(".").glob(runs_glob)):
        report = json.loads(path.read_text())
        for result in report.get("results", []):
            traj = result.get("trajectory", [])
            problem = result.get("question", "")
            correct = bool(result.get("correct", False))
            for i, entry in enumerate(traj):
                steps.append(Step(
                    problem=problem,
                    context=_format_context(traj[:i]),
                    thought=entry.get("thought", ""),
                    action=entry.get("action", ""),
                    action_input=entry.get("action_input", ""),
                    result=entry.get("result", ""),
                    outcome_correct=correct,
                ))
    return steps


def load_prm_examples(path: str) -> list[PRMExample]:
    """Read a JSONL file of PRMExample rows (written by judge.py)."""
    rows: list[PRMExample] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(PRMExample.model_validate_json(line))
    return rows
