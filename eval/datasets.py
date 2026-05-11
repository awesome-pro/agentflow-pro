import random
from typing import Literal
from pydantic import BaseModel

_BOXED_INSTRUCTION = "\n\nWork step by step. End your final answer with \\boxed{...}."
_MC_INSTRUCTION = "\n\nEnd your response with 'Answer: <letter>'."


class Task(BaseModel):
    id: str
    question: str
    gold: str
    kind: Literal["math", "mc"]


def _patch_datasets_for_py314() -> None:
    """
    dill's Pickler._batch_setitems changed signature in Python 3.14, breaking
    datasets' legacy-cache fingerprint check. Patch it to a no-op — there's
    no old-style cache to find on a fresh install anyway.
    """
    try:
        from datasets.builder import DatasetBuilder  # type: ignore
        DatasetBuilder._use_legacy_cache_dir_if_possible = lambda self, *a, **kw: None
    except Exception:
        pass


def load_aime24() -> list[Task]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install eval deps: uv sync --extra eval")

    _patch_datasets_for_py314()
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        problem = row.get("Problem") or row.get("problem") or ""
        answer = str(row.get("Answer") or row.get("answer") or "")
        tasks.append(Task(
            id=f"aime24_{i}",
            question=problem + _BOXED_INSTRUCTION,
            gold=answer.strip(),
            kind="math",
        ))
    return tasks


def load_gpqa_diamond() -> list[Task]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install eval deps: uv sync --extra eval")

    _patch_datasets_for_py314()
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        if "authentication" in str(e).lower() or "gated" in str(e).lower() or "token" in str(e).lower():
            raise EnvironmentError(
                "GPQA Diamond is a gated dataset. Run `huggingface-cli login` or set HF_TOKEN."
            ) from e
        raise

    letters = ["A", "B", "C", "D"]
    tasks: list[Task] = []
    for i, row in enumerate(ds):
        correct = row.get("Correct Answer") or ""
        wrong = [
            row.get("Incorrect Answer 1") or "",
            row.get("Incorrect Answer 2") or "",
            row.get("Incorrect Answer 3") or "",
        ]
        options = [correct] + wrong
        rng = random.Random(42 + i)
        rng.shuffle(options)
        gold_letter = letters[options.index(correct)]
        choices = "\n".join(f"{l}) {o}" for l, o in zip(letters, options))
        question = (row.get("Question") or "") + "\n" + choices + _MC_INSTRUCTION
        tasks.append(Task(
            id=f"gpqa_{i}",
            question=question,
            gold=gold_letter,
            kind="mc",
        ))
    return tasks
