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


def _norm(text: str) -> str:
    """Normalize problem text for dedup — collapse whitespace, lowercase."""
    return " ".join(text.split()).lower()


def load_aime_train() -> list[Task]:
    """Historical AIME problems (1983–2023) for RL training.

    Disjoint from the AIME 2024 test set: the `Year <= 2023` filter is the
    decontamination, and a text-level dedup against AIME24 is kept as auditable
    insurance. Shuffled with a fixed seed so `--limit N` gives a spread of years.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise ImportError("Install eval deps: uv sync --extra eval")

    _patch_datasets_for_py314()
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")

    test_texts = {_norm(t.question) for t in load_aime24()}
    tasks: list[Task] = []
    skipped_2024 = 0
    skipped_dup = 0
    skipped_badans = 0
    for row in ds:
        try:
            year = int(row.get("Year"))
        except (TypeError, ValueError):
            continue
        if year >= 2024:
            skipped_2024 += 1
            continue
        problem = (row.get("Question") or "").strip()
        answer = str(row.get("Answer") or "").strip()
        if not problem:
            continue
        # AIME answers are integers 0–999; skip the rare ambiguous-key rows.
        if not answer.isdigit():
            skipped_badans += 1
            continue
        question = problem + _BOXED_INSTRUCTION
        if _norm(question) in test_texts:
            skipped_dup += 1
            continue
        rid = str(row.get("ID") or len(tasks))
        tasks.append(Task(id=f"train_{rid}", question=question, gold=answer, kind="math"))

    print(f"[load_aime_train] {len(tasks)} train problems (excluded {skipped_2024} from 2024, "
          f"{skipped_dup} text-dups vs AIME24, {skipped_badans} non-integer answers)")
    random.Random(42).shuffle(tasks)
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
