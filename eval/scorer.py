import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datasets import Task


def extract_final_answer(text: str) -> str:
    # 1. \boxed{...}. The leading "\b" is a JSON escape (backspace), so a \boxed
    # emitted inside a JSON string arrives mangled as "\x08oxed{...}". Match the
    # "oxed{...}" tail, which is shared by both the intact and mangled forms.
    m = re.search(r"oxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    # 2. "Answer: X" (letter or number)
    m = re.search(r"[Aa]nswer[:\s]+([A-D\d]+)", text)
    if m:
        return m.group(1).strip()
    # 3. last integer in the text
    nums = re.findall(r"\b\d+\b", text)
    if nums:
        return nums[-1]
    return ""


def score_math(pred: str, gold: str) -> bool:
    if not pred:
        return False
    try:
        from math_verify import verify, parse  # type: ignore
        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        pass
    # Fallback: integer comparison (AIME answers are integers 0–999)
    try:
        return int(pred.strip()) == int(gold.strip())
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


def score_mc(pred: str, gold_letter: str) -> bool:
    m = re.search(r"[Aa]nswer[:\s]+([A-Da-d])", pred)
    if m:
        return m.group(1).upper() == gold_letter.upper()
    # Fallback: lone trailing capital A–D
    m = re.search(r"\b([A-D])\b\s*$", pred.strip())
    if m:
        return m.group(1).upper() == gold_letter.upper()
    return False


def score(task: "Task", pred: str) -> bool:
    if task.kind == "math":
        return score_math(extract_final_answer(pred), task.gold)
    return score_mc(pred, task.gold)
