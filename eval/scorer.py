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


def _mc_letter(pred: str) -> str:
    """Pull the chosen A–D option letter out of a free-form answer."""
    # 1. explicit "Answer: X" / "answer is X"
    m = re.search(r"answer\s*(?:is\s*)?[:=\-]?\s*\(?([A-Da-d])\b", pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2. leading "X)" / "X." / "(X)" — e.g. "D) 10^-4 eV"
    m = re.match(r"\s*\(?([A-Da-d])[)\.:]", pred)
    if m:
        return m.group(1).upper()
    # 3. the whole answer is a single letter
    s = pred.strip()
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    # 4. exactly one distinct A–D token anywhere (last resort)
    letters = {x.upper() for x in re.findall(r"\b([A-Da-d])\b", pred)}
    return letters.pop() if len(letters) == 1 else ""


def score_mc(pred: str, gold_letter: str) -> bool:
    letter = _mc_letter(pred)
    return bool(letter) and letter == gold_letter.strip().upper()


def score(task: "Task", pred: str) -> bool:
    if task.kind == "math":
        return score_math(extract_final_answer(pred), task.gold)
    return score_mc(pred, task.gold)
