import json
import re
from openai import OpenAI
from .types import PlannerOutput, Action

_SYSTEM = """\
You are a reasoning agent. Your job is to decide the next step toward answering a question.

Available actions:
  think  — reason internally without any external tool
  search — look up information on the web
  code   — write and run Python code (calculations, data processing)
  answer — provide the final answer (only when you are confident)

Rules:
- Use "think" to break down the problem before acting.
- Use "search" when you need facts you don't know.
- Use "code" for math, counting, or anything computational.
- Use "answer" only when you have enough information to respond correctly.
- Never repeat an action with the exact same input.

Respond with valid JSON only — no markdown, no extra text:
{
  "thought": "your reasoning about what to do next",
  "action": "think | search | code | answer",
  "action_input": "the search query, code to run, or final answer text"
}

/no_think"""

_VALID_ACTIONS = {a.value for a in Action}
_RETRY_MSG = "That response was not valid. Reply with ONLY the JSON object — no markdown, no commentary."


class Planner:
    def __init__(self, client: OpenAI, model: str, temperature: float = 0.7):
        self._client = client
        self._model = model
        self._temperature = temperature

    def plan(self, query: str, memory_context: str) -> PlannerOutput:
        user_msg = (
            f"Question: {query}\n\n"
            f"Steps taken so far:\n{memory_context}\n\n"
            "What is the next best action?"
        )
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        last_err: Exception | None = None
        for _ in range(2):
            raw = self._complete(messages)
            try:
                data = json.loads(raw)
                action = str(data.get("action", "")).strip().lower()
                data["action"] = action if action in _VALID_ACTIONS else Action.THINK.value
                return PlannerOutput(**data)
            except Exception as e:  # JSONDecodeError, ValidationError, TypeError, ...
                last_err = e
                messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": _RETRY_MSG}]
        # Couldn't get valid JSON even after a retry — degrade to a 'think' step instead of crashing.
        return PlannerOutput(
            thought=f"(planner returned invalid output: {last_err})",
            action=Action.THINK,
            action_input="Reconsider the question and the steps taken so far.",
        )

    def _complete(self, messages: list[dict]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        return _strip_markdown(_strip_think_tags((response.choices[0].message.content or "").strip()))


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_markdown(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner)
    return text
