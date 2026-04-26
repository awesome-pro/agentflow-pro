import json
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
}"""


class Planner:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def plan(self, query: str, memory_context: str) -> PlannerOutput:
        user_msg = (
            f"Question: {query}\n\n"
            f"Steps taken so far:\n{memory_context}\n\n"
            "What is the next best action?"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_markdown(raw)
        data = json.loads(raw)
        # Normalise action to valid enum value
        data["action"] = data["action"].strip().lower()
        return PlannerOutput(**data)


def _strip_markdown(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner)
    return text
