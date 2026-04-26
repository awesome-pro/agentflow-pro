import json
from openai import OpenAI
from .types import VerifierOutput
from .planner import _strip_markdown

_SYSTEM = """\
You are a verifier. Given a question and the agent's work so far, decide if there is enough \
information to give a correct, complete answer.

Be strict: only mark sufficient=true if the answer is directly supported by the results. \
Do not guess.

Respond with valid JSON only — no markdown, no extra text:
{
  "sufficient": true | false,
  "reason": "one sentence explaining your decision",
  "answer": "the final answer if sufficient, otherwise null"
}"""


class Verifier:
    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def verify(self, query: str, memory_context: str, last_result: str) -> VerifierOutput:
        user_msg = (
            f"Question: {query}\n\n"
            f"Work done so far:\n{memory_context}\n\n"
            f"Latest result: {last_result}\n\n"
            "Is there enough information to answer the question correctly?"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        raw = _strip_markdown(raw)
        data = json.loads(raw)
        return VerifierOutput(**data)
