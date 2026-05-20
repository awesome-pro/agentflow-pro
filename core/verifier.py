import json
from .llm import OllamaClient
from .types import VerifierOutput
from .planner import _strip_markdown, _strip_think_tags, _RETRY_MSG

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

_SCHEMA = VerifierOutput.model_json_schema()


class Verifier:
    def __init__(self, client: OllamaClient, temperature: float = 0.2):
        self._client = client
        self._temperature = temperature

    def verify(self, query: str, memory_context: str, last_result: str) -> VerifierOutput:
        user_msg = (
            f"Question: {query}\n\n"
            f"Work done so far:\n{memory_context}\n\n"
            f"Latest result: {last_result}\n\n"
            "Is there enough information to answer the question correctly?"
        )
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        for _ in range(2):
            raw = ""
            try:
                raw = self._complete(messages)
                return VerifierOutput(**json.loads(raw))
            except Exception:  # JSONDecodeError, ValidationError, TypeError, ...
                messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": _RETRY_MSG}]
        # Couldn't parse the verifier — assume "not sufficient" so the solver keeps going.
        return VerifierOutput(sufficient=False, reason="Verifier output could not be parsed; continuing.", answer=None)

    def _complete(self, messages: list[dict]) -> str:
        raw = self._client.complete(
            messages, temperature=self._temperature, max_tokens=256, response_format=_SCHEMA,
        )
        return _strip_markdown(_strip_think_tags(raw))
