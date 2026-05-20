import httpx

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """Minimal client for Ollama's native /api/chat endpoint.

    We deliberately use the native endpoint instead of the OpenAI-compatible
    /v1 one: only the native endpoint honors `think: false`, which disables
    Qwen3's thinking mode. On the /v1 endpoint Qwen3 keeps thinking regardless
    of the request — it spends the whole token budget on reasoning and returns
    empty content, roughly 11x slower per call.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = "qwen3:8b",
        think: bool = False,
        timeout: float = 180.0,
    ):
        # Tolerate a /v1 suffix so callers can pass the OpenAI-style base URL.
        self._base_url = base_url.rstrip("/").removesuffix("/v1")
        self._model = model
        self._think = think
        self._http = httpx.Client(timeout=timeout)

    def complete(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: str | dict = "json",
    ) -> str:
        """`response_format` is passed straight to Ollama's `format` field:
        "json" for free-form JSON, or a JSON Schema dict to grammar-constrain
        the output to an exact shape — which guarantees every required field
        is present, eliminating missing-field validation errors."""
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": self._think,
            "format": response_format,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = self._http.post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}).get("content") or "").strip()
