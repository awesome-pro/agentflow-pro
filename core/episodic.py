"""Episodic (cross-episode) memory — Phase 5.

The in-task `Memory` (core/memory.py) is wiped at the start of every solve. This
module adds a SEPARATE, persistent store of *past solves*: after a problem is
finished its query + the strategy that worked is embedded and saved; when a new
query arrives, the most semantically-similar past episodes are retrieved and
injected into the Planner's prompt as hints ("here's how you solved something
like this before").

Design notes:
- Two concerns, deliberately not merged into the per-task `Memory` API: the
  Solver only *reads* (retrieves hints); the caller (eval runner / main.py)
  *writes* the episode after solving, because only the caller knows whether the
  answer was actually correct, and writing after the solve guarantees a problem
  can never retrieve itself.
- Storage is a LOCAL on-disk Qdrant (`QdrantClient(path=...)`) — no server, no
  Docker. Embeddings are sentence-transformers (all-MiniLM-L6-v2, CPU, 384-d).
- Heavy imports (qdrant_client, sentence_transformers) happen inside __init__ so
  that `import core` stays cheap for the base install; episodic memory only
  loads when you actually opt in with `--memory` (needs `uv sync --extra memory`).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from .types import MemoryEntry

_DEFAULT_PATH = "artifacts/qdrant"
_DEFAULT_COLLECTION = "episodes"
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Episode:
    """A retrieved past solve."""
    query: str
    answer: str
    approach: str          # e.g. "think → code → answer"
    success: bool | None   # True/False if scored, None if unknown (live `main.py`)
    benchmark: str | None
    score: float           # cosine similarity to the current query


def _approach(trajectory: list[MemoryEntry] | list[dict]) -> str:
    """Compact one-line summary of the action chain a solve took."""
    actions: list[str] = []
    for e in trajectory:
        a = e.action if isinstance(e, MemoryEntry) else e.get("action", "")
        if a:
            actions.append(a)
    actions.append("answer")  # every solve ends by answering
    return " → ".join(actions)


class EpisodicMemory:
    def __init__(
        self,
        path: str = _DEFAULT_PATH,
        collection: str = _DEFAULT_COLLECTION,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        top_k: int = 3,
        min_score: float = 0.45,
        successful_only: bool = True,
    ):
        # Lazy, opt-in imports — see module docstring.
        from qdrant_client import QdrantClient, models
        from sentence_transformers import SentenceTransformer

        self._models = models
        self._encoder = SentenceTransformer(embed_model)
        self._dim = self._encoder.get_sentence_embedding_dimension()
        self._collection = collection
        self._top_k = top_k
        self._min_score = min_score
        self._successful_only = successful_only

        # ":memory:" (used by tests) and on-disk paths both go through path=.
        self._client = QdrantClient(path=path)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=self._models.VectorParams(
                    size=self._dim, distance=self._models.Distance.COSINE
                ),
            )

    def reset(self) -> None:
        """Drop and recreate the collection on the SAME client. (Local Qdrant
        holds an exclusive file lock, so a second client on the path would fail.)"""
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._ensure_collection()

    def _embed(self, text: str) -> list[float]:
        vec = self._encoder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def add_episode(
        self,
        query: str,
        answer: str,
        trajectory: list[MemoryEntry] | list[dict],
        success: bool | None = None,
        benchmark: str | None = None,
        episode_id: str | None = None,
    ) -> None:
        """Persist one finished solve. The query is what we embed and search on —
        we want to retrieve by 'a problem like this', not by its solution.

        Pass a stable `episode_id` (e.g. a hash of the question) to make
        re-ingesting the same solve idempotent; omit it for a fresh random id.
        """
        self._client.upsert(
            collection_name=self._collection,
            points=[
                self._models.PointStruct(
                    id=episode_id or uuid.uuid4().hex,
                    vector=self._embed(query),
                    payload={
                        "query": query,
                        "answer": answer,
                        "approach": _approach(trajectory),
                        "success": success,
                        "benchmark": benchmark,
                    },
                )
            ],
        )

    def retrieve(self, query: str) -> list[Episode]:
        """Top-k past episodes most similar to `query`, above `min_score`."""
        query_filter = None
        if self._successful_only:
            # Only surface solves that actually worked — a wrong past approach is
            # worse than no hint. Points with success=None are excluded too.
            query_filter = self._models.Filter(
                must=[self._models.FieldCondition(
                    key="success", match=self._models.MatchValue(value=True)
                )]
            )
        hits = self._client.query_points(
            collection_name=self._collection,
            query=self._embed(query),
            limit=self._top_k,
            score_threshold=self._min_score,
            query_filter=query_filter,
            with_payload=True,
        ).points
        out: list[Episode] = []
        for h in hits:
            p = h.payload or {}
            out.append(Episode(
                query=p.get("query", ""),
                answer=p.get("answer", ""),
                approach=p.get("approach", ""),
                success=p.get("success"),
                benchmark=p.get("benchmark"),
                score=h.score,
            ))
        return out

    def to_hints(self, query: str) -> str:
        """A planner-ready hints block, or "" if nothing relevant is stored.

        The answer of a *different* problem is shown only as an illustration of
        the expected output format — the block is explicitly framed as guidance,
        never as the answer to the current question.
        """
        episodes = self.retrieve(query)
        if not episodes:
            return ""
        lines = ["Hints from similar problems you solved before (guidance, NOT the answer to this question):"]
        for ep in episodes:
            q = ep.query.split("\n", 1)[0][:160]
            lines.append(f'- A similar problem: "{q}"')
            lines.append(f"    Approach that worked: {ep.approach}")
        return "\n".join(lines)

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count
