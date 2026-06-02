"""Episodic-memory tests (Phase 5). CPU-only — no Ollama, no GPU.

Run with: uv run --extra memory --extra dev pytest tests/test_episodic.py
"""

import uuid

import pytest

# Skip the whole module cleanly if the `memory` extra isn't installed.
pytest.importorskip("qdrant_client")
pytest.importorskip("sentence_transformers")

from core.episodic import EpisodicMemory, _approach
from core.types import MemoryEntry


@pytest.fixture
def mem(tmp_path):
    return EpisodicMemory(path=str(tmp_path / "qd"))


def test_approach_from_dicts_and_models():
    assert _approach([{"action": "think"}, {"action": "code"}]) == "think → code → answer"
    entry = MemoryEntry(step=1, thought="t", action="search", action_input="x", result="r")
    assert _approach([entry]) == "search → answer"
    assert _approach([]) == "answer"  # direct-answer solves still summarise


def test_retrieve_finds_similar_and_excludes_failures(mem):
    mem.add_episode("largest even integer not a sum of two odd composites?",
                    "38", [{"action": "code"}], success=True, benchmark="aime_train")
    mem.add_episode("a totally unrelated cooking question about pasta",
                    "boil", [{"action": "think"}], success=False, benchmark="aime_train")
    hints = mem.to_hints("what is the largest even number not expressible as two odd composite numbers?")
    assert "Approach that worked" in hints
    assert "pasta" not in hints  # failed solve must never surface


def test_min_score_filters_unrelated(tmp_path):
    m = EpisodicMemory(path=str(tmp_path / "qd"), min_score=0.6)
    m.add_episode("quantum chromodynamics confinement energy scale",
                  "x", [{"action": "think"}], success=True)
    assert m.to_hints("what time is the next train to Boston?") == ""


def test_episode_id_is_idempotent(mem):
    eid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "q1"))
    mem.add_episode("q1", "a", [{"action": "think"}], success=True, episode_id=eid)
    mem.add_episode("q1", "a", [{"action": "think"}], success=True, episode_id=eid)
    assert mem.count() == 1


def test_reset_clears_store(mem):
    mem.add_episode("q", "a", [{"action": "think"}], success=True)
    assert mem.count() == 1
    mem.reset()
    assert mem.count() == 0
