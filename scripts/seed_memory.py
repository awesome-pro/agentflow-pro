"""Warm the episodic store from existing eval reports (Phase 5).

Ingests past `runs/eval_*.json` trajectories into the Qdrant episodic memory so
that a subsequent `--memory` run has real, relevant hints to retrieve.

LEAKAGE GUARD: pass `--exclude-benchmark aime24` (etc.) when seeding for an
A/B study — you must NOT seed the store with the very problems you're about to
evaluate on. Seeding AIME24 from the AIME *train* split is safe (disjoint, Year
≤ 2023, de-duped vs the 2024 test set).

Idempotent: each episode is keyed by a hash of (benchmark, question), so
re-running over the same reports overwrites rather than duplicates.

Usage:
    uv run python -m scripts.seed_memory runs/eval_aime_train_*.json \
        --exclude-benchmark aime24
"""

import glob
import json
import uuid

import typer

app = typer.Typer(help="Seed the episodic store from eval reports")

# A fixed namespace so episode ids are stable across runs (deterministic).
_NS = uuid.UUID("a9e1f3c2-5e5d-4c3b-8a00-000000000001")


@app.command()
def main(
    reports: list[str] = typer.Argument(..., help="Report JSON paths or globs (runs/eval_*.json)"),
    path: str = typer.Option("artifacts/qdrant", "--path", help="On-disk Qdrant store location"),
    successful_only: bool = typer.Option(
        True, "--successful-only/--all",
        help="Seed only solves that scored correct (default) — a wrong approach is a bad hint.",
    ),
    exclude_benchmark: list[str] = typer.Option(
        [], "--exclude-benchmark",
        help="Skip episodes from these benchmarks (use to avoid seeding your eval set).",
    ),
    reset: bool = typer.Option(False, "--reset", help="Drop the collection before seeding."),
):
    from core.episodic import EpisodicMemory

    paths: list[str] = []
    for pattern in reports:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        typer.echo("No report files matched.")
        raise typer.Exit(1)

    mem = EpisodicMemory(path=path)
    if reset:
        mem.reset()

    seeded = skipped = 0
    for fp in paths:
        report = json.load(open(fp))
        bench = report.get("benchmark")
        if bench in exclude_benchmark:
            typer.echo(f"  skip {fp} (benchmark={bench} excluded)")
            continue
        for r in report.get("results", []):
            if successful_only and not r.get("correct"):
                skipped += 1
                continue
            q = r["question"]
            eid = str(uuid.uuid5(_NS, f"{bench}::{q}"))
            mem.add_episode(
                query=q,
                answer=r.get("predicted", ""),
                trajectory=r.get("trajectory", []),
                success=r.get("correct"),
                benchmark=bench,
                episode_id=eid,
            )
            seeded += 1

    typer.echo(f"Seeded {seeded} episodes (skipped {skipped} unsuccessful). Store now holds {mem.count()}.")


if __name__ == "__main__":
    app()
