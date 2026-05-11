import json
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from core.types import MemoryEntry
from .datasets import Task
from .scorer import score

if TYPE_CHECKING:
    from core.solver import Solver

console = Console()


class EvalResult(BaseModel):
    id: str
    question: str
    predicted: str
    gold: str
    correct: bool
    steps_taken: int
    trajectory: list[MemoryEntry]
    error: str | None = None


class EvalReport(BaseModel):
    benchmark: str
    model: str
    max_steps: int
    temperature: float
    n: int
    accuracy: float
    results: list[EvalResult]
    timestamp: str


def run_eval(
    solver: "Solver",
    tasks: list[Task],
    benchmark: str,
    limit: int | None = None,
) -> EvalReport:
    subset = tasks[:limit] if limit is not None else tasks
    results: list[EvalResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task(f"[cyan]{benchmark}[/cyan]", total=len(subset))
        for t in subset:
            progress.update(bar, description=f"[cyan]{t.id}[/cyan]")
            try:
                r = solver.solve(t.question)
                correct = score(t, r.answer)
                results.append(EvalResult(
                    id=t.id,
                    question=t.question,
                    predicted=r.answer,
                    gold=t.gold,
                    correct=correct,
                    steps_taken=r.steps_taken,
                    trajectory=r.trajectory,
                ))
            except Exception as e:
                results.append(EvalResult(
                    id=t.id,
                    question=t.question,
                    predicted="",
                    gold=t.gold,
                    correct=False,
                    steps_taken=0,
                    trajectory=[],
                    error=str(e),
                ))
            progress.advance(bar)

    accuracy = sum(r.correct for r in results) / len(results) if results else 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = EvalReport(
        benchmark=benchmark,
        model=solver.planner._model,
        max_steps=solver.max_steps,
        temperature=solver.planner._temperature,
        n=len(results),
        accuracy=accuracy,
        results=results,
        timestamp=timestamp,
    )

    _save_report(report, benchmark, timestamp)
    _print_summary(report)
    return report


def _save_report(report: EvalReport, benchmark: str, timestamp: str) -> None:
    os.makedirs("runs", exist_ok=True)
    path = f"runs/eval_{benchmark}_{timestamp}.json"
    with open(path, "w") as f:
        f.write(report.model_dump_json(indent=2))
    console.print(f"\n[dim]Report saved → {path}[/dim]")


def _print_summary(report: EvalReport) -> None:
    table = Table(title=f"{report.benchmark} — {report.model}", show_lines=True)
    table.add_column("ID", style="yellow", width=12)
    table.add_column("✓", width=3)
    table.add_column("Steps", width=5)
    table.add_column("Predicted", width=30)
    table.add_column("Gold", width=10)

    for r in report.results:
        mark = "[green]✓[/green]" if r.correct else "[red]✗[/red]"
        err = f"[red]ERR: {r.error[:25]}[/red]" if r.error else r.predicted[:30]
        table.add_row(r.id, mark, str(r.steps_taken), err, r.gold)

    console.print(table)
    pct = f"{report.accuracy:.1%}"
    console.print(f"\n[bold]Accuracy: {pct}[/bold] ({sum(r.correct for r in report.results)}/{report.n})")
