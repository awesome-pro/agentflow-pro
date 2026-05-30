import json
import os
import signal
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from core.types import MemoryEntry
from .datasets import Task
from .scorer import score

if TYPE_CHECKING:
    from core.solver import Solver

console = Console()


class _TaskTimeout(BaseException):
    """Raised when a single task exceeds its wall-clock budget.

    Deliberately a BaseException, not Exception: the broad `except Exception`
    blocks inside the tools (e.g. `python_exec.run_python`), planner, and
    verifier must NOT swallow it — it has to propagate up to the per-task
    handler in `run_eval` so the hung problem is abandoned, not silently
    turned into a tool-level "Error: ..." string that lets the loop crawl on.
    """


@contextmanager
def _time_limit(seconds: float):
    """Abort the wrapped block after `seconds` via SIGALRM.

    Catches the dominant hang on this stack: a small model emitting a `code`
    action with an infinite / brute-force loop, which `run_python`'s bare
    `exec()` would otherwise run forever. A hang buried in a C extension
    (numpy/sympy) may not interrupt until the C call returns — acceptable; the
    common pure-Python loop is interrupted at the bytecode boundary.

    SIGALRM is POSIX-only (Linux pod + macOS dev box) and must be armed on the
    main thread, which is where `run_eval` runs. On platforms without SIGALRM,
    or when `seconds <= 0`, this is a no-op.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise _TaskTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


class EvalResult(BaseModel):
    id: str
    question: str
    predicted: str
    gold: str
    correct: bool
    steps_taken: int
    trajectory: list[MemoryEntry]
    elapsed_seconds: float = 0.0
    error: str | None = None


class EvalReport(BaseModel):
    benchmark: str
    model: str
    max_steps: int
    temperature: float
    think: bool
    n: int
    accuracy: float
    avg_steps: float
    avg_elapsed_seconds: float
    results: list[EvalResult]
    timestamp: str


def run_eval(
    solver: "Solver",
    tasks: list[Task],
    benchmark: str,
    limit: int | None = None,
    task_timeout_s: float = 300.0,
) -> EvalReport:
    subset = tasks[:limit] if limit is not None else tasks
    results: list[EvalResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        bar = progress.add_task(f"[cyan]{benchmark}[/cyan]", total=len(subset))
        for t in subset:
            progress.update(bar, description=f"[cyan]{t.id}[/cyan]")
            progress.console.print(f"[dim]Starting {t.id}...[/dim]")
            started = perf_counter()

            def on_step(step: int, max_steps: int, _id: str = t.id) -> None:
                progress.update(bar, description=f"[cyan]{_id}[/cyan] step {step}/{max_steps}")

            try:
                with _time_limit(task_timeout_s):
                    r = solver.solve(t.question, on_step=on_step)
                    correct = score(t, r.answer)
                elapsed = perf_counter() - started
                result = EvalResult(
                    id=t.id,
                    question=t.question,
                    predicted=r.answer,
                    gold=t.gold,
                    correct=correct,
                    steps_taken=r.steps_taken,
                    trajectory=r.trajectory,
                    elapsed_seconds=elapsed,
                )
            except (_TaskTimeout, Exception) as e:
                elapsed = perf_counter() - started
                # Salvage whatever the solver completed before the hang — the
                # steps already in memory are still useful PRM/DAPO data; only
                # the offending (hung) step is missing.
                partial = list(getattr(solver.memory, "entries", []))
                err = (
                    f"task timed out after {task_timeout_s:.0f}s"
                    if isinstance(e, _TaskTimeout)
                    else str(e)
                )
                result = EvalResult(
                    id=t.id,
                    question=t.question,
                    predicted="",
                    gold=t.gold,
                    correct=False,
                    steps_taken=len(partial),
                    trajectory=partial,
                    elapsed_seconds=elapsed,
                    error=err,
                )
            results.append(result)
            mark = "[green]✓[/green]" if result.correct else "[red]✗[/red]"
            progress.console.print(
                f"{mark} {t.id}: steps={result.steps_taken}, "
                f"time={_format_elapsed(result.elapsed_seconds)}"
            )
            progress.advance(bar)

    accuracy = sum(r.correct for r in results) / len(results) if results else 0.0
    avg_steps = sum(r.steps_taken for r in results) / len(results) if results else 0.0
    avg_elapsed = sum(r.elapsed_seconds for r in results) / len(results) if results else 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = EvalReport(
        benchmark=benchmark,
        model=solver.model,
        max_steps=solver.max_steps,
        temperature=solver.planner._temperature,
        think=solver.think,
        n=len(results),
        accuracy=accuracy,
        avg_steps=avg_steps,
        avg_elapsed_seconds=avg_elapsed,
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
    table.add_column("Time", width=8)
    table.add_column("Predicted", width=30)
    table.add_column("Gold", width=10)

    for r in report.results:
        mark = "[green]✓[/green]" if r.correct else "[red]✗[/red]"
        err = f"[red]ERR: {r.error[:25]}[/red]" if r.error else r.predicted[:30]
        table.add_row(r.id, mark, str(r.steps_taken), _format_elapsed(r.elapsed_seconds), err, r.gold)

    console.print(table)
    pct = f"{report.accuracy:.1%}"
    console.print(f"\n[bold]Accuracy: {pct}[/bold] ({sum(r.correct for r in report.results)}/{report.n})")
    console.print(
        f"[bold]Avg steps:[/bold] {report.avg_steps:.2f} · "
        f"[bold]Avg time:[/bold] {_format_elapsed(report.avg_elapsed_seconds)}"
    )


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(int(seconds), 60)
    return f"{minutes}m {rem:02d}s"
