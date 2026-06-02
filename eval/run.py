import typer
from dotenv import load_dotenv
from core.solver import Solver
from eval.datasets import load_aime24, load_aime_train, load_gpqa_diamond
from eval.runner import run_eval

load_dotenv()

app = typer.Typer(help="AgentFlow-Pro — eval harness")


@app.command()
def main(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="aime24, gpqa, or aime_train"),
    model: str = typer.Option("qwen3:8b", "--model", "-m"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
    max_steps: int = typer.Option(8, "--max-steps", "-s"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Evaluate only the first N tasks"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
    think: bool = typer.Option(False, "--think", help="Enable Qwen thinking tokens; slower, default off"),
    task_timeout: float = typer.Option(
        300.0, "--task-timeout",
        help="Per-problem wall-clock budget (s); abandons a hung problem and moves on. 0 disables.",
    ),
    memory: bool = typer.Option(
        False, "--memory",
        help="Use cross-episode (Qdrant) memory: inject hints from similar past solves and store each scored solve. Needs `uv sync --extra memory`.",
    ),
    memory_path: str = typer.Option(
        "artifacts/qdrant", "--memory-path",
        help="On-disk location of the Qdrant episodic store (only used with --memory).",
    ),
    memory_readonly: bool = typer.Option(
        True, "--memory-readonly/--memory-record",
        help="Read-only: retrieve hints but don't write eval solves back (keeps an A/B's "
             "seeded store fixed and the experiment reproducible). Default on.",
    ),
):
    if benchmark == "aime24":
        tasks = load_aime24()
    elif benchmark == "gpqa":
        tasks = load_gpqa_diamond()
    elif benchmark == "aime_train":
        tasks = load_aime_train()
    else:
        typer.echo(f"Unknown benchmark: {benchmark!r}. Choose 'aime24', 'gpqa', or 'aime_train'.")
        raise typer.Exit(1)

    episodic = None
    if memory:
        from core.episodic import EpisodicMemory
        episodic = EpisodicMemory(path=memory_path)
        typer.echo(f"Episodic memory ON ({memory_path}, {episodic.count()} episodes stored).")

    solver = Solver(
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        temperature=temperature,
        verbose=False,
        think=think,
        episodic=episodic,
    )
    run_eval(
        solver, tasks, benchmark=benchmark, limit=limit, task_timeout_s=task_timeout,
        record_episodes=(episodic is not None and not memory_readonly),
    )


if __name__ == "__main__":
    app()
