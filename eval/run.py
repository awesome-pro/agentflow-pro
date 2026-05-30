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

    solver = Solver(
        model=model,
        base_url=base_url,
        max_steps=max_steps,
        temperature=temperature,
        verbose=False,
        think=think,
    )
    run_eval(solver, tasks, benchmark=benchmark, limit=limit, task_timeout_s=task_timeout)


if __name__ == "__main__":
    app()
