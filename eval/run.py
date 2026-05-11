import typer
from dotenv import load_dotenv
from core.solver import Solver
from eval.datasets import load_aime24, load_gpqa_diamond
from eval.runner import run_eval

load_dotenv()

app = typer.Typer(help="AgentFlow-Pro — eval harness")


@app.command()
def main(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="aime24 or gpqa"),
    model: str = typer.Option("qwen3.5:4b", "--model", "-m"),
    base_url: str = typer.Option("http://localhost:11434/v1", "--base-url"),
    max_steps: int = typer.Option(8, "--max-steps", "-s"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="Evaluate only the first N tasks"),
    temperature: float = typer.Option(0.0, "--temperature", "-t"),
):
    if benchmark == "aime24":
        tasks = load_aime24()
    elif benchmark == "gpqa":
        tasks = load_gpqa_diamond()
    else:
        typer.echo(f"Unknown benchmark: {benchmark!r}. Choose 'aime24' or 'gpqa'.")
        raise typer.Exit(1)

    solver = Solver(model=model, base_url=base_url, max_steps=max_steps, temperature=temperature, verbose=False)
    run_eval(solver, tasks, benchmark=benchmark, limit=limit)


if __name__ == "__main__":
    app()
