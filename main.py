import typer
from dotenv import load_dotenv
from rich.console import Console
from core.solver import Solver

load_dotenv()

app = typer.Typer(help="AgentFlow-Pro — trainable multi-agent reasoning framework")
console = Console()


@app.command()
def solve(
    query: str = typer.Argument(..., help="The question or task to solve"),
    model: str = typer.Option("qwen3:8b", "--model", "-m", help="Ollama model name"),
    max_steps: int = typer.Option(6, "--max-steps", "-s", help="Maximum solver steps"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url", help="Ollama API base URL"),
    think: bool = typer.Option(False, "--think", help="Enable Qwen thinking tokens; slower, default off"),
    memory: bool = typer.Option(
        False, "--memory",
        help="Use cross-episode (Qdrant) memory: inject hints from similar past solves and store this one. Needs `uv sync --extra memory`.",
    ),
):
    episodic = None
    if memory:
        from core.episodic import EpisodicMemory
        episodic = EpisodicMemory()

    solver = Solver(model=model, base_url=base_url, max_steps=max_steps, think=think, episodic=episodic)
    result = solver.solve(query)
    console.print(f"\n[bold]Answer:[/bold] {result.answer}")
    console.print(f"[bold]Steps taken:[/bold] {result.steps_taken}")

    # Record this solve so future runs can learn from it. We don't know
    # correctness here (no gold answer), so success stays None.
    if episodic is not None:
        episodic.add_episode(query, result.answer, result.trajectory, success=None)


if __name__ == "__main__":
    app()
