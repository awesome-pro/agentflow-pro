from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .memory import Memory
from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .types import Action, SolverResult

console = Console()

_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL = "qwen3.5:4b"


class Solver:
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _OLLAMA_BASE_URL,
        max_steps: int = 6,
        temperature: float = 0.7,
        verbose: bool = True,
    ):
        client = OpenAI(base_url=base_url, api_key="ollama", timeout=120.0)
        self.planner = Planner(client=client, model=model, temperature=temperature)
        self.executor = Executor()
        self.verifier = Verifier(client=client, model=model, temperature=temperature)
        self.memory = Memory()
        self.max_steps = max_steps
        self.verbose = verbose

    def solve(self, query: str) -> SolverResult:
        self.memory.clear()
        if self.verbose:
            console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", expand=False))

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                console.rule(f"[yellow]Step {step} / {self.max_steps}[/yellow]")

            # --- Planner ---
            if self.verbose:
                with console.status("[dim]Planner thinking…[/dim]", spinner="dots"):
                    plan = self.planner.plan(query, self.memory.to_context())
            else:
                plan = self.planner.plan(query, self.memory.to_context())
            if self.verbose:
                console.print(f"[green]Thought:[/green] {plan.thought}")
                console.print(f"[green]Action:[/green]  {plan.action.value}  →  {plan.action_input[:120]}")

            # Planner can signal a direct answer without calling executor/verifier
            if plan.action == Action.ANSWER:
                if self.verbose:
                    console.print(Panel(
                        f"[bold green]Answer:[/bold green] {plan.action_input}",
                        title="Done", expand=False,
                    ))
                return SolverResult(
                    answer=plan.action_input,
                    steps_taken=step,
                    trajectory=self.memory.entries,
                )

            # --- Executor ---
            result = self.executor.execute(plan)
            if self.verbose:
                console.print(f"[cyan]Result:[/cyan]  {result.result[:200]}")
            self.memory.add(step, plan, result)

            # --- Verifier ---
            if self.verbose:
                with console.status("[dim]Verifier checking…[/dim]", spinner="dots"):
                    verdict = self.verifier.verify(query, self.memory.to_context(), result.result)
            else:
                verdict = self.verifier.verify(query, self.memory.to_context(), result.result)
            if self.verbose:
                icon = "[bold green]✓[/bold green]" if verdict.sufficient else "[bold red]✗[/bold red]"
                console.print(f"{icon} [magenta]Sufficient:[/magenta] {verdict.reason}")

            if verdict.sufficient and verdict.answer:
                if self.verbose:
                    console.print(Panel(
                        f"[bold green]Answer:[/bold green] {verdict.answer}",
                        title="Done", expand=False,
                    ))
                return SolverResult(
                    answer=verdict.answer,
                    steps_taken=step,
                    trajectory=self.memory.entries,
                )

        fallback = "Max steps reached without a confident answer."
        if self.verbose:
            console.print(Panel(f"[bold red]{fallback}[/bold red]", title="Stopped", expand=False))
        return SolverResult(answer=fallback, steps_taken=self.max_steps, trajectory=self.memory.entries)

    def _print_trajectory(self, result: SolverResult) -> None:
        table = Table(title="Trajectory", show_lines=True)
        table.add_column("Step", style="yellow", width=4)
        table.add_column("Action", style="green", width=8)
        table.add_column("Input", width=40)
        table.add_column("Result", width=50)
        for e in result.trajectory:
            table.add_row(str(e.step), e.action, e.action_input[:40], e.result[:50])
        console.print(table)
