from .types import Action, PlannerOutput, ExecutorOutput


class Executor:
    """
    Runs the action chosen by the Planner.
    Step 1: stubs for search and code — replaced with real MCP tools in Step 2.
    """

    def execute(self, plan: PlannerOutput) -> ExecutorOutput:
        match plan.action:
            case Action.THINK:
                return ExecutorOutput(tool="think", result=plan.action_input, success=True)
            case Action.ANSWER:
                return ExecutorOutput(tool="answer", result=plan.action_input, success=True)
            case Action.SEARCH:
                return self._search(plan.action_input)
            case Action.CODE:
                return self._code(plan.action_input)
            case _:
                return ExecutorOutput(tool="unknown", result="Unrecognised action.", success=False)

    def _search(self, query: str) -> ExecutorOutput:
        # Stub — replaced by DuckDuckGo/MCP tool in Step 2
        return ExecutorOutput(
            tool="search",
            result=f"[search stub] No live search yet. Query was: {query}",
            success=True,
        )

    def _code(self, code: str) -> ExecutorOutput:
        # Stub — replaced by sandboxed Python executor in Step 2
        return ExecutorOutput(
            tool="code",
            result=f"[code stub] No execution yet. Code was:\n{code}",
            success=True,
        )
