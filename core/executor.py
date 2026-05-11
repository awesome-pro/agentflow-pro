from .types import Action, PlannerOutput, ExecutorOutput
from tools.builtin.search import web_search
from tools.builtin.python_exec import run_python


class Executor:
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
        try:
            result = web_search(query)
            return ExecutorOutput(tool="search", result=result, success=True)
        except Exception as e:
            return ExecutorOutput(tool="search", result=f"Search failed: {e}", success=False)

    def _code(self, code: str) -> ExecutorOutput:
        result = run_python(code)
        return ExecutorOutput(tool="code", result=result, success=True)
