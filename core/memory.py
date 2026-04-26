from .types import MemoryEntry, PlannerOutput, ExecutorOutput


class Memory:
    def __init__(self):
        self._entries: list[MemoryEntry] = []

    def add(self, step: int, plan: PlannerOutput, result: ExecutorOutput) -> None:
        self._entries.append(MemoryEntry(
            step=step,
            thought=plan.thought,
            action=plan.action.value,
            action_input=plan.action_input,
            result=result.result,
        ))

    def to_context(self) -> str:
        if not self._entries:
            return "No previous steps."
        lines = []
        for e in self._entries:
            lines.append(f"Step {e.step}:")
            lines.append(f"  Thought: {e.thought}")
            lines.append(f"  Action:  {e.action}({e.action_input})")
            lines.append(f"  Result:  {e.result}")
        return "\n".join(lines)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()
