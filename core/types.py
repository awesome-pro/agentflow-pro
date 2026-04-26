from enum import Enum
from pydantic import BaseModel


class Action(str, Enum):
    THINK = "think"
    SEARCH = "search"
    CODE = "code"
    ANSWER = "answer"


class PlannerOutput(BaseModel):
    thought: str
    action: Action
    action_input: str


class ExecutorOutput(BaseModel):
    tool: str
    result: str
    success: bool


class VerifierOutput(BaseModel):
    sufficient: bool
    reason: str
    answer: str | None = None


class MemoryEntry(BaseModel):
    step: int
    thought: str
    action: str
    action_input: str
    result: str


class SolverResult(BaseModel):
    answer: str
    steps_taken: int
    trajectory: list[MemoryEntry]
