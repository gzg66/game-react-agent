"""Sliding window context manager for ReAct step history."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class ReActStep:
    """One complete Thought -> Action -> Observation cycle."""

    step_number: int
    thought: str
    action_name: str
    action_params: dict
    observation: str
    timestamp: float


class ContextManager:
    """Maintains a sliding window of the last N ReAct steps.

    Serializes to a text block suitable for LLM prompt injection.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._steps: deque[ReActStep] = deque(maxlen=window_size)

    def add_step(self, step: ReActStep) -> None:
        self._steps.append(step)

    def to_prompt_text(self) -> str:
        if not self._steps:
            return "（尚未执行任何操作）"

        lines = []
        for s in self._steps:
            params_str = ", ".join(f"{k}={v!r}" for k, v in s.action_params.items())
            lines.append(
                f"步骤 {s.step_number}：\n"
                f"  思考：{s.thought}\n"
                f"  动作：{s.action_name}({params_str})\n"
                f"  观察：{s.observation}"
            )
        return "\n\n".join(lines)

    def last_n(self, n: int) -> list[ReActStep]:
        return list(self._steps)[-n:]

    def consecutive_failures(self) -> int:
        count = 0
        for step in reversed(self._steps):
            observation = step.observation.lower()
            if (
                "failed" in observation
                or "not found" in observation
                or "失败" in step.observation
                or "未找到" in step.observation
                or "超时" in step.observation
            ):
                count += 1
            else:
                break
        return count

    def clear(self) -> None:
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)
