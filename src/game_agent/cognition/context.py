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
    page_hash: str = ""


class ContextManager:
    """Maintains a sliding window of the last N ReAct steps.

    Evicted steps are compressed into a running summary so the LLM
    retains awareness of early actions without consuming too many tokens.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._steps: deque[ReActStep] = deque(maxlen=window_size)
        self._evicted_summary: list[str] = []
        self._total_added: int = 0

    def add_step(self, step: ReActStep) -> None:
        if len(self._steps) == self._steps.maxlen:
            evicted = self._steps[0]
            self._evicted_summary.append(self._summarize_step(evicted))
        self._steps.append(step)
        self._total_added += 1

    @staticmethod
    def _summarize_step(step: ReActStep) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in step.action_params.items())
        return f"步骤{step.step_number}：{step.action_name}({params_str}) → {step.observation[:60]}"

    def to_prompt_text(self) -> str:
        if not self._steps and not self._evicted_summary:
            return "（尚未执行任何操作）"

        parts = []

        if self._evicted_summary:
            parts.append("### 早期操作摘要（已压缩）")
            parts.append("\n".join(self._evicted_summary[-20:]))
            parts.append("")

        if self._steps:
            parts.append("### 最近操作详情")
            for s in self._steps:
                params_str = ", ".join(f"{k}={v!r}" for k, v in s.action_params.items())
                parts.append(
                    f"步骤 {s.step_number}：\n"
                    f"  思考：{s.thought}\n"
                    f"  动作：{s.action_name}({params_str})\n"
                    f"  观察：{s.observation}"
                )

        return "\n\n".join(parts)

    def last_n(self, n: int) -> list[ReActStep]:
        return list(self._steps)[-n:]

    def page_stale_count(self) -> int:
        """Count consecutive trailing steps on the same page hash."""
        if len(self._steps) < 2:
            return 0
        latest_hash = self._steps[-1].page_hash
        if not latest_hash:
            return 0
        count = 0
        for step in reversed(self._steps):
            if step.page_hash == latest_hash:
                count += 1
            else:
                break
        return count

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

    def cycle_detected(self, min_cycle_length: int = 2, min_repetitions: int = 3) -> bool:
        """Detect if recent page_hash sequence forms a repeating cycle (e.g. A→B→A→B→A→B)."""
        hashes = [s.page_hash for s in self._steps if s.page_hash]
        for period in range(min_cycle_length, min_cycle_length + 2):
            required = period * min_repetitions
            if len(hashes) < required:
                continue
            tail = hashes[-required:]
            pattern = tail[:period]
            if tail == pattern * min_repetitions:
                return True
        return False

    def cycle_description(self) -> str:
        """Human-readable description of the detected cycle for logging/prompt injection."""
        hashes = [s.page_hash for s in self._steps if s.page_hash]
        if len(hashes) < 4:
            return ""
        tail = hashes[-6:] if len(hashes) >= 6 else hashes
        unique = list(dict.fromkeys(tail))
        pages = " → ".join(h[:8] for h in unique)
        return f"检测到导航死循环：{pages}"

    def clear(self) -> None:
        self._steps.clear()
        self._evicted_summary.clear()
        self._total_added = 0

    def __len__(self) -> int:
        return len(self._steps)
