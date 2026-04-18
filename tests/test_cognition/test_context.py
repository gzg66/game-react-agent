"""Tests for the sliding window context manager."""

import time

from game_agent.cognition.context import ContextManager, ReActStep


def _make_step(num: int, observation: str = "ok") -> ReActStep:
    return ReActStep(
        step_number=num,
        thought=f"Thinking step {num}",
        action_name="poco_click",
        action_params={"poco_path": f"btn_{num}"},
        observation=observation,
        timestamp=time.time(),
    )


def test_empty_context():
    ctx = ContextManager(window_size=5)
    assert len(ctx) == 0
    assert "尚未执行" in ctx.to_prompt_text()


def test_add_and_retrieve():
    ctx = ContextManager(window_size=5)
    ctx.add_step(_make_step(0))
    ctx.add_step(_make_step(1))
    assert len(ctx) == 2
    text = ctx.to_prompt_text()
    assert "步骤 0" in text
    assert "步骤 1" in text


def test_sliding_window_eviction():
    ctx = ContextManager(window_size=3)
    for i in range(5):
        ctx.add_step(_make_step(i))
    assert len(ctx) == 3
    text = ctx.to_prompt_text()
    assert "步骤 0" not in text
    assert "步骤 1" not in text
    assert "步骤 2" in text
    assert "步骤 4" in text


def test_last_n():
    ctx = ContextManager(window_size=10)
    for i in range(5):
        ctx.add_step(_make_step(i))
    last = ctx.last_n(2)
    assert len(last) == 2
    assert last[0].step_number == 3
    assert last[1].step_number == 4


def test_consecutive_failures():
    ctx = ContextManager(window_size=10)
    ctx.add_step(_make_step(0, "ok"))
    ctx.add_step(_make_step(1, "failed to click"))
    ctx.add_step(_make_step(2, "node not found"))
    assert ctx.consecutive_failures() == 2


def test_consecutive_failures_resets():
    ctx = ContextManager(window_size=10)
    ctx.add_step(_make_step(0, "failed"))
    ctx.add_step(_make_step(1, "success"))
    ctx.add_step(_make_step(2, "failed"))
    assert ctx.consecutive_failures() == 1


def test_clear():
    ctx = ContextManager(window_size=5)
    ctx.add_step(_make_step(0))
    ctx.clear()
    assert len(ctx) == 0
