"""Tests for UI diff calculation."""

from game_agent.config import PerceptionConfig
from game_agent.perception.state import PerceptionState
from game_agent.perception.ui_diff import UIDiffCalculator


def test_identical_states_are_loading():
    calc = UIDiffCalculator(PerceptionConfig(diff_threshold=0.05))
    state = PerceptionState(
        page_hash="abc123",
        poco_node_names=frozenset(["btn_a", "btn_b"]),
    )
    diff = calc.compute(state, state)
    assert diff.hash_changed is False
    assert diff.is_loading is True
    assert diff.added_nodes == []
    assert diff.removed_nodes == []


def test_different_hashes():
    calc = UIDiffCalculator(PerceptionConfig())
    prev = PerceptionState(page_hash="aaa", poco_node_names=frozenset(["btn_a"]))
    curr = PerceptionState(page_hash="bbb", poco_node_names=frozenset(["btn_b"]))
    diff = calc.compute(prev, curr)
    assert diff.hash_changed is True
    assert diff.is_loading is False


def test_added_and_removed_nodes():
    calc = UIDiffCalculator(PerceptionConfig())
    prev = PerceptionState(page_hash="aaa", poco_node_names=frozenset(["btn_a", "btn_b"]))
    curr = PerceptionState(page_hash="bbb", poco_node_names=frozenset(["btn_b", "btn_c"]))
    diff = calc.compute(prev, curr)
    assert "btn_a" in diff.removed_nodes
    assert "btn_c" in diff.added_nodes
    assert "btn_b" not in diff.added_nodes
    assert "btn_b" not in diff.removed_nodes


def test_both_screenshots_none_gives_full_similarity():
    calc = UIDiffCalculator(PerceptionConfig())
    prev = PerceptionState(page_hash="a", screenshot_bytes=None)
    curr = PerceptionState(page_hash="b", screenshot_bytes=None)
    diff = calc.compute(prev, curr)
    assert diff.structural_similarity == 1.0


def test_one_screenshot_none_gives_zero_similarity():
    calc = UIDiffCalculator(PerceptionConfig())
    prev = PerceptionState(page_hash="a", screenshot_bytes=b"img")
    curr = PerceptionState(page_hash="b", screenshot_bytes=None)
    diff = calc.compute(prev, curr)
    assert diff.structural_similarity == 0.0
