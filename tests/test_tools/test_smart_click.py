"""Integration tests for the smart_click macro tool."""

from unittest.mock import patch

from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.perception.ui_tree_store import UITreeStore
from game_agent.tools.macros import smart_click, _click_effective
from game_agent.tools.schemas import SmartClickInput


def _node(
    name="btn",
    pos=(0.5, 0.5),
    size=(0.1, 0.1),
    z_global=0,
    path="Root > btn",
    visible=True,
    text=None,
):
    return PocoNode(
        name=name,
        type="Button",
        text=text,
        visible=visible,
        pos=pos,
        size=size,
        poco_path=path,
        payload={
            "zOrders": {"local": z_global, "global": z_global},
            "anchorPoint": [0.5, 0.5],
        },
    )


def _make_store(tmp_path, nodes=None, page_hash="page1"):
    store = UITreeStore(str(tmp_path / "ui_trees"))
    store.update(page_hash, nodes or [])
    return store


class TestClickEffective:
    def test_target_disappeared(self):
        post = [_node(name="otherNode")]
        assert _click_effective("btnOK", (0.5, 0.5), [_node(name="btnOK")], post) is True

    def test_target_moved(self):
        post = [_node(name="btnOK", pos=(0.5, 0.3))]
        assert _click_effective("btnOK", (0.5, 0.5), [_node(name="btnOK")], post) is True

    def test_target_unmoved(self):
        pre = [_node(name="btnOK", pos=(0.5, 0.5))]
        post = [_node(name="btnOK", pos=(0.5, 0.5))]
        assert _click_effective("btnOK", (0.5, 0.5), pre, post) is False

    def test_target_tiny_jitter_ignored(self):
        pre = [_node(name="btnOK", pos=(0.5, 0.5))]
        post = [_node(name="btnOK", pos=(0.51, 0.50))]
        assert _click_effective("btnOK", (0.5, 0.5), pre, post) is False

    def test_page_hash_changed(self):
        pre = [
            _node(name="btnOK", pos=(0.5, 0.5)),
            _node(name="btnStart", pos=(0.2, 0.2), path="Root > btnStart"),
        ]
        post = [
            _node(name="btnOK", pos=(0.5, 0.5)),
            _node(name="btnTrack", pos=(0.2, 0.2), path="Root > btnTrack"),
        ]
        assert _click_effective("btnOK", (0.5, 0.5), pre, post) is True


@patch("game_agent.tools.macros.time.sleep", return_value=None)
class TestSmartClick:
    """All tests patch time.sleep so the post-action waits are instant."""

    def test_click_verified_success(self, _sleep, tmp_path):
        """Target on-screen, target disappears after click → success."""
        target = _node(name="btnOK", pos=(0.5, 0.4), path="Root > btnOK")
        new_page = _node(name="resultLabel", pos=(0.5, 0.5), path="Root > resultLabel")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[target]),    # Phase 1: find target
            MockScreen(poco_tree=[new_page]),  # Phase 2 verify: target gone
        ])
        store = _make_store(tmp_path, [target])

        result = smart_click(device, store, SmartClickInput(node_name="btnOK"))

        assert result.success is True
        assert "btnOK" in result.message
        clicks = [c for c in device.call_log if c["method"] == "click"]
        assert len(clicks) == 1
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) == 0

    def test_click_blocked_then_scrolls(self, _sleep, tmp_path):
        """Target on-screen but occluded — target stays put after click, triggers scroll."""
        target = _node(
            name="btnChallenge", pos=(0.84, 0.92), size=(0.07, 0.03),
            z_global=2, path="Root > List > btnChallenge",
        )
        # After blocked click, target is still at exact same position
        target_still_there = _node(
            name="btnChallenge", pos=(0.84, 0.92), size=(0.07, 0.03),
            z_global=2, path="Root > List > btnChallenge",
        )
        # After scroll, target moved up into safe area
        revealed = _node(
            name="btnChallenge", pos=(0.84, 0.70), size=(0.07, 0.03),
            z_global=2, path="Root > List > btnChallenge",
        )
        after_click = _node(name="newPage", pos=(0.5, 0.5), path="Root > newPage")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[target]),              # attempt 0 Phase 1: find target
            MockScreen(poco_tree=[target_still_there]),   # attempt 0 Phase 2: target unmoved → blocked
            # swipe() advances cursor
            MockScreen(poco_tree=[revealed]),             # attempt 1 Phase 1: target moved after scroll
            MockScreen(poco_tree=[after_click]),          # attempt 1 Phase 2: target gone → success
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnChallenge"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1, "should have scrolled after blocked click"

    def test_off_screen_scrolls_then_clicks(self, _sleep, tmp_path):
        """Target off-screen below — scroll up, then click when on-screen."""
        off_target = _node(name="btnFar", pos=(0.5, 1.3), size=(0.1, 0.05),
                           path="Root > btnFar")
        on_target = _node(name="btnFar", pos=(0.5, 0.7), size=(0.1, 0.05),
                          path="Root > btnFar")
        after_click = _node(name="done", pos=(0.5, 0.5), path="Root > done")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[off_target]),   # Phase 1: off-screen → scroll
            MockScreen(poco_tree=[on_target]),    # next iter Phase 1: on-screen → click
            MockScreen(poco_tree=[after_click]),  # Phase 2 verify: target gone → success
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnFar"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1

    def test_node_not_found_returns_failure(self, _sleep, tmp_path):
        """Node never appears — fail after blind scroll limit."""
        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[]),
            MockScreen(poco_tree=[]),
            MockScreen(poco_tree=[]),
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnMissing"))

        assert result.success is False
        assert "未找到" in result.message

    def test_blind_scroll_finds_node(self, _sleep, tmp_path):
        """Node absent at first, appears after a blind scroll."""
        target = _node(name="btnNew", pos=(0.5, 0.5), path="Root > btnNew")
        after_click = _node(name="result", pos=(0.5, 0.5), path="Root > result")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[]),         # empty → blind swipe
            MockScreen(poco_tree=[target]),   # target found → click
            MockScreen(poco_tree=[after_click]),  # verify: target gone → success
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnNew"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1

    def test_find_by_text(self, _sleep, tmp_path):
        """Node found by its text field when name doesn't match."""
        target = _node(name="GTextField", text="挑战",
                       pos=(0.5, 0.4), path="Root > GTextField")
        after_click = _node(name="newPage", pos=(0.5, 0.5), path="Root > newPage")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[target]),       # find by text
            MockScreen(poco_tree=[after_click]),  # verify: target gone
        ])
        store = _make_store(tmp_path, [target])

        result = smart_click(device, store, SmartClickInput(node_name="挑战"))

        assert result.success is True

    def test_max_attempts_exceeded(self, _sleep, tmp_path):
        """Target stays blocked across all attempts — fail gracefully."""
        target = _node(name="btnStuck", pos=(0.84, 0.85), size=(0.07, 0.03),
                       path="Root > btnStuck")
        always_same = MockScreen(poco_tree=[target])
        device = MockDevice()
        device.load_scenario([always_same] * 20)
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnStuck"))

        assert result.success is False
        assert "尝试" in result.message

    def test_edge_position_clamped(self, _sleep, tmp_path):
        """Target at screen edge — coordinates clamped to safe bounds."""
        target = _node(name="btnEdge", pos=(0.99, 0.96), path="Root > btnEdge")
        after_click = _node(name="newPage", pos=(0.5, 0.5), path="Root > newPage")

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[target]),       # find target
            MockScreen(poco_tree=[after_click]),   # verify: target gone
        ])
        store = _make_store(tmp_path, [target])

        result = smart_click(device, store, SmartClickInput(node_name="btnEdge"))

        assert result.success is True
        clicks = [c for c in device.call_log if c["method"] == "click"]
        assert len(clicks) == 1
        cx, cy = clicks[0]["pos"]
        assert cx <= 0.98
        assert cy <= 0.98
