"""Integration tests for the smart_click macro tool."""

from unittest.mock import patch

from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.perception.ui_tree_store import UITreeStore
from game_agent.tools.macros import smart_click
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


@patch("game_agent.tools.macros.time.sleep", return_value=None)
class TestSmartClick:
    """All tests patch time.sleep so the 1-second post-scroll wait is instant."""

    def test_no_occlusion_clicks_directly(self, _sleep, tmp_path):
        """Target is fully visible — click on first attempt."""
        target = _node(name="btnOK", pos=(0.5, 0.4), z_global=5, path="Root > btnOK")
        device = MockDevice()
        device.load_scenario([MockScreen(poco_tree=[target])])
        store = _make_store(tmp_path, [target])

        result = smart_click(device, store, SmartClickInput(node_name="btnOK"))

        assert result.success is True
        assert "btnOK" in result.message
        clicks = [c for c in device.call_log if c["method"] == "click"]
        assert len(clicks) == 1

    def test_occluded_scrolls_then_clicks(self, _sleep, tmp_path):
        """Target is occluded on screen 1; after swipe, screen 2 shows it clear."""
        occluded_target = _node(
            name="btnChallenge", pos=(0.84, 0.845), size=(0.07, 0.03),
            z_global=2, path="Root > List > btnChallenge",
        )
        bottom_bar = _node(
            name="barBottom", pos=(0.84, 0.846), size=(0.07, 0.03),
            z_global=3, path="Root > BottomBar > barBottom",
        )
        revealed_target = _node(
            name="btnChallenge", pos=(0.84, 0.694), size=(0.07, 0.03),
            z_global=2, path="Root > List > btnChallenge",
        )

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[occluded_target, bottom_bar]),  # before scroll
            MockScreen(poco_tree=[revealed_target]),               # after scroll
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnChallenge"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1, "should have swiped at least once"

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

    def test_large_popup_returns_popup_warning(self, _sleep, tmp_path):
        """Target blocked by a large popup — advise clear_all_popups."""
        target = _node(
            name="btnAction", pos=(0.5, 0.5), size=(0.1, 0.05),
            z_global=1, path="Root > btnAction",
        )
        popup = _node(
            name="popup_bg", pos=(0.5, 0.5), size=(0.9, 0.9),
            z_global=10, path="Root > Popup > popup_bg",
        )
        device = MockDevice()
        device.load_scenario([MockScreen(poco_tree=[target, popup])])
        store = _make_store(tmp_path, [target, popup])

        result = smart_click(device, store, SmartClickInput(node_name="btnAction"))

        assert result.success is False
        assert "弹窗" in result.message

    def test_blind_scroll_finds_node(self, _sleep, tmp_path):
        """Node absent at first, appears after a blind scroll."""
        target = _node(name="btnNew", pos=(0.5, 0.5), z_global=1, path="Root > btnNew")
        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[]),       # screen 1: empty
            MockScreen(poco_tree=[target]), # screen 2: target appears after swipe
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnNew"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1

    def test_find_by_text(self, _sleep, tmp_path):
        """Node found by its text field when name doesn't match."""
        target = _node(
            name="GTextField", text="挑战",
            pos=(0.5, 0.4), z_global=5, path="Root > GTextField",
        )
        device = MockDevice()
        device.load_scenario([MockScreen(poco_tree=[target])])
        store = _make_store(tmp_path, [target])

        result = smart_click(device, store, SmartClickInput(node_name="挑战"))

        assert result.success is True

    def test_max_attempts_exceeded(self, _sleep, tmp_path):
        """Target stays occluded across all attempts — fail gracefully."""
        target = _node(
            name="btnStuck", pos=(0.84, 0.85), size=(0.07, 0.03),
            z_global=1, path="Root > btnStuck",
        )
        blocker = _node(
            name="bar", pos=(0.84, 0.85), size=(0.07, 0.03),
            z_global=5, path="Root > bar",
        )
        always_occluded = MockScreen(poco_tree=[target, blocker])
        device = MockDevice()
        device.load_scenario([always_occluded] * 6)
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnStuck"))

        assert result.success is False
        assert "尝试" in result.message

    def test_structural_overlay_scrolls_then_clicks(self, _sleep, tmp_path):
        """ClickEffect overlay is not a popup — smart_click scrolls past it."""
        target = _node(
            name="btnChallenge", pos=(0.84, 0.84), size=(0.07, 0.03),
            z_global=2, path="Root > Window > btnChallenge",
        )
        click_effect = _node(
            name="ClickEffect", pos=(0.5, 0.5), size=(1.0, 1.0),
            z_global=10, path="Root > ClickEffect",
        )
        revealed_target = _node(
            name="btnChallenge", pos=(0.84, 0.60), size=(0.07, 0.03),
            z_global=2, path="Root > Window > btnChallenge",
        )

        device = MockDevice()
        device.load_scenario([
            MockScreen(poco_tree=[target, click_effect]),
            MockScreen(poco_tree=[revealed_target]),
        ])
        store = _make_store(tmp_path)

        result = smart_click(device, store, SmartClickInput(node_name="btnChallenge"))

        assert result.success is True
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) >= 1

    def test_large_popup_scrolls_before_failing(self, _sleep, tmp_path):
        """Real popup: smart_click tries scrolling before giving up."""
        target = _node(
            name="btnAction", pos=(0.5, 0.5), size=(0.1, 0.05),
            z_global=1, path="Root > btnAction",
        )
        popup = _node(
            name="popup_bg", pos=(0.5, 0.5), size=(0.9, 0.9),
            z_global=10, path="Root > Popup > popup_bg",
        )
        device = MockDevice()
        device.load_scenario([MockScreen(poco_tree=[target, popup])])
        store = _make_store(tmp_path, [target, popup])

        result = smart_click(device, store, SmartClickInput(node_name="btnAction"))

        assert result.success is False
        assert "弹窗" in result.message
        swipes = [c for c in device.call_log if c["method"] == "swipe"]
        assert len(swipes) == 2, "should have tried scrolling before giving up"
