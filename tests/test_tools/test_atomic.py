"""Tests for atomic tools."""

from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.perception.ui_tree_store import UITreeStore
from game_agent.tools.atomic import airtest_touch_pos, poco_click, swipe, wait_for_node
from game_agent.tools.schemas import (
    AirtestTouchPosInput,
    PocoClickInput,
    SwipeInput,
    WaitForNodeInput,
)


def _make_store_with_nodes(tmp_path, nodes, page_hash="test_page"):
    store = UITreeStore(str(tmp_path / "ui_trees"))
    store.update(page_hash, nodes)
    return store


def test_poco_click_success_by_name(tmp_path):
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btnLogin": True})]
    )
    nodes = [
        PocoNode(
            name="btnLogin", type="Button", visible=True,
            pos=(0.67, 0.57), poco_path="Scene > GRoot > btnLogin",
        ),
    ]
    store = _make_store_with_nodes(tmp_path, nodes)
    result = poco_click(device, store, PocoClickInput(node_name="btnLogin"))
    assert result.success is True
    assert "btnLogin" in result.message


def test_poco_click_fallback_to_coordinates(tmp_path):
    """When Poco name query fails, fall back to coordinate touch."""
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btnLogin": False})]
    )
    nodes = [
        PocoNode(
            name="btnLogin", type="Button", visible=True,
            pos=(0.67, 0.57), poco_path="Scene > GRoot > btnLogin",
        ),
    ]
    store = _make_store_with_nodes(tmp_path, nodes)
    result = poco_click(device, store, PocoClickInput(node_name="btnLogin"))
    assert result.success is True
    assert "坐标" in result.message
    assert "0.67" in result.message


def test_poco_click_not_found_anywhere(tmp_path):
    """Node not in store and not in device → failure."""
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btnLogin": False})]
    )
    store = UITreeStore(str(tmp_path / "ui_trees"))
    store.update("page1", [])
    result = poco_click(device, store, PocoClickInput(node_name="btnLogin"))
    assert result.success is False
    assert "未找到" in result.message


def test_airtest_touch_pos():
    device = MockDevice()
    device.load_scenario([MockScreen()])
    result = airtest_touch_pos(device, AirtestTouchPosInput(x=0.5, y=0.5))
    assert result.success is True
    assert "0.50" in result.message


def test_swipe():
    device = MockDevice()
    device.load_scenario([MockScreen()])
    result = swipe(
        device,
        SwipeInput(start_x=0.1, start_y=0.5, end_x=0.9, end_y=0.5, duration=0.5),
    )
    assert result.success is True


def test_wait_for_node_found():
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btn_hero": True})]
    )
    result = wait_for_node(device, WaitForNodeInput(poco_path="btn_hero"))
    assert result.success is True
    assert "已出现" in result.message


def test_wait_for_node_timeout():
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btn_hero": False})]
    )
    result = wait_for_node(device, WaitForNodeInput(poco_path="btn_hero", timeout=1.0))
    assert result.success is False
    assert "超时" in result.message
