"""Tests for atomic tools."""

from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.tools.atomic import airtest_touch_pos, poco_click, swipe, wait_for_node
from game_agent.tools.schemas import (
    AirtestTouchPosInput,
    PocoClickInput,
    SwipeInput,
    WaitForNodeInput,
)


def test_poco_click_success():
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btn_hero": True})]
    )
    result = poco_click(device, PocoClickInput(poco_path="btn_hero"))
    assert result.success is True
    assert "btn_hero" in result.message


def test_poco_click_not_found():
    device = MockDevice()
    device.load_scenario(
        [MockScreen(node_existence={"btn_hero": False})]
    )
    result = poco_click(device, PocoClickInput(poco_path="btn_hero"))
    assert result.success is False


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
