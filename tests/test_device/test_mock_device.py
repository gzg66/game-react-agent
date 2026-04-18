"""Tests for MockDevice."""

from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen


def test_mock_device_default_state():
    device = MockDevice()
    assert device.get_screen_size() == (720, 1280)
    assert device.get_poco_tree() == []


def test_mock_device_load_scenario():
    device = MockDevice()
    nodes = [PocoNode(name="btn_test", type="Button", poco_path="btn_test")]
    device.load_scenario([MockScreen(poco_tree=nodes)])
    tree = device.get_poco_tree()
    assert len(tree) == 1
    assert tree[0].name == "btn_test"


def test_mock_device_click_advances():
    device = MockDevice()
    screen1 = MockScreen(poco_tree=[PocoNode(name="page1", type="Button")])
    screen2 = MockScreen(poco_tree=[PocoNode(name="page2", type="Button")])
    device.load_scenario([screen1, screen2])

    assert device.get_poco_tree()[0].name == "page1"
    device.click((0.5, 0.5))
    assert device.get_poco_tree()[0].name == "page2"


def test_mock_device_click_poco_not_found():
    device = MockDevice()
    device.load_scenario([MockScreen(node_existence={"btn_ok": False})])
    result = device.click_poco("btn_ok")
    assert result.success is False


def test_mock_device_call_log():
    device = MockDevice()
    device.load_scenario([MockScreen()])
    device.click((0.1, 0.2))
    device.swipe((0.1, 0.1), (0.9, 0.9))
    assert len(device.call_log) == 2
    assert device.call_log[0]["method"] == "click"
    assert device.call_log[1]["method"] == "swipe"


def test_mock_device_wait_for_node():
    device = MockDevice()
    device.load_scenario([MockScreen(node_existence={"btn_hero": True, "btn_missing": False})])
    assert device.wait_for_node("btn_hero") is True
    assert device.wait_for_node("btn_missing") is False


def test_mock_device_screenshot():
    device = MockDevice()
    data = device.screenshot()
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_mock_device_stays_on_last_screen():
    device = MockDevice()
    device.load_scenario([MockScreen(poco_tree=[PocoNode(name="only", type="Button")])])
    device.click((0.5, 0.5))
    device.click((0.5, 0.5))
    assert device.get_poco_tree()[0].name == "only"
