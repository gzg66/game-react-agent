"""Tests for the tool registry."""

from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.tools.atomic import build_atomic_tools
from game_agent.tools.registry import ToolRegistry
from game_agent.tools.schemas import PocoClickInput, ToolResult


def test_registry_register_and_list():
    registry = ToolRegistry()
    registry.register(
        "test_tool",
        lambda params: ToolResult(success=True, message="ok"),
        PocoClickInput,
        "A test tool",
    )
    assert "test_tool" in registry.list_tools()


def test_registry_execute():
    registry = ToolRegistry()
    registry.register(
        "test_tool",
        lambda params: ToolResult(success=True, message=f"clicked {params.poco_path}"),
        PocoClickInput,
    )
    result = registry.execute("test_tool", {"poco_path": "btn_hero"})
    assert result.success is True
    assert "btn_hero" in result.message


def test_registry_execute_unknown_tool():
    registry = ToolRegistry()
    result = registry.execute("nonexistent", {})
    assert result.success is False
    assert "未知工具" in result.message


def test_registry_execute_invalid_params():
    registry = ToolRegistry()
    registry.register(
        "test_tool",
        lambda params: ToolResult(success=True, message="ok"),
        PocoClickInput,
    )
    result = registry.execute("test_tool", {})
    assert result.success is False


def test_registry_build_atomic_tools():
    device = MockDevice()
    device.load_scenario([MockScreen(node_existence={"btn_hero": True})])
    registry = ToolRegistry()
    for name, (fn, schema, desc) in build_atomic_tools(device).items():
        registry.register(name, fn, schema, desc)
    assert len(registry.list_tools()) == 4
    result = registry.execute("poco_click", {"poco_path": "btn_hero"})
    assert result.success is True


def test_registry_gemini_tools_format():
    registry = ToolRegistry()
    registry.register("test", lambda p: ToolResult(True, "ok"), PocoClickInput, "test")
    tools = registry.get_gemini_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "test"
    assert "parameters" in tools[0]
    assert "properties" in tools[0]["parameters"]
