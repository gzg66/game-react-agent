"""Atomic tools wrapping individual device operations."""

from __future__ import annotations

from dataclasses import asdict

from game_agent.device.base import DeviceController
from game_agent.tools.schemas import (
    AirtestTouchPosInput,
    PocoClickInput,
    SwipeInput,
    ToolResult,
    WaitForNodeInput,
)


def poco_click(device: DeviceController, params: PocoClickInput) -> ToolResult:
    result = device.click_poco(params.poco_path)
    return ToolResult(
        success=result.success,
        message=f"已点击 Poco 节点：{params.poco_path}",
        data=asdict(result),
    )


def airtest_touch_pos(device: DeviceController, params: AirtestTouchPosInput) -> ToolResult:
    result = device.click((params.x, params.y))
    return ToolResult(
        success=result.success,
        message=f"已点击坐标 ({params.x:.2f}, {params.y:.2f})",
        data=asdict(result),
    )


def swipe(device: DeviceController, params: SwipeInput) -> ToolResult:
    result = device.swipe(
        start=(params.start_x, params.start_y),
        end=(params.end_x, params.end_y),
        duration=params.duration,
    )
    return ToolResult(
        success=result.success,
        message=f"已滑动 ({params.start_x:.2f},{params.start_y:.2f}) -> ({params.end_x:.2f},{params.end_y:.2f})",
        data=asdict(result),
    )


def wait_for_node(device: DeviceController, params: WaitForNodeInput) -> ToolResult:
    found = device.wait_for_node(params.poco_path, timeout=params.timeout)
    return ToolResult(
        success=found,
        message=f"节点 {params.poco_path} {'已出现' if found else '等待超时'}",
        data={"found": found, "poco_path": params.poco_path},
    )


def build_atomic_tools(device: DeviceController) -> dict[str, tuple]:
    """Return a mapping of (name -> (bound_fn, schema, description)) for registration."""
    return {
        "poco_click": (
            lambda params: poco_click(device, params),
            PocoClickInput,
            "按 Poco 路径点击界面元素",
        ),
        "airtest_touch_pos": (
            lambda params: airtest_touch_pos(device, params),
            AirtestTouchPosInput,
            "按相对坐标点击指定屏幕位置",
        ),
        "swipe": (
            lambda params: swipe(device, params),
            SwipeInput,
            "从一个位置滑动到另一个位置",
        ),
        "wait_for_node": (
            lambda params: wait_for_node(device, params),
            WaitForNodeInput,
            "等待 Poco 节点出现在屏幕上",
        ),
    }
