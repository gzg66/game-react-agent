"""Atomic tools wrapping individual device operations."""

from __future__ import annotations

import logging
from dataclasses import asdict

from game_agent.device.base import DeviceController
from game_agent.exceptions import PocoNodeNotFoundError
from game_agent.perception.ui_tree_store import UITreeStore
from game_agent.tools.schemas import (
    AirtestTouchPosInput,
    PocoClickInput,
    SwipeInput,
    ToolResult,
    WaitForNodeInput,
)

logger = logging.getLogger(__name__)


def poco_click(
    device: DeviceController,
    ui_tree_store: UITreeStore,
    params: PocoClickInput,
) -> ToolResult:
    node = ui_tree_store.resolve(params.node_name)

    try:
        result = device.click_poco(params.node_name)
        if result.success:
            return ToolResult(
                success=True,
                message=f"已点击节点：{params.node_name}",
                data=asdict(result),
            )
        logger.debug("Poco 按名称点击返回失败，尝试坐标兜底")
    except (PocoNodeNotFoundError, Exception) as exc:
        logger.debug("Poco 按名称点击异常（%s），尝试坐标兜底", exc)

    if node is not None:
        x, y = node.pos
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            result = device.click((x, y))
            return ToolResult(
                success=result.success,
                message=f"已通过坐标点击节点 {params.node_name}（{x:.2f}, {y:.2f}）",
                data=asdict(result),
            )

    return ToolResult(
        success=False,
        message=f"未找到节点：{params.node_name}",
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


def build_atomic_tools(
    device: DeviceController,
    ui_tree_store: UITreeStore,
) -> dict[str, tuple]:
    """Return a mapping of (name -> (bound_fn, schema, description)) for registration."""
    return {
        "poco_click": (
            lambda params: poco_click(device, ui_tree_store, params),
            PocoClickInput,
            "按节点名称点击界面元素（自动解析坐标，无需完整路径）",
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
