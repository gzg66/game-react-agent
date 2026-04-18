"""Macro skills — higher-level operations composed from atomic tools and graph queries."""

from __future__ import annotations

import logging

from game_agent.device.base import DeviceController
from game_agent.graph.models import UIStateGraph
from game_agent.graph.navigator import GraphNavigator
from game_agent.perception.poco_tree import PocoTreeExtractor
from game_agent.tools.schemas import ClearAllPopupsInput, MapsToInput, ToolResult

logger = logging.getLogger(__name__)

CLOSE_BUTTON_PATTERNS = ["btn_close", "btn_cancel", "btn_confirm", "btn_skip", "close", "cancel"]


def maps_to(
    device: DeviceController,
    graph: UIStateGraph,
    navigator: GraphNavigator,
    tree_extractor: PocoTreeExtractor,
    params: MapsToInput,
) -> ToolResult:
    """Navigate to a target page using the cached shortest path. Zero LLM cost."""
    current_perception = tree_extractor.extract()
    current_hash = current_perception.page_hash

    path = navigator.shortest_path(current_hash, params.target_node_id)
    if path is None:
        target_node = graph.get_node(params.target_node_id)
        target_name = target_node.page_name if target_node else params.target_node_id
        return ToolResult(
            success=False,
            message=f"当前页面到“{target_name}”没有缓存路径",
        )

    if not path:
        return ToolResult(success=True, message="已经位于目标页面")

    for edge in path:
        for action in edge.actions:
            if action.action_type == "poco_click":
                result = device.click_poco(action.target)
            elif action.action_type == "touch_pos":
                pos = action.params.get("pos", (0.5, 0.5))
                result = device.click(tuple(pos))
            elif action.action_type == "swipe":
                start = tuple(action.params.get("start", (0.5, 0.5)))
                end = tuple(action.params.get("end", (0.5, 0.3)))
                result = device.swipe(start, end)
            else:
                continue

            if not result.success:
                edge.failure_count += 1
                return ToolResult(
                    success=False,
                    message=f"导航在动作 {action.action_type}({action.target}) 处失败",
                )

        edge.success_count += 1

    target_node = graph.get_node(params.target_node_id)
    target_name = target_node.page_name if target_node else params.target_node_id
    return ToolResult(
        success=True,
        message=f"已通过 {len(path)} 条边导航到“{target_name}”",
        data={"path_length": len(path)},
    )


def clear_all_popups(device: DeviceController, params: ClearAllPopupsInput) -> ToolResult:
    """Repeatedly scan for and dismiss popup dialogs."""
    dismissed = 0
    for attempt in range(params.max_attempts):
        tree = device.get_poco_tree()
        found = False
        for node in tree:
            if not node.visible:
                continue
            name_lower = node.name.lower()
            if any(pat in name_lower for pat in CLOSE_BUTTON_PATTERNS):
                result = device.click_poco(node.poco_path)
                if result.success:
                    dismissed += 1
                    found = True
                    break
        if not found:
            break

    return ToolResult(
        success=True,
        message=f"共扫描 {attempt + 1} 轮，已关闭 {dismissed} 个弹窗",
        data={"dismissed_count": dismissed},
    )


def build_macro_tools(
    device: DeviceController,
    graph: UIStateGraph,
    navigator: GraphNavigator,
    tree_extractor: PocoTreeExtractor,
) -> dict[str, tuple]:
    """Return macro tools for registry registration."""
    return {
        "maps_to": (
            lambda params: maps_to(device, graph, navigator, tree_extractor, params),
            MapsToInput,
            "使用缓存图中的最短路径导航到目标页面（无需额外 LLM 成本）",
        ),
        "clear_all_popups": (
            lambda params: clear_all_popups(device, params),
            ClearAllPopupsInput,
            "关闭当前可见的所有弹窗",
        ),
    }
