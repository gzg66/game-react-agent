"""Macro skills — higher-level operations composed from atomic tools and graph queries."""

from __future__ import annotations

import logging
import time

from game_agent.device.base import DeviceController, PocoNode
from game_agent.graph.models import UIStateGraph
from game_agent.graph.navigator import GraphNavigator
from game_agent.perception.occlusion import (
    ScrollVector,
    check_occlusion,
    compute_scroll_to_reveal,
)
from game_agent.perception.poco_tree import PocoTreeExtractor
from game_agent.perception.ui_tree_store import UITreeStore
from game_agent.tools.schemas import ClearAllPopupsInput, MapsToInput, SmartClickInput, ToolResult

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


def _find_node_in_tree(nodes: list[PocoNode], name: str) -> PocoNode | None:
    """Resolve a node by name, path suffix, or display text."""
    for node in nodes:
        if node.name == name:
            return node
    suffix = f" > {name}"
    for node in nodes:
        if node.poco_path.endswith(suffix):
            return node
    for node in nodes:
        if node.text and node.text == name:
            return node
    return None


_SMART_CLICK_MAX_ATTEMPTS = 5
_SMART_CLICK_SCROLL_STEP = 0.25
_SMART_CLICK_POST_SCROLL_WAIT_S = 1.0
_SMART_CLICK_VERIFY_WAIT_S = 0.5
_SMART_CLICK_BLIND_SCROLL_LIMIT = 2
_DEFAULT_BLIND_SCROLL = ScrollVector(
    start=(0.5, 0.65), end=(0.5, 0.40), description="盲滑：默认向上滑动",
)


def _tree_changed(before: list[PocoNode], after: list[PocoNode]) -> bool:
    """Check whether the visible node set changed between two tree snapshots."""
    pre = {n.name for n in before if n.visible}
    post = {n.name for n in after if n.visible}
    return pre != post


def smart_click(
    device: DeviceController,
    ui_tree_store: UITreeStore,
    params: SmartClickInput,
) -> ToolResult:
    """Click a node with post-click verification and scroll fallback.

    1. Find target → if on-screen, click it
    2. Verify: re-read tree → if changed, click worked → return success
    3. If unchanged (blocked) or off-screen → scroll to reveal → retry
    """
    blind_scrolls = 0

    for attempt in range(_SMART_CLICK_MAX_ATTEMPTS):
        # --- Phase 1: fetch tree, find target ---
        all_nodes = device.get_poco_tree()
        visible_nodes = [n for n in all_nodes if n.visible]
        target = _find_node_in_tree(visible_nodes, params.node_name)

        if target is None:
            target = ui_tree_store.resolve(params.node_name)

        if target is None:
            if blind_scrolls >= _SMART_CLICK_BLIND_SCROLL_LIMIT:
                return ToolResult(
                    success=False,
                    message=(
                        f"经过 {blind_scrolls} 次盲滑后仍未找到节点「{params.node_name}」"
                    ),
                )
            logger.info(
                "smart_click：未找到「%s」，执行盲滑（第 %d 次）",
                params.node_name,
                blind_scrolls + 1,
            )
            device.swipe(
                start=_DEFAULT_BLIND_SCROLL.start,
                end=_DEFAULT_BLIND_SCROLL.end,
                duration=0.5,
            )
            time.sleep(_SMART_CLICK_POST_SCROLL_WAIT_S)
            blind_scrolls += 1
            continue

        # --- Phase 2: try clicking if target center is on-screen ---
        x, y = target.pos
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            cx = max(0.02, min(0.98, x))
            cy = max(0.02, min(0.98, y))
            device.click((cx, cy))

            time.sleep(_SMART_CLICK_VERIFY_WAIT_S)
            post_nodes = device.get_poco_tree()
            post_visible = [n for n in post_nodes if n.visible]

            if _tree_changed(visible_nodes, post_visible):
                return ToolResult(
                    success=True,
                    message=(
                        f"智能点击成功：「{params.node_name}」@ ({cx:.2f}, {cy:.2f})，"
                        f"共尝试 {attempt + 1} 次"
                    ),
                    data={"x": cx, "y": cy, "attempts": attempt + 1},
                )

            logger.info(
                "smart_click[%d/%d]：点击 (%0.2f, %0.2f) 后 UI 未变化，"
                "判定被遮挡，尝试滑动",
                attempt + 1,
                _SMART_CLICK_MAX_ATTEMPTS,
                cx,
                cy,
            )

        # --- Phase 3: scroll to reveal (off-screen or click was blocked) ---
        occlusion = check_occlusion(target, visible_nodes)
        scroll = compute_scroll_to_reveal(
            target, occlusion, _SMART_CLICK_SCROLL_STEP,
        )
        if scroll is None:
            scroll = _DEFAULT_BLIND_SCROLL

        logger.info(
            "smart_click[%d/%d]：%s (%.2f,%.2f)->(%.2f,%.2f)",
            attempt + 1,
            _SMART_CLICK_MAX_ATTEMPTS,
            scroll.description,
            *scroll.start,
            *scroll.end,
        )
        device.swipe(start=scroll.start, end=scroll.end, duration=0.5)
        time.sleep(_SMART_CLICK_POST_SCROLL_WAIT_S)

    return ToolResult(
        success=False,
        message=(
            f"已尝试 {_SMART_CLICK_MAX_ATTEMPTS} 次，"
            f"仍无法安全点击「{params.node_name}」"
        ),
    )


def build_macro_tools(
    device: DeviceController,
    graph: UIStateGraph,
    navigator: GraphNavigator,
    tree_extractor: PocoTreeExtractor,
    ui_tree_store: UITreeStore | None = None,
) -> dict[str, tuple]:
    """Return macro tools for registry registration."""
    tools: dict[str, tuple] = {
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
    if ui_tree_store is not None:
        tools["smart_click"] = (
            lambda params: smart_click(device, ui_tree_store, params),
            SmartClickInput,
            (
                "智能点击：自动检测节点是否被遮挡或在屏幕外，"
                "若是则自动滑动使其可见后再点击。"
                "适用于长列表、滚动视图、底栏遮挡等场景"
            ),
        )
    return tools
