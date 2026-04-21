"""L1 perception: Poco tree extraction, filtering, hashing, and markdown formatting."""

from __future__ import annotations

import logging
import time

from game_agent.config import PerceptionConfig
from game_agent.device.base import DeviceController, PocoNode
from game_agent.graph.hasher import PageHasher
from game_agent.perception.state import L1Perception

logger = logging.getLogger(__name__)

INTERACTIVE_TYPES = {"Button", "InputField", "Toggle", "Slider", "ScrollView", "Dropdown"}

OVERLAY_SIZE_THRESHOLD = 0.7

OVERLAY_NAME_EXCLUSIONS: tuple[str, ...] = (
    "mask", "side", "black", "background", "bg", "border", "shadow", "frame",
    "ggroup", "ggraph", "container", "node", "panel", "layer", "root",
    "canvas", "scene", "wrapper",
)


class PocoTreeExtractor:
    """Extracts and processes the Poco UI tree for L1 perception."""

    def __init__(self, device: DeviceController, config: PerceptionConfig) -> None:
        self._device = device
        self._config = config
        self._hasher = PageHasher()

    def extract(self) -> L1Perception:
        raw_tree = self._device.get_poco_tree()
        visible_nodes = [n for n in raw_tree if n.visible]
        filtered = [n for n in raw_tree if n.visible and self._is_interactive(n)]
        filtered = self._filter_obscured(raw_tree, filtered)
        markdown = self._to_full_markdown(visible_nodes)
        page_hash = self.compute_hash(filtered)
        return L1Perception(
            timestamp=time.time(),
            poco_tree_markdown=markdown,
            interactive_nodes=filtered,
            page_hash=page_hash,
        )

    def compute_hash(self, nodes: list[PocoNode]) -> str:
        return self._hasher.compute(nodes)

    def extract_visible_tree_markdown(self) -> str:
        """Return markdown for all visible UI nodes without interaction filtering."""
        raw_tree = self._device.get_poco_tree()
        visible_nodes = [n for n in raw_tree if n.visible]
        return self._to_full_markdown(visible_nodes)

    def _is_interactive(self, node: PocoNode) -> bool:
        if node.type in INTERACTIVE_TYPES:
            return True
        name_lower = node.name.lower()
        return any(prefix in name_lower for prefix in ("btn", "button", "toggle", "input", "tab"))

    def _is_structural(self, node: PocoNode) -> bool:
        if not node.name:
            return False
        return node.type in INTERACTIVE_TYPES or bool(node.name.strip())

    def _filter_obscured(
        self, all_nodes: list[PocoNode], interactive: list[PocoNode]
    ) -> list[PocoNode]:
        """Filter out interactive nodes likely obscured by an overlay panel.

        Finds the topmost (last in DFS order) large panel covering most of the
        screen.  Interactive nodes outside that panel's subtree are excluded.
        """
        overlay = self._find_topmost_overlay(all_nodes)
        if overlay is None:
            return interactive

        overlay_prefix = overlay.poco_path
        result = [
            n for n in interactive
            if n.poco_path.startswith(overlay_prefix)
        ]

        # If the "overlay" would remove >50% of buttons, it's likely a
        # structural container (GGroup, GComponent, etc.), not a real modal.
        if not result or len(result) < len(interactive) * 0.5:
            return interactive

        removed_nodes = [
            n for n in interactive
            if not n.poco_path.startswith(overlay_prefix)
        ]
        if removed_nodes:
            removed_names = ", ".join(n.name for n in removed_nodes[:10])
            logger.info(
                "遮挡过滤：覆盖层「%s」遮挡了 %d 个底层按钮：%s",
                overlay.name,
                len(removed_nodes),
                removed_names,
            )
        return result

    def _find_topmost_overlay(self, nodes: list[PocoNode]) -> PocoNode | None:
        """Find the topmost large panel that is likely an overlay.

        Walk DFS order (last = topmost) looking for non-interactive visible
        nodes that cover > OVERLAY_SIZE_THRESHOLD of the screen.
        """
        candidate: PocoNode | None = None
        for node in nodes:
            if not node.visible:
                continue
            if self._is_interactive(node):
                continue
            name_lower = node.name.lower()
            if any(excl in name_lower for excl in OVERLAY_NAME_EXCLUSIONS):
                continue
            w, h = node.size
            if w >= OVERLAY_SIZE_THRESHOLD and h >= OVERLAY_SIZE_THRESHOLD:
                candidate = node
        return candidate

    def _to_markdown(self, nodes: list[PocoNode]) -> str:
        lines = []
        for n in nodes:
            text_part = f' "{n.text}"' if n.text else ""
            lines.append(
                f"- [{n.type}] {n.name}{text_part} (位置: {n.pos[0]:.2f}, {n.pos[1]:.2f})"
            )
        return "\n".join(lines)

    def _to_full_markdown(self, nodes: list[PocoNode]) -> str:
        lines = []
        for n in nodes:
            text_part = f' text="{n.text}"' if n.text else ""
            lines.append(
                "- "
                f"[{n.type}] {n.name or '（空名称）'}"
                f"{text_part}"
                f" visible={n.visible}"
                f" pos=({n.pos[0]:.2f}, {n.pos[1]:.2f})"
                f" size=({n.size[0]:.2f}, {n.size[1]:.2f})"
                f" children={n.children_count}"
                f" path={n.poco_path or '（空路径）'}"
            )
        return "\n".join(lines)
