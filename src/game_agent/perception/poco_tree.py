"""L1 perception: Poco tree extraction, filtering, hashing, and markdown formatting."""

from __future__ import annotations

import hashlib
import time

from game_agent.config import PerceptionConfig
from game_agent.device.base import DeviceController, PocoNode
from game_agent.perception.state import L1Perception

INTERACTIVE_TYPES = {"Button", "InputField", "Toggle", "Slider", "ScrollView", "Dropdown"}


class PocoTreeExtractor:
    """Extracts and processes the Poco UI tree for L1 perception."""

    def __init__(self, device: DeviceController, config: PerceptionConfig) -> None:
        self._device = device
        self._config = config

    def extract(self) -> L1Perception:
        raw_tree = self._device.get_poco_tree()
        filtered = [n for n in raw_tree if n.visible and self._is_interactive(n)]
        markdown = self._to_markdown(filtered)
        page_hash = self.compute_hash(filtered)
        return L1Perception(
            timestamp=time.time(),
            poco_tree_markdown=markdown,
            interactive_nodes=filtered,
            page_hash=page_hash,
        )

    def compute_hash(self, nodes: list[PocoNode]) -> str:
        fingerprint_parts = sorted(f"{n.name}:{n.type}" for n in nodes if self._is_structural(n))
        raw = "|".join(fingerprint_parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _is_interactive(self, node: PocoNode) -> bool:
        if node.type in INTERACTIVE_TYPES:
            return True
        name_lower = node.name.lower()
        return any(prefix in name_lower for prefix in ("btn", "button", "toggle", "input", "tab"))

    def _is_structural(self, node: PocoNode) -> bool:
        if not node.name:
            return False
        return node.type in INTERACTIVE_TYPES or bool(node.name.strip())

    def _to_markdown(self, nodes: list[PocoNode]) -> str:
        lines = []
        for n in nodes:
            text_part = f' "{n.text}"' if n.text else ""
            lines.append(
                f"- [{n.type}] {n.name}{text_part} (位置: {n.pos[0]:.2f}, {n.pos[1]:.2f})"
            )
        return "\n".join(lines)
