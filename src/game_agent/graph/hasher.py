"""Page feature hashing — stable hash based on Poco tree structure."""

from __future__ import annotations

import hashlib

from game_agent.device.base import PocoNode

STRUCTURAL_TYPES = {"Button", "InputField", "Toggle", "Slider", "ScrollView", "Dropdown", "Panel"}


class PageHasher:
    """Computes a stable hash for a page based on its interactive node structure.

    Uses sorted (name, type) pairs from visible interactive nodes.
    Deliberately ignores dynamic text content so the same page with
    different resource counts hashes identically.
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        self._algorithm = algorithm

    def compute(self, nodes: list[PocoNode]) -> str:
        structural_nodes = [n for n in nodes if n.visible and self._is_structural(n)]
        fingerprint_parts = sorted(f"{n.name}:{n.type}" for n in structural_nodes)
        raw = "|".join(fingerprint_parts)
        return hashlib.new(self._algorithm, raw.encode()).hexdigest()[:16]

    def _is_structural(self, node: PocoNode) -> bool:
        if not node.name:
            return False
        if node.type in STRUCTURAL_TYPES:
            return True
        name_lower = node.name.lower()
        return any(prefix in name_lower for prefix in ("btn", "button", "tab", "panel", "input"))
