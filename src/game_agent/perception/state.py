"""Perception data models for L1/L2 snapshots and UI diff."""

from __future__ import annotations

from dataclasses import dataclass, field

from game_agent.device.base import PocoNode


@dataclass
class L1Perception:
    """Lightweight text-only perception (default level)."""

    timestamp: float
    poco_tree_markdown: str
    interactive_nodes: list[PocoNode]
    page_hash: str


@dataclass
class L2Perception(L1Perception):
    """Full multi-modal perception with screenshot."""

    screenshot_b64: str = ""
    screenshot_raw: bytes = b""


@dataclass
class PerceptionState:
    """Minimal snapshot used for diff comparison between steps."""

    page_hash: str
    poco_node_names: frozenset[str] = field(default_factory=frozenset)
    screenshot_bytes: bytes | None = None


@dataclass
class UIDiff:
    """Result of comparing two perception states."""

    hash_changed: bool
    structural_similarity: float
    added_nodes: list[str]
    removed_nodes: list[str]
    is_loading: bool
