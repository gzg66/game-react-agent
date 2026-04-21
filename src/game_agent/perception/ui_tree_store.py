"""Persistent UI tree store — saves visible nodes per page and resolves names to paths."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from game_agent.device.base import PocoNode

logger = logging.getLogger(__name__)


class UITreeStore:
    """Stores the full visible UI tree per page_hash on disk.

    When the ReAct loop captures a new page, it calls ``update()`` to cache
    all visible nodes in memory and persist them to
    ``<store_dir>/<page_hash>.json``.  The ``resolve()`` method lets tools
    look up a node by its short name (e.g. ``btnLogin``) and get back the
    full ``PocoNode`` with its ``poco_path`` and ``pos``.
    """

    def __init__(self, store_dir: str = "data/ui_trees") -> None:
        self._store_dir = Path(store_dir)
        self._current_page_hash: str | None = None
        self._current_nodes: list[PocoNode] = []

    @property
    def current_page_hash(self) -> str | None:
        return self._current_page_hash

    def update(self, page_hash: str, nodes: list[PocoNode]) -> None:
        self._current_page_hash = page_hash
        self._current_nodes = list(nodes)
        self._save_if_new(page_hash, nodes)

    def resolve(self, node_name: str) -> PocoNode | None:
        """Find a node by name in the current page's tree.

        Match priority:
        1. Exact ``node.name`` match
        2. ``poco_path`` ends with ``" > <node_name>"``
        """
        if not self._current_nodes:
            return None

        for node in self._current_nodes:
            if node.name == node_name:
                return node

        suffix = f" > {node_name}"
        for node in self._current_nodes:
            if node.poco_path.endswith(suffix):
                return node

        return None

    # ---- persistence ----

    def _save_if_new(self, page_hash: str, nodes: list[PocoNode]) -> None:
        path = self._store_dir / f"{page_hash}.json"
        if path.exists():
            return
        self._store_dir.mkdir(parents=True, exist_ok=True)
        data = [_node_to_dict(n) for n in nodes]
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
        logger.info("UI 树已保存：%s（%d 个节点）", path.name, len(nodes))

    def load(self, page_hash: str) -> list[PocoNode] | None:
        """Load a previously saved tree from disk."""
        path = self._store_dir / f"{page_hash}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [_dict_to_node(d) for d in raw]
        except Exception as exc:
            logger.warning("加载 UI 树失败 %s：%s", page_hash, exc)
            return None


def _node_to_dict(node: PocoNode) -> dict:
    return {
        "name": node.name,
        "type": node.type,
        "text": node.text,
        "visible": node.visible,
        "pos": list(node.pos),
        "size": list(node.size),
        "children_count": node.children_count,
        "poco_path": node.poco_path,
    }


def _dict_to_node(d: dict) -> PocoNode:
    return PocoNode(
        name=d["name"],
        type=d["type"],
        text=d.get("text"),
        visible=d.get("visible", True),
        pos=tuple(d.get("pos", [0.0, 0.0])),
        size=tuple(d.get("size", [0.0, 0.0])),
        children_count=d.get("children_count", 0),
        poco_path=d.get("poco_path", ""),
    )
