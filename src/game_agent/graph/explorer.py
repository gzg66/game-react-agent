"""DFS exploration of game UI to build the state machine graph."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from game_agent.config import GraphConfig
from game_agent.device.base import DeviceController
from game_agent.graph.annotator import PageAnnotator
from game_agent.graph.hasher import PageHasher
from game_agent.graph.models import EdgeAction, GraphEdge, GraphNode, UIStateGraph
from game_agent.graph.storage import GraphStorage
from game_agent.perception.poco_tree import PocoTreeExtractor

logger = logging.getLogger(__name__)

BACK_BUTTON_PATTERNS = ["btn_back", "btn_close", "btn_return", "button_back"]


@dataclass
class ExplorationReport:
    """Summary of a DFS exploration run."""

    nodes_discovered: int = 0
    edges_discovered: int = 0
    nodes_revisited: int = 0
    max_depth_reached: int = 0
    errors: list[str] = field(default_factory=list)


class GraphExplorer:
    """Automated DFS exploration of the game UI.

    For each unvisited page:
      1. Capture L1 perception and compute page hash.
      2. If new: call Gemini annotator for semantic labeling.
      3. For each interactive button: click, observe result, record edge.
      4. Navigate back and continue DFS.
    """

    def __init__(
        self,
        device: DeviceController,
        tree_extractor: PocoTreeExtractor,
        hasher: PageHasher,
        annotator: PageAnnotator,
        graph: UIStateGraph,
        storage: GraphStorage,
        config: GraphConfig,
    ) -> None:
        self._device = device
        self._tree_extractor = tree_extractor
        self._hasher = hasher
        self._annotator = annotator
        self._graph = graph
        self._storage = storage
        self._config = config
        self._report = ExplorationReport()

    async def explore(self, max_depth: int | None = None) -> ExplorationReport:
        """Run DFS exploration from the current screen state."""
        self._report = ExplorationReport()
        depth_limit = max_depth or self._config.exploration_max_depth
        await self._explore_node(0, depth_limit)
        logger.info(
            "探索完成：共发现 %d 个节点、%d 条边",
            self._report.nodes_discovered,
            self._report.edges_discovered,
        )
        return self._report

    async def _explore_node(self, depth: int, max_depth: int) -> None:
        if depth > self._report.max_depth_reached:
            self._report.max_depth_reached = depth

        perception = self._tree_extractor.extract()
        page_hash = perception.page_hash

        if self._graph.has_node(page_hash):
            node = self._graph.get_node(page_hash)
            node.visit_count += 1
            self._storage.save_node(node)
            self._report.nodes_revisited += 1
            return

        try:
            annotation = self._annotator.annotate(perception)
        except Exception as exc:
            logger.warning("页面 %s 标注失败：%s", page_hash, exc)
            annotation_name = f"页面_{page_hash[:8]}"
            from game_agent.graph.annotator import PageAnnotation

            annotation = PageAnnotation(
                page_name=annotation_name,
                page_description="自动发现（标注失败）",
                page_category="未知",
                key_buttons=[n.name for n in perception.interactive_nodes[:5]],
            )

        node = GraphNode(
            node_id=page_hash,
            page_name=annotation.page_name,
            page_description=annotation.page_description,
            page_category=annotation.page_category,
            key_buttons=annotation.key_buttons,
            poco_tree_snapshot=perception.poco_tree_markdown,
            discovered_at=time.time(),
        )
        self._graph.add_node(node)
        self._storage.save_node(node)
        self._report.nodes_discovered += 1
        logger.info("发现页面：%s（hash=%s，深度=%d）", node.page_name, page_hash, depth)

        if depth >= max_depth:
            return

        for button in perception.interactive_nodes:
            if self._is_back_button(button.name):
                continue

            try:
                result = self._device.click_poco(button.poco_path)
                if not result.success:
                    continue

                await asyncio.sleep(1.0)

                new_perception = self._tree_extractor.extract()
                new_hash = new_perception.page_hash

                if new_hash != page_hash:
                    edge = GraphEdge(
                        source_id=page_hash,
                        target_id=new_hash,
                        actions=[EdgeAction("poco_click", button.poco_path, {})],
                        discovered_at=time.time(),
                    )
                    self._graph.add_edge(edge)
                    self._storage.save_edge(edge)
                    self._report.edges_discovered += 1

                    await self._explore_node(depth + 1, max_depth)

                    await self._navigate_back()
                    await asyncio.sleep(0.5)

            except Exception as exc:
                error_msg = f"探索按钮 {button.name} 时出错：{exc}"
                logger.warning(error_msg)
                self._report.errors.append(error_msg)

    async def _navigate_back(self) -> None:
        tree = self._device.get_poco_tree()
        for node in tree:
            if self._is_back_button(node.name) and node.visible:
                self._device.click_poco(node.poco_path)
                return
        self._device.click((0.05, 0.05))

    def _is_back_button(self, name: str) -> bool:
        name_lower = name.lower()
        return any(pat in name_lower for pat in BACK_BUTTON_PATTERNS)
