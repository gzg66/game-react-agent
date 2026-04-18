"""BFS shortest path navigation on the cached UI state graph."""

from __future__ import annotations

import logging
from collections import deque

from game_agent.graph.models import GraphEdge, UIStateGraph

logger = logging.getLogger(__name__)


class GraphNavigator:
    """Finds shortest paths through the UI state machine graph."""

    def __init__(self, graph: UIStateGraph) -> None:
        self._graph = graph

    def shortest_path(self, source_id: str, target_id: str) -> list[GraphEdge] | None:
        """BFS shortest path from source to target. Returns ordered edge list or None."""
        if source_id == target_id:
            return []
        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return None

        visited: set[str] = {source_id}
        queue: deque[tuple[str, list[GraphEdge]]] = deque([(source_id, [])])

        while queue:
            current_id, path = queue.popleft()
            for edge in self._graph.get_edges_from(current_id):
                if edge.target_id in visited:
                    continue
                new_path = path + [edge]
                if edge.target_id == target_id:
                    return new_path
                visited.add(edge.target_id)
                queue.append((edge.target_id, new_path))

        return None

    def reachable_from(self, source_id: str) -> set[str]:
        """Return all node IDs reachable from source via BFS."""
        if not self._graph.has_node(source_id):
            return set()

        visited: set[str] = {source_id}
        queue: deque[str] = deque([source_id])

        while queue:
            current_id = queue.popleft()
            for edge in self._graph.get_edges_from(current_id):
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append(edge.target_id)

        return visited

    def find_node_by_name(self, page_name: str) -> str | None:
        """Find a node ID by its semantic page name (case-insensitive partial match)."""
        page_name_lower = page_name.lower()
        for node in self._graph.all_nodes():
            if page_name_lower in node.page_name.lower():
                return node.node_id
        return None
