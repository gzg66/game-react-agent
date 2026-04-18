"""Graph data structures for the UI state machine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EdgeAction:
    """Single atomic action in a transition sequence."""

    action_type: str
    target: str
    params: dict = field(default_factory=dict)


@dataclass
class GraphNode:
    """A page/screen in the UI state machine."""

    node_id: str
    page_name: str
    page_description: str = ""
    page_category: str = "未知"
    key_buttons: list[str] = field(default_factory=list)
    poco_tree_snapshot: str = ""
    screenshot_path: str | None = None
    discovered_at: float = 0.0
    visit_count: int = 0


@dataclass
class GraphEdge:
    """A transition between two pages."""

    source_id: str
    target_id: str
    actions: list[EdgeAction] = field(default_factory=list)
    discovered_at: float = 0.0
    success_count: int = 0
    failure_count: int = 0


class UIStateGraph:
    """In-memory graph with fast lookup. Backed by SQLite for persistence."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._adjacency: dict[str, list[GraphEdge]] = {}

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node
        if node.node_id not in self._adjacency:
            self._adjacency[node.node_id] = []

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self._adjacency:
            self._adjacency[edge.source_id] = []
        self._adjacency[edge.source_id].append(edge)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def neighbors(self, node_id: str) -> list[tuple[GraphEdge, GraphNode]]:
        result = []
        for edge in self._adjacency.get(node_id, []):
            target = self._nodes.get(edge.target_id)
            if target is not None:
                result.append((edge, target))
        return result

    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        return list(self._adjacency.get(node_id, []))

    def all_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[GraphEdge]:
        edges = []
        for edge_list in self._adjacency.values():
            edges.extend(edge_list)
        return edges

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adjacency.values())
