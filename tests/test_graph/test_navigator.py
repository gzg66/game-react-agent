"""Tests for graph navigation (shortest path)."""

from game_agent.graph.models import GraphEdge, GraphNode, UIStateGraph
from game_agent.graph.navigator import GraphNavigator


def _build_simple_graph() -> UIStateGraph:
    """A -> B -> C, A -> C (direct)."""
    graph = UIStateGraph()
    graph.add_node(GraphNode(node_id="a", page_name="page_a"))
    graph.add_node(GraphNode(node_id="b", page_name="page_b"))
    graph.add_node(GraphNode(node_id="c", page_name="page_c"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b"))
    graph.add_edge(GraphEdge(source_id="b", target_id="c"))
    graph.add_edge(GraphEdge(source_id="a", target_id="c"))
    return graph


def test_shortest_path_direct():
    graph = _build_simple_graph()
    nav = GraphNavigator(graph)
    path = nav.shortest_path("a", "c")
    assert path is not None
    assert len(path) == 1
    assert path[0].target_id == "c"


def test_shortest_path_indirect():
    graph = UIStateGraph()
    graph.add_node(GraphNode(node_id="a", page_name="a"))
    graph.add_node(GraphNode(node_id="b", page_name="b"))
    graph.add_node(GraphNode(node_id="c", page_name="c"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b"))
    graph.add_edge(GraphEdge(source_id="b", target_id="c"))
    nav = GraphNavigator(graph)
    path = nav.shortest_path("a", "c")
    assert path is not None
    assert len(path) == 2


def test_shortest_path_same_node():
    graph = _build_simple_graph()
    nav = GraphNavigator(graph)
    path = nav.shortest_path("a", "a")
    assert path == []


def test_shortest_path_unreachable():
    graph = UIStateGraph()
    graph.add_node(GraphNode(node_id="a", page_name="a"))
    graph.add_node(GraphNode(node_id="b", page_name="b"))
    nav = GraphNavigator(graph)
    assert nav.shortest_path("a", "b") is None


def test_shortest_path_missing_node():
    graph = _build_simple_graph()
    nav = GraphNavigator(graph)
    assert nav.shortest_path("a", "nonexistent") is None


def test_reachable_from():
    graph = _build_simple_graph()
    nav = GraphNavigator(graph)
    reachable = nav.reachable_from("a")
    assert reachable == {"a", "b", "c"}


def test_find_node_by_name():
    graph = _build_simple_graph()
    nav = GraphNavigator(graph)
    assert nav.find_node_by_name("page_b") == "b"
    assert nav.find_node_by_name("nonexistent") is None


def test_sample_graph_navigation(sample_graph):
    nav = GraphNavigator(sample_graph)
    path = nav.shortest_path("aabb1122ccdd3344", "ffeeddccbbaa9988")
    assert path is not None
    assert len(path) == 2
    reachable = nav.reachable_from("aabb1122ccdd3344")
    assert len(reachable) == 5
