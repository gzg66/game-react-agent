"""Tests for graph data models."""

from game_agent.graph.models import EdgeAction, GraphEdge, GraphNode, UIStateGraph


def test_graph_add_and_get_node():
    graph = UIStateGraph()
    node = GraphNode(node_id="abc", page_name="main_city")
    graph.add_node(node)
    assert graph.has_node("abc")
    assert graph.get_node("abc").page_name == "main_city"
    assert graph.node_count() == 1


def test_graph_missing_node():
    graph = UIStateGraph()
    assert graph.has_node("missing") is False
    assert graph.get_node("missing") is None


def test_graph_add_edge_and_neighbors():
    graph = UIStateGraph()
    graph.add_node(GraphNode(node_id="a", page_name="page_a"))
    graph.add_node(GraphNode(node_id="b", page_name="page_b"))
    edge = GraphEdge(
        source_id="a",
        target_id="b",
        actions=[EdgeAction(action_type="poco_click", target="btn_go")],
    )
    graph.add_edge(edge)
    neighbors = graph.neighbors("a")
    assert len(neighbors) == 1
    assert neighbors[0][1].page_name == "page_b"
    assert graph.edge_count() == 1


def test_graph_all_nodes_and_edges():
    graph = UIStateGraph()
    graph.add_node(GraphNode(node_id="a", page_name="a"))
    graph.add_node(GraphNode(node_id="b", page_name="b"))
    graph.add_edge(GraphEdge(source_id="a", target_id="b"))
    graph.add_edge(GraphEdge(source_id="b", target_id="a"))
    assert len(graph.all_nodes()) == 2
    assert len(graph.all_edges()) == 2
