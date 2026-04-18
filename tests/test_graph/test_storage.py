"""Tests for SQLite graph storage."""

import json
import tempfile
from pathlib import Path

from game_agent.graph.models import EdgeAction, GraphEdge, GraphNode
from game_agent.graph.storage import GraphStorage


def test_save_and_load_node():
    storage = GraphStorage(":memory:")
    node = GraphNode(
        node_id="abc123",
        page_name="main_city",
        page_description="Main hub",
        page_category="navigation",
        key_buttons=["btn_hero", "btn_shop"],
        discovered_at=1000.0,
        visit_count=3,
    )
    storage.save_node(node)
    graph = storage.load_graph()
    loaded = graph.get_node("abc123")
    assert loaded is not None
    assert loaded.page_name == "main_city"
    assert loaded.key_buttons == ["btn_hero", "btn_shop"]
    assert loaded.visit_count == 3


def test_save_and_load_edge():
    storage = GraphStorage(":memory:")
    storage.save_node(GraphNode(node_id="a", page_name="a"))
    storage.save_node(GraphNode(node_id="b", page_name="b"))
    edge = GraphEdge(
        source_id="a",
        target_id="b",
        actions=[
            EdgeAction(action_type="poco_click", target="btn_go", params={"delay": 0.5}),
            EdgeAction(action_type="swipe", target="(0.5,0.5)", params={"direction": "up"}),
        ],
        discovered_at=2000.0,
    )
    storage.save_edge(edge)
    graph = storage.load_graph()
    edges = graph.get_edges_from("a")
    assert len(edges) == 1
    assert len(edges[0].actions) == 2
    assert edges[0].actions[0].action_type == "poco_click"
    assert edges[0].actions[1].params["direction"] == "up"


def test_upsert_node():
    storage = GraphStorage(":memory:")
    node = GraphNode(node_id="abc", page_name="old_name")
    storage.save_node(node)
    node.page_name = "new_name"
    node.visit_count = 10
    storage.save_node(node)
    graph = storage.load_graph()
    assert graph.node_count() == 1
    assert graph.get_node("abc").page_name == "new_name"


def test_export_import_json():
    storage = GraphStorage(":memory:")
    storage.save_node(GraphNode(node_id="a", page_name="page_a"))
    storage.save_node(GraphNode(node_id="b", page_name="page_b"))
    storage.save_edge(GraphEdge(
        source_id="a", target_id="b",
        actions=[EdgeAction("poco_click", "btn_go")],
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = str(Path(tmpdir) / "graph.json")
        storage.export_json(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

        storage2 = GraphStorage(":memory:")
        graph = storage2.import_json(json_path)
        assert graph.node_count() == 2
        assert graph.edge_count() == 1


def test_load_sample_fixture(in_memory_storage, sample_graph):
    assert sample_graph.node_count() == 5
    assert sample_graph.edge_count() == 8
