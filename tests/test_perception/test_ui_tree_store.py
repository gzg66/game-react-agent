"""Tests for UITreeStore."""

import json

from game_agent.device.base import PocoNode
from game_agent.perception.ui_tree_store import UITreeStore


def _sample_nodes():
    return [
        PocoNode(
            name="Scene", type="Scene", visible=True,
            pos=(0.0, 1.0), size=(0.0, 0.0), poco_path="Scene",
        ),
        PocoNode(
            name="btnLogin", type="Button", visible=True,
            pos=(0.67, 0.57), size=(0.2, 0.1),
            poco_path="Scene > GRoot > Container > btnLogin",
        ),
        PocoNode(
            name="btnRegister", type="Button", visible=True,
            pos=(0.34, 0.57), size=(0.2, 0.1),
            poco_path="Scene > GRoot > Container > btnRegister",
        ),
        PocoNode(
            name="inputAccount", type="InputField", visible=True,
            pos=(0.32, 0.4), size=(0.3, 0.05),
            poco_path="Scene > GRoot > Container > LoginWindow > inputAccount",
        ),
    ]


def test_resolve_exact_name(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    store.update("page_abc", _sample_nodes())
    node = store.resolve("btnLogin")
    assert node is not None
    assert node.name == "btnLogin"
    assert node.pos == (0.67, 0.57)


def test_resolve_by_path_suffix(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    nodes = [
        PocoNode(
            name="Container", type="Node", visible=True,
            pos=(0.0, 0.0), size=(1.0, 1.0),
            poco_path="Scene > GRoot > Container",
        ),
        PocoNode(
            name="root", type="Component", visible=True,
            pos=(0.5, 0.5), size=(1.0, 1.0),
            poco_path="Scene > GRoot > Container > LoginWindow > root",
        ),
    ]
    store.update("page_abc", nodes)
    node = store.resolve("LoginWindow")
    assert node is None

    node = store.resolve("root")
    assert node is not None
    assert node.poco_path == "Scene > GRoot > Container > LoginWindow > root"


def test_resolve_not_found(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    store.update("page_abc", _sample_nodes())
    assert store.resolve("nonexistent") is None


def test_save_to_disk(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    nodes = _sample_nodes()
    store.update("abc123", nodes)

    json_path = tmp_path / "trees" / "abc123.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == len(nodes)
    assert data[1]["name"] == "btnLogin"
    assert data[1]["poco_path"] == "Scene > GRoot > Container > btnLogin"


def test_save_idempotent(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    store.update("abc123", _sample_nodes())
    json_path = tmp_path / "trees" / "abc123.json"
    mtime1 = json_path.stat().st_mtime_ns

    store.update("abc123", [])
    mtime2 = json_path.stat().st_mtime_ns
    assert mtime1 == mtime2


def test_load_from_disk(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    store.update("abc123", _sample_nodes())

    store2 = UITreeStore(str(tmp_path / "trees"))
    loaded = store2.load("abc123")
    assert loaded is not None
    assert len(loaded) == 4
    assert loaded[1].name == "btnLogin"
    assert loaded[1].pos == (0.67, 0.57)


def test_load_nonexistent(tmp_path):
    store = UITreeStore(str(tmp_path / "trees"))
    assert store.load("no_such_page") is None
