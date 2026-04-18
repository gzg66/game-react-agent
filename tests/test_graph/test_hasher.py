"""Tests for page hashing."""

from game_agent.device.base import PocoNode
from game_agent.graph.hasher import PageHasher


def test_hash_stability():
    hasher = PageHasher()
    nodes = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_b", type="Button", visible=True),
    ]
    h1 = hasher.compute(nodes)
    h2 = hasher.compute(nodes)
    assert h1 == h2


def test_hash_ignores_order():
    hasher = PageHasher()
    nodes_ab = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_b", type="Button", visible=True),
    ]
    nodes_ba = [
        PocoNode(name="btn_b", type="Button", visible=True),
        PocoNode(name="btn_a", type="Button", visible=True),
    ]
    assert hasher.compute(nodes_ab) == hasher.compute(nodes_ba)


def test_hash_ignores_text():
    hasher = PageHasher()
    nodes1 = [PocoNode(name="btn_a", type="Button", text="100", visible=True)]
    nodes2 = [PocoNode(name="btn_a", type="Button", text="999", visible=True)]
    assert hasher.compute(nodes1) == hasher.compute(nodes2)


def test_hash_differs_for_different_structure():
    hasher = PageHasher()
    nodes1 = [PocoNode(name="btn_a", type="Button", visible=True)]
    nodes2 = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_b", type="Button", visible=True),
    ]
    assert hasher.compute(nodes1) != hasher.compute(nodes2)


def test_hash_ignores_invisible():
    hasher = PageHasher()
    nodes = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_hidden", type="Button", visible=False),
    ]
    nodes_only_visible = [PocoNode(name="btn_a", type="Button", visible=True)]
    assert hasher.compute(nodes) == hasher.compute(nodes_only_visible)


def test_hash_length():
    hasher = PageHasher()
    nodes = [PocoNode(name="btn_a", type="Button", visible=True)]
    h = hasher.compute(nodes)
    assert len(h) == 16
