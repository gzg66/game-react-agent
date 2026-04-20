"""Tests for Poco tree extraction and perception."""

from game_agent.config import PerceptionConfig
from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.graph.hasher import PageHasher
from game_agent.perception.poco_tree import PocoTreeExtractor


def _make_device(nodes: list[PocoNode]) -> MockDevice:
    device = MockDevice()
    device.load_scenario([MockScreen(poco_tree=nodes)])
    return device


def test_extract_filters_invisible_nodes():
    nodes = [
        PocoNode(name="btn_visible", type="Button", visible=True, pos=(0.5, 0.5)),
        PocoNode(name="btn_hidden", type="Button", visible=False, pos=(0.1, 0.1)),
    ]
    device = _make_device(nodes)
    extractor = PocoTreeExtractor(device, PerceptionConfig())
    result = extractor.extract()
    assert len(result.interactive_nodes) == 1
    assert result.interactive_nodes[0].name == "btn_visible"


def test_extract_filters_non_interactive():
    nodes = [
        PocoNode(name="btn_action", type="Button", visible=True, pos=(0.5, 0.5)),
        PocoNode(name="bg_image", type="Image", visible=True, pos=(0.5, 0.5)),
    ]
    device = _make_device(nodes)
    extractor = PocoTreeExtractor(device, PerceptionConfig())
    result = extractor.extract()
    assert len(result.interactive_nodes) == 1


def test_markdown_format():
    nodes = [
        PocoNode(name="btn_hero", type="Button", text="Heroes", visible=True, pos=(0.1, 0.9)),
    ]
    device = _make_device(nodes)
    extractor = PocoTreeExtractor(device, PerceptionConfig())
    result = extractor.extract()
    assert '- [Button] btn_hero "Heroes"' in result.poco_tree_markdown
    assert "(位置: 0.10, 0.90)" in result.poco_tree_markdown


def test_page_hash_stability():
    nodes = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_b", type="Button", visible=True),
    ]
    device = _make_device(nodes)
    extractor = PocoTreeExtractor(device, PerceptionConfig())
    hash1 = extractor.extract().page_hash
    hash2 = extractor.extract().page_hash
    assert hash1 == hash2


def test_page_hash_ignores_text_content():
    nodes1 = [PocoNode(name="btn_a", type="Button", text="100", visible=True)]
    nodes2 = [PocoNode(name="btn_a", type="Button", text="999", visible=True)]

    ext1 = PocoTreeExtractor(_make_device(nodes1), PerceptionConfig())
    ext2 = PocoTreeExtractor(_make_device(nodes2), PerceptionConfig())

    assert ext1.extract().page_hash == ext2.extract().page_hash


def test_different_structures_produce_different_hashes():
    nodes1 = [PocoNode(name="btn_a", type="Button", visible=True)]
    nodes2 = [
        PocoNode(name="btn_a", type="Button", visible=True),
        PocoNode(name="btn_b", type="Button", visible=True),
    ]

    ext1 = PocoTreeExtractor(_make_device(nodes1), PerceptionConfig())
    ext2 = PocoTreeExtractor(_make_device(nodes2), PerceptionConfig())

    assert ext1.extract().page_hash != ext2.extract().page_hash


def test_page_hash_matches_page_hasher():
    nodes = [
        PocoNode(name="btn_start", type="Button", visible=True),
        PocoNode(name="tab_main", type="Tab", visible=True),
        PocoNode(name="txt_gold", type="Text", visible=True),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())
    hasher = PageHasher()

    assert extractor.extract().page_hash == hasher.compute(nodes)
