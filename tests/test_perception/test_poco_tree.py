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
    assert '[Button] btn_hero text="Heroes"' in result.poco_tree_markdown
    assert "pos=(0.10, 0.90)" in result.poco_tree_markdown


def test_extract_visible_tree_markdown_keeps_all_visible_nodes():
    nodes = [
        PocoNode(
            name="btn_hero",
            type="Button",
            text="Heroes",
            visible=True,
            pos=(0.1, 0.9),
            size=(0.2, 0.1),
            children_count=1,
            poco_path="Root > MainPanel > btn_hero",
        ),
        PocoNode(
            name="bg_image",
            type="Image",
            visible=True,
            pos=(0.5, 0.5),
            size=(1.0, 1.0),
            poco_path="Root > MainPanel > bg_image",
        ),
        PocoNode(name="hidden_tip", type="Text", visible=False, pos=(0.2, 0.2)),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())

    markdown = extractor.extract_visible_tree_markdown()

    assert "[Button] btn_hero" in markdown
    assert "[Image] bg_image" in markdown
    assert "hidden_tip" not in markdown
    assert "path=Root > MainPanel > btn_hero" in markdown


def test_extract_uses_full_visible_tree_markdown():
    nodes = [
        PocoNode(name="btn_action", type="Button", visible=True, poco_path="Root > btn_action"),
        PocoNode(name="bg_image", type="Image", visible=True, poco_path="Root > bg_image"),
        PocoNode(name="hidden_tip", type="Text", visible=False, poco_path="Root > hidden_tip"),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())

    result = extractor.extract()

    assert "[Button] btn_action" in result.poco_tree_markdown
    assert "[Image] bg_image" in result.poco_tree_markdown
    assert "hidden_tip" not in result.poco_tree_markdown


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


def test_detect_guide_returns_node_when_present():
    nodes = [
        PocoNode(name="btn_start", type="Button", visible=True, pos=(0.5, 0.5)),
        PocoNode(
            name="GLoader3D",
            type="GLoader3D",
            visible=True,
            pos=(0.3, 0.7),
            poco_path="Scene > GRoot > Guide > Container > GLoader3D",
        ),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())
    result = extractor.extract()
    assert result.guide_node is not None
    assert result.guide_node.name == "GLoader3D"
    assert result.guide_node.pos == (0.3, 0.7)
    assert result.guide_node in result.interactive_nodes


def test_detect_guide_returns_none_when_absent():
    nodes = [
        PocoNode(name="btn_start", type="Button", visible=True, pos=(0.5, 0.5)),
        PocoNode(name="bg_image", type="Image", visible=True, pos=(0.5, 0.5)),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())
    result = extractor.extract()
    assert result.guide_node is None


def test_detect_guide_ignores_gloader3d_outside_guide_path():
    nodes = [
        PocoNode(
            name="GLoader3D",
            type="GLoader3D",
            visible=True,
            pos=(0.5, 0.5),
            poco_path="Scene > GRoot > Container > GLoader3D",
        ),
    ]
    extractor = PocoTreeExtractor(_make_device(nodes), PerceptionConfig())
    result = extractor.extract()
    assert result.guide_node is None
