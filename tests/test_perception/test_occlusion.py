"""Tests for UI occlusion detection and scroll computation."""

from game_agent.device.base import PocoNode
from game_agent.perception.occlusion import (
    BoundingBox,
    OcclusionResult,
    _is_structural_overlay,
    check_occlusion,
    compute_bbox,
    compute_scroll_to_reveal,
    get_global_zorder,
)


def _node(
    name="n",
    pos=(0.5, 0.5),
    size=(0.1, 0.1),
    z_global=0,
    anchor=(0.5, 0.5),
    path="Root > n",
    visible=True,
    alpha=1.0,
    text=None,
):
    return PocoNode(
        name=name,
        type="Button",
        text=text,
        visible=visible,
        pos=pos,
        size=size,
        poco_path=path,
        payload={
            "zOrders": {"local": z_global, "global": z_global},
            "anchorPoint": list(anchor),
            "alpha": alpha,
        },
    )


# ---------------------------------------------------------------------------
# BoundingBox basics
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_area(self):
        bb = BoundingBox(left=0.0, right=0.5, top=0.0, bottom=0.5)
        assert bb.area == 0.25

    def test_center(self):
        bb = BoundingBox(left=0.2, right=0.8, top=0.1, bottom=0.9)
        assert bb.center_x == 0.5
        assert bb.center_y == 0.5

    def test_intersects_true(self):
        a = BoundingBox(0.0, 0.5, 0.0, 0.5)
        b = BoundingBox(0.3, 0.8, 0.3, 0.8)
        assert a.intersects(b) is True

    def test_intersects_false(self):
        a = BoundingBox(0.0, 0.3, 0.0, 0.3)
        b = BoundingBox(0.5, 0.8, 0.5, 0.8)
        assert a.intersects(b) is False

    def test_intersection_area(self):
        a = BoundingBox(0.0, 0.6, 0.0, 0.6)
        b = BoundingBox(0.4, 1.0, 0.4, 1.0)
        assert abs(a.intersection_area(b) - 0.04) < 1e-9

    def test_is_on_screen(self):
        assert BoundingBox(0.1, 0.9, 0.1, 0.9).is_on_screen() is True
        assert BoundingBox(1.1, 1.5, 0.0, 0.5).is_on_screen() is False
        assert BoundingBox(-0.5, -0.1, 0.0, 0.5).is_on_screen() is False


# ---------------------------------------------------------------------------
# compute_bbox
# ---------------------------------------------------------------------------

class TestComputeBbox:
    def test_default_anchor(self):
        node = _node(pos=(0.5, 0.5), size=(0.2, 0.1), anchor=(0.5, 0.5))
        bb = compute_bbox(node)
        assert abs(bb.left - 0.4) < 1e-9
        assert abs(bb.right - 0.6) < 1e-9
        assert abs(bb.top - 0.45) < 1e-9
        assert abs(bb.bottom - 0.55) < 1e-9

    def test_topleft_anchor(self):
        node = _node(pos=(0.0, 0.0), size=(0.2, 0.1), anchor=(0.0, 0.0))
        bb = compute_bbox(node)
        assert abs(bb.left - 0.0) < 1e-9
        assert abs(bb.right - 0.2) < 1e-9
        assert abs(bb.top - 0.0) < 1e-9
        assert abs(bb.bottom - 0.1) < 1e-9

    def test_missing_anchor_defaults(self):
        node = PocoNode(
            name="n", type="Button", pos=(0.5, 0.5), size=(0.2, 0.2),
            poco_path="n", payload={},
        )
        bb = compute_bbox(node)
        assert abs(bb.left - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# get_global_zorder
# ---------------------------------------------------------------------------

class TestZOrder:
    def test_normal(self):
        assert get_global_zorder(_node(z_global=5)) == 5

    def test_missing(self):
        node = PocoNode(name="n", type="B", poco_path="n", payload={})
        assert get_global_zorder(node) == 0


# ---------------------------------------------------------------------------
# check_occlusion  — core scenarios
# ---------------------------------------------------------------------------

class TestCheckOcclusion:
    def test_user_example_occluded(self):
        """Reproduce the exact user example: challenge occluded by recover."""
        target = _node(
            name="GTextField", text="challenge",
            pos=(0.838, 0.845), size=(0.072, 0.029), z_global=2,
            path="Root > ScrollList > Item5 > GTextField",
        )
        occluder = _node(
            name="GTextField2", text="recover",
            pos=(0.838, 0.846), size=(0.072, 0.029), z_global=3,
            path="Root > BottomBar > GTextField2",
        )
        result = check_occlusion(target, [target, occluder])
        assert result.is_occluded is True
        assert result.is_off_screen is False
        assert result.coverage_ratio > 0.9
        assert len(result.occluders) == 1

    def test_not_occluded(self):
        target = _node(pos=(0.5, 0.5), size=(0.1, 0.1), z_global=5, path="Root > a")
        other = _node(pos=(0.5, 0.5), size=(0.1, 0.1), z_global=1, path="Root > b")
        result = check_occlusion(target, [target, other])
        assert result.is_occluded is False
        assert result.coverage_ratio == 0.0

    def test_off_screen_below(self):
        target = _node(pos=(0.5, 1.3), size=(0.2, 0.1), path="Root > off")
        result = check_occlusion(target, [target])
        assert result.is_occluded is True
        assert result.is_off_screen is True

    def test_off_screen_left(self):
        target = _node(pos=(-0.3, 0.5), size=(0.1, 0.1), path="Root > off")
        result = check_occlusion(target, [target])
        assert result.is_off_screen is True

    def test_large_popup_detected(self):
        target = _node(pos=(0.5, 0.5), size=(0.1, 0.05), z_global=1, path="Root > btn")
        popup = _node(
            name="popup", pos=(0.5, 0.5), size=(0.9, 0.9), z_global=10,
            path="Root > popup",
        )
        result = check_occlusion(target, [target, popup])
        assert result.is_occluded is True
        assert result.is_large_popup is True

    def test_transparent_node_ignored(self):
        target = _node(pos=(0.5, 0.5), size=(0.1, 0.1), z_global=1, path="Root > t")
        overlay = _node(
            pos=(0.5, 0.5), size=(0.1, 0.1), z_global=5,
            path="Root > overlay", alpha=0.0,
        )
        result = check_occlusion(target, [target, overlay])
        assert result.is_occluded is False

    def test_ancestor_not_treated_as_occluder(self):
        parent = _node(
            name="Panel", pos=(0.5, 0.5), size=(0.5, 0.5), z_global=5,
            path="Root > Panel",
        )
        child = _node(
            name="btn", pos=(0.5, 0.5), size=(0.1, 0.1), z_global=2,
            path="Root > Panel > btn",
        )
        result = check_occlusion(child, [parent, child])
        assert result.is_occluded is False

    def test_no_overlap_different_positions(self):
        target = _node(pos=(0.1, 0.1), size=(0.1, 0.1), z_global=1, path="Root > a")
        other = _node(pos=(0.9, 0.9), size=(0.1, 0.1), z_global=5, path="Root > b")
        result = check_occlusion(target, [target, other])
        assert result.is_occluded is False

    def test_zero_size_target(self):
        target = _node(pos=(0.5, 0.5), size=(0.0, 0.0), path="Root > z")
        result = check_occlusion(target, [target])
        assert result.is_occluded is False

    def test_structural_overlay_not_treated_as_popup(self):
        """ClickEffect and Guide overlays should NOT trigger is_large_popup."""
        target = _node(
            name="btnChallenge", pos=(0.84, 0.84), size=(0.07, 0.03),
            z_global=2, path="Root > Window > btnChallenge",
        )
        click_effect = _node(
            name="ClickEffect", pos=(0.5, 0.5), size=(1.0, 1.0),
            z_global=10, path="Root > ClickEffect",
        )
        result = check_occlusion(target, [target, click_effect])
        assert result.is_occluded is True
        assert result.is_large_popup is False

    def test_guide_overlay_not_treated_as_popup(self):
        target = _node(
            name="btnAction", pos=(0.5, 0.5), size=(0.1, 0.05),
            z_global=1, path="Root > btnAction",
        )
        guide = _node(
            name="SoftGuideView", pos=(0.5, 0.5), size=(1.0, 1.0),
            z_global=10, path="Root > Guide > SoftGuideView",
        )
        result = check_occlusion(target, [target, guide])
        assert result.is_occluded is True
        assert result.is_large_popup is False

    def test_semitransparent_overlay_not_treated_as_popup(self):
        target = _node(
            name="btn", pos=(0.5, 0.5), size=(0.1, 0.05),
            z_global=1, path="Root > btn",
        )
        overlay = _node(
            name="dimLayer", pos=(0.5, 0.5), size=(1.0, 1.0),
            z_global=10, path="Root > dimLayer", alpha=0.3,
        )
        result = check_occlusion(target, [target, overlay])
        assert result.is_occluded is True
        assert result.is_large_popup is False

    def test_real_popup_still_detected(self):
        """A non-structural large node should still be detected as popup."""
        target = _node(
            name="btn", pos=(0.5, 0.5), size=(0.1, 0.05),
            z_global=1, path="Root > btn",
        )
        popup = _node(
            name="DialogPanel", pos=(0.5, 0.5), size=(0.9, 0.9),
            z_global=10, path="Root > DialogPanel",
        )
        result = check_occlusion(target, [target, popup])
        assert result.is_occluded is True
        assert result.is_large_popup is True


# ---------------------------------------------------------------------------
# _is_structural_overlay
# ---------------------------------------------------------------------------

class TestIsStructuralOverlay:
    def test_click_effect(self):
        assert _is_structural_overlay(_node(name="ClickEffect")) is True

    def test_guide_node(self):
        assert _is_structural_overlay(_node(name="SoftGuideView")) is True

    def test_ggraph(self):
        assert _is_structural_overlay(_node(name="GGraph")) is True

    def test_background(self):
        assert _is_structural_overlay(_node(name="background")) is True

    def test_mask(self):
        assert _is_structural_overlay(_node(name="popupMask")) is True

    def test_semitransparent(self):
        assert _is_structural_overlay(_node(name="layer", alpha=0.3)) is True

    def test_normal_button_is_not_overlay(self):
        assert _is_structural_overlay(_node(name="btnOK")) is False

    def test_popup_panel_is_not_overlay(self):
        assert _is_structural_overlay(_node(name="DialogPanel")) is False

    def test_opaque_panel_is_not_overlay(self):
        assert _is_structural_overlay(_node(name="Panel", alpha=1.0)) is False


# ---------------------------------------------------------------------------
# compute_scroll_to_reveal
# ---------------------------------------------------------------------------

class TestComputeScroll:
    def test_not_occluded_returns_none(self):
        target = _node()
        result = OcclusionResult(
            is_occluded=False, is_off_screen=False, is_large_popup=False,
        )
        assert compute_scroll_to_reveal(target, result) is None

    def test_large_popup_returns_none(self):
        target = _node()
        result = OcclusionResult(
            is_occluded=True, is_off_screen=False, is_large_popup=True,
        )
        assert compute_scroll_to_reveal(target, result) is None

    def test_off_screen_below_scrolls_up(self):
        target = _node(pos=(0.5, 1.3), size=(0.2, 0.1))
        result = OcclusionResult(
            is_occluded=True, is_off_screen=True, is_large_popup=False,
            target_bbox=compute_bbox(target),
        )
        scroll = compute_scroll_to_reveal(target, result)
        assert scroll is not None
        assert scroll.start[1] > scroll.end[1], "should swipe upward"

    def test_off_screen_above_scrolls_down(self):
        target = _node(pos=(0.5, -0.3), size=(0.2, 0.1))
        result = OcclusionResult(
            is_occluded=True, is_off_screen=True, is_large_popup=False,
            target_bbox=compute_bbox(target),
        )
        scroll = compute_scroll_to_reveal(target, result)
        assert scroll is not None
        assert scroll.start[1] < scroll.end[1], "should swipe downward"

    def test_occluded_by_bottom_bar_scrolls_up(self):
        """User's scenario: bottom bar occludes target."""
        target = _node(pos=(0.84, 0.845), size=(0.07, 0.03), z_global=2, path="Root > a")
        occluder = _node(pos=(0.84, 0.846), size=(0.07, 0.03), z_global=3, path="Root > b")
        occlusion = check_occlusion(target, [target, occluder])
        scroll = compute_scroll_to_reveal(target, occlusion)
        assert scroll is not None
        assert scroll.start[1] > scroll.end[1], "should swipe upward to reveal"

    def test_scroll_coords_within_bounds(self):
        target = _node(pos=(0.5, 1.5), size=(0.2, 0.1))
        result = OcclusionResult(
            is_occluded=True, is_off_screen=True, is_large_popup=False,
            target_bbox=compute_bbox(target),
        )
        scroll = compute_scroll_to_reveal(target, result, scroll_step=0.5)
        assert scroll is not None
        sx, sy = scroll.start
        ex, ey = scroll.end
        assert 0.0 <= sx <= 1.0
        assert 0.0 <= sy <= 1.0
        assert 0.0 <= ex <= 1.0
        assert 0.0 <= ey <= 1.0
