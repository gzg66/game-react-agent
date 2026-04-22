"""UI spatial occlusion detection and scroll-to-reveal computation.

Provides geometric analysis of Poco UI tree nodes to determine whether a
target element is visually blocked by higher-z-order siblings, and if so,
computes the swipe vector required to scroll the target into a safe
(unoccluded, on-screen) region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from game_agent.device.base import PocoNode

logger = logging.getLogger(__name__)

OCCLUSION_COVERAGE_THRESHOLD = 0.15
LARGE_POPUP_AREA_THRESHOLD = 0.6

_STRUCTURAL_OVERLAY_KEYWORDS = frozenset({
    "clickeffect", "effect", "guide", "softguide",
    "ggraph", "background", "mask", "loading", "transition",
})


def _is_structural_overlay(node: PocoNode) -> bool:
    """Return True if *node* is a structural UI overlay rather than a popup."""
    name_lower = node.name.lower()
    if any(kw in name_lower for kw in _STRUCTURAL_OVERLAY_KEYWORDS):
        return True
    alpha = node.payload.get("alpha", 1.0)
    if isinstance(alpha, (int, float)) and 0.01 < alpha < 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in normalised [0, 1] screen space."""

    left: float
    right: float
    top: float
    bottom: float

    def intersects(self, other: BoundingBox) -> bool:
        return (
            self.left < other.right
            and self.right > other.left
            and self.top < other.bottom
            and self.bottom > other.top
        )

    def intersection_area(self, other: BoundingBox) -> float:
        if not self.intersects(other):
            return 0.0
        ix = min(self.right, other.right) - max(self.left, other.left)
        iy = min(self.bottom, other.bottom) - max(self.top, other.top)
        return max(0.0, ix) * max(0.0, iy)

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)

    @property
    def center_x(self) -> float:
        return (self.left + self.right) / 2

    @property
    def center_y(self) -> float:
        return (self.top + self.bottom) / 2

    def is_on_screen(self) -> bool:
        return self.right > 0.0 and self.left < 1.0 and self.bottom > 0.0 and self.top < 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_bbox(node: PocoNode) -> BoundingBox:
    """Compute AABB from *pos*, *size* and optional *anchorPoint*."""
    px, py = node.pos
    w, h = node.size
    anchor = node.payload.get("anchorPoint", [0.5, 0.5])
    if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
        ax, ay = float(anchor[0]), float(anchor[1])
    else:
        ax, ay = 0.5, 0.5

    return BoundingBox(
        left=px - w * ax,
        right=px + w * (1.0 - ax),
        top=py - h * ay,
        bottom=py + h * (1.0 - ay),
    )


def get_global_zorder(node: PocoNode) -> int:
    z = node.payload.get("zOrders", {})
    if isinstance(z, dict):
        return z.get("global", 0)
    return 0


# ---------------------------------------------------------------------------
# Occlusion analysis
# ---------------------------------------------------------------------------

@dataclass
class OcclusionResult:
    is_occluded: bool
    is_off_screen: bool
    is_large_popup: bool
    occluders: list[PocoNode] = field(default_factory=list)
    target_bbox: BoundingBox | None = None
    coverage_ratio: float = 0.0


def check_occlusion(
    target: PocoNode,
    all_visible_nodes: list[PocoNode],
) -> OcclusionResult:
    """Determine whether *target* is occluded by higher-z-order nodes.

    Returns an ``OcclusionResult`` describing the occlusion state and the
    list of blocking nodes.
    """
    target_bbox = compute_bbox(target)

    # --- off-screen check ---
    if not target_bbox.is_on_screen():
        return OcclusionResult(
            is_occluded=True,
            is_off_screen=True,
            is_large_popup=False,
            target_bbox=target_bbox,
            coverage_ratio=1.0,
        )

    if target_bbox.area <= 0:
        return OcclusionResult(
            is_occluded=False,
            is_off_screen=False,
            is_large_popup=False,
            target_bbox=target_bbox,
        )

    target_z = get_global_zorder(target)
    occluders: list[PocoNode] = []
    total_coverage = 0.0

    for node in all_visible_nodes:
        if node.poco_path == target.poco_path:
            continue
        if not node.visible:
            continue
        w, h = node.size
        if w <= 0 or h <= 0:
            continue
        alpha = node.payload.get("alpha", 1.0)
        if isinstance(alpha, (int, float)) and alpha <= 0.01:
            continue

        node_z = get_global_zorder(node)
        if node_z <= target_z:
            continue

        # skip tree ancestors / descendants
        if (
            target.poco_path.startswith(node.poco_path + " > ")
            or node.poco_path.startswith(target.poco_path + " > ")
        ):
            continue

        node_bbox = compute_bbox(node)
        intersection = target_bbox.intersection_area(node_bbox)
        if intersection > 0:
            coverage = intersection / target_bbox.area
            total_coverage += coverage
            occluders.append(node)

    total_coverage = min(total_coverage, 1.0)

    is_large_popup = any(
        compute_bbox(o).area >= LARGE_POPUP_AREA_THRESHOLD
        and not _is_structural_overlay(o)
        for o in occluders
    )
    is_occluded = total_coverage > OCCLUSION_COVERAGE_THRESHOLD

    return OcclusionResult(
        is_occluded=is_occluded,
        is_off_screen=False,
        is_large_popup=is_large_popup,
        occluders=occluders,
        target_bbox=target_bbox,
        coverage_ratio=total_coverage,
    )


# ---------------------------------------------------------------------------
# Scroll vector computation
# ---------------------------------------------------------------------------

@dataclass
class ScrollVector:
    start: tuple[float, float]
    end: tuple[float, float]
    description: str


def _clamp(v: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, v))


def compute_scroll_to_reveal(
    target: PocoNode,
    occlusion: OcclusionResult,
    scroll_step: float = 0.25,
) -> ScrollVector | None:
    """Return a swipe vector that should reveal *target*.

    The swipe start is placed away from the occluders (inside the scrollable
    content area) and the direction moves the content so *target* shifts
    towards the safe (unoccluded) screen region.

    Returns ``None`` when no scroll is needed or computable.
    """
    if not occlusion.is_occluded or occlusion.is_large_popup:
        return None

    target_bbox = occlusion.target_bbox
    if target_bbox is None:
        return None

    ty = target_bbox.center_y
    tx = target_bbox.center_x

    # --- off-screen: scroll towards the target ---
    if occlusion.is_off_screen:
        if ty >= 1.0:
            sy = _clamp(0.65)
            return ScrollVector(
                start=(0.5, sy), end=(0.5, _clamp(sy - scroll_step)),
                description=f"目标在屏幕下方(y={ty:.2f})，向上滑动",
            )
        if ty <= 0.0:
            sy = _clamp(0.35)
            return ScrollVector(
                start=(0.5, sy), end=(0.5, _clamp(sy + scroll_step)),
                description=f"目标在屏幕上方(y={ty:.2f})，向下滑动",
            )
        if tx >= 1.0:
            sx = _clamp(0.65)
            return ScrollVector(
                start=(sx, 0.5), end=(_clamp(sx - scroll_step), 0.5),
                description=f"目标在屏幕右侧(x={tx:.2f})，向左滑动",
            )
        sx = _clamp(0.35)
        return ScrollVector(
            start=(sx, 0.5), end=(_clamp(sx + scroll_step), 0.5),
            description=f"目标在屏幕左侧(x={tx:.2f})，向右滑动",
        )

    # --- on-screen but occluded: scroll away from occluder mass ---
    if not occlusion.occluders:
        return None

    occ_y_avg = sum(compute_bbox(o).center_y for o in occlusion.occluders) / len(occlusion.occluders)
    occ_x_avg = sum(compute_bbox(o).center_x for o in occlusion.occluders) / len(occlusion.occluders)

    dy = abs(occ_y_avg - 0.5)
    dx = abs(occ_x_avg - 0.5)

    if dy >= dx:
        if occ_y_avg > 0.5:
            sy = _clamp(min(0.6, occ_y_avg - 0.2))
            return ScrollVector(
                start=(0.5, sy), end=(0.5, _clamp(sy - scroll_step)),
                description=f"遮挡物集中在屏幕下方(y={occ_y_avg:.2f})，向上滑动",
            )
        else:
            sy = _clamp(max(0.4, occ_y_avg + 0.2))
            return ScrollVector(
                start=(0.5, sy), end=(0.5, _clamp(sy + scroll_step)),
                description=f"遮挡物集中在屏幕上方(y={occ_y_avg:.2f})，向下滑动",
            )
    else:
        if occ_x_avg > 0.5:
            sx = _clamp(min(0.6, occ_x_avg - 0.2))
            return ScrollVector(
                start=(sx, 0.5), end=(_clamp(sx - scroll_step), 0.5),
                description=f"遮挡物集中在屏幕右侧(x={occ_x_avg:.2f})，向左滑动",
            )
        else:
            sx = _clamp(max(0.4, occ_x_avg + 0.2))
            return ScrollVector(
                start=(sx, 0.5), end=(_clamp(sx + scroll_step), 0.5),
                description=f"遮挡物集中在屏幕左侧(x={occ_x_avg:.2f})，向右滑动",
            )
