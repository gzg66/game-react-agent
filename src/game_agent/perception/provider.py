"""Concrete PerceptionProvider implementation wiring device to perception components."""

from __future__ import annotations

from game_agent.config import PerceptionConfig
from game_agent.device.base import DeviceController
from game_agent.perception.base import PerceptionProvider
from game_agent.perception.poco_tree import PocoTreeExtractor
from game_agent.perception.screenshot import ScreenshotCapture
from game_agent.perception.state import L1Perception, L2Perception, PerceptionState, UIDiff
from game_agent.perception.ui_diff import UIDiffCalculator


class DefaultPerceptionProvider(PerceptionProvider):
    """Standard perception provider backed by a device controller."""

    def __init__(self, device: DeviceController, config: PerceptionConfig) -> None:
        self._device = device
        self._config = config
        self._tree_extractor = PocoTreeExtractor(device, config)
        self._screenshot_capture = ScreenshotCapture(device, self._tree_extractor, config)
        self._diff_calculator = UIDiffCalculator(config)

    @property
    def tree_extractor(self) -> PocoTreeExtractor:
        return self._tree_extractor

    def capture_l1(self) -> L1Perception:
        return self._tree_extractor.extract()

    def capture_l2(self) -> L2Perception:
        return self._screenshot_capture.capture()

    def get_current_page_hash(self) -> str:
        l1 = self.capture_l1()
        return l1.page_hash

    def compute_diff(self, prev: PerceptionState, curr: PerceptionState) -> UIDiff:
        return self._diff_calculator.compute(prev, curr)

    def to_state(self, perception: L1Perception) -> PerceptionState:
        return PerceptionState(
            page_hash=perception.page_hash,
            poco_node_names=frozenset(n.name for n in perception.interactive_nodes),
            screenshot_bytes=getattr(perception, "screenshot_raw", None),
        )
