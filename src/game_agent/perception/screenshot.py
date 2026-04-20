"""L2 perception: screenshot capture and encoding."""

from __future__ import annotations

import base64
import time

from game_agent.config import PerceptionConfig
from game_agent.device.base import DeviceController
from game_agent.exceptions import PerceptionError
from game_agent.perception.poco_tree import PocoTreeExtractor
from game_agent.perception.state import L2Perception


class ScreenshotCapture:
    """Captures screenshots and produces L2 multi-modal perception."""

    def __init__(
        self,
        device: DeviceController,
        tree_extractor: PocoTreeExtractor,
        config: PerceptionConfig,
    ) -> None:
        self._device = device
        self._tree_extractor = tree_extractor
        self._config = config

    def capture(self) -> L2Perception:
        l1 = self._tree_extractor.extract()
        raw_screenshot = self._device.screenshot()
        if not raw_screenshot:
            raise PerceptionError("截图为空，无法执行 L2 感知")
        resized = self._resize_screenshot(raw_screenshot)
        if not resized:
            raise PerceptionError("截图缩放后为空，无法执行 L2 感知")
        b64 = base64.b64encode(resized).decode("ascii")

        return L2Perception(
            timestamp=time.time(),
            poco_tree_markdown=l1.poco_tree_markdown,
            interactive_nodes=l1.interactive_nodes,
            page_hash=l1.page_hash,
            screenshot_b64=b64,
            screenshot_raw=resized,
        )

    def _resize_screenshot(self, png_bytes: bytes) -> bytes:
        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(png_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return png_bytes
            target_w, target_h = self._config.screenshot_resize
            resized = cv2.resize(img, (target_w, target_h))
            _, buf = cv2.imencode(".png", resized)
            return buf.tobytes()
        except ImportError:
            return png_bytes
