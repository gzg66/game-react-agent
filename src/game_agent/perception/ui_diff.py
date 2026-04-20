"""UI diff calculation using OpenCV SSIM and Poco tree set comparison."""

from __future__ import annotations

import logging

from game_agent.config import PerceptionConfig
from game_agent.perception.state import PerceptionState, UIDiff

logger = logging.getLogger(__name__)


class UIDiffCalculator:
    """Compares two perception states to detect UI changes and loading states."""

    def __init__(self, config: PerceptionConfig) -> None:
        self._config = config

    def compute(self, prev: PerceptionState, curr: PerceptionState) -> UIDiff:
        hash_changed = prev.page_hash != curr.page_hash
        ssim = self._compute_ssim(prev.screenshot_bytes, curr.screenshot_bytes)
        added = list(curr.poco_node_names - prev.poco_node_names)
        removed = list(prev.poco_node_names - curr.poco_node_names)
        has_visual_state = (
            prev.screenshot_bytes is not None and curr.screenshot_bytes is not None
        )

        is_loading = (
            has_visual_state
            and not hash_changed
            and not added
            and not removed
            and ssim > (1.0 - self._config.diff_threshold)
        )

        return UIDiff(
            hash_changed=hash_changed,
            structural_similarity=ssim,
            added_nodes=added,
            removed_nodes=removed,
            is_loading=is_loading,
        )

    def _compute_ssim(self, img_a: bytes | None, img_b: bytes | None) -> float:
        if img_a is None and img_b is None:
            return 1.0
        if img_a is None or img_b is None:
            return 0.0
        if img_a == img_b:
            return 1.0

        try:
            import cv2
            import numpy as np

            arr_a = np.frombuffer(img_a, dtype=np.uint8)
            arr_b = np.frombuffer(img_b, dtype=np.uint8)
            decoded_a = cv2.imdecode(arr_a, cv2.IMREAD_GRAYSCALE)
            decoded_b = cv2.imdecode(arr_b, cv2.IMREAD_GRAYSCALE)

            if decoded_a is None or decoded_b is None:
                return 0.0

            h, w = 256, 256
            resized_a = cv2.resize(decoded_a, (w, h))
            resized_b = cv2.resize(decoded_b, (w, h))

            result = cv2.matchTemplate(resized_a, resized_b, cv2.TM_CCOEFF_NORMED)
            return float(result[0][0])
        except ImportError:
            logger.warning("OpenCV 不可用，回退到字节级比较")
            return 1.0 if img_a == img_b else 0.0
