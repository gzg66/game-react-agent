"""Deterministic mock device for testing without a real device."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from game_agent.device.base import DeviceController, PocoNode, TouchResult

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass
class MockScreen:
    """A single screen state in a mock scenario."""

    poco_tree: list[PocoNode] = field(default_factory=list)
    screenshot_bytes: bytes = TINY_PNG
    screen_size: tuple[int, int] = (720, 1280)
    node_existence: dict[str, bool] = field(default_factory=dict)


class MockDevice(DeviceController):
    """Scriptable mock that replays a sequence of screen states.

    Usage:
        device = MockDevice()
        device.load_scenario([screen1, screen2, ...])
        # Each mutating action advances to the next screen.
    """

    def __init__(self) -> None:
        self._screens: list[MockScreen] = [MockScreen()]
        self._cursor: int = 0
        self.call_log: list[dict] = []

    @property
    def current_screen(self) -> MockScreen:
        return self._screens[min(self._cursor, len(self._screens) - 1)]

    def load_scenario(self, screens: list[MockScreen]) -> None:
        self._screens = screens if screens else [MockScreen()]
        self._cursor = 0
        self.call_log.clear()

    def advance(self) -> None:
        if self._cursor < len(self._screens) - 1:
            self._cursor += 1

    def _log(self, method: str, **kwargs) -> None:
        self.call_log.append({"method": method, "timestamp": time.time(), **kwargs})

    def click(self, pos: tuple[float, float]) -> TouchResult:
        self._log("click", pos=pos)
        self.advance()
        return TouchResult(success=True, timestamp=time.time(), details=f"click({pos})")

    def click_poco(self, poco_path: str) -> TouchResult:
        self._log("click_poco", poco_path=poco_path)
        exists = self.current_screen.node_existence.get(poco_path, True)
        if exists:
            self.advance()
        return TouchResult(
            success=exists,
            timestamp=time.time(),
            details=f"click_poco({poco_path}) -> {'ok' if exists else 'not found'}",
        )

    def swipe(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        duration: float = 0.5,
    ) -> TouchResult:
        self._log("swipe", start=start, end=end, duration=duration)
        self.advance()
        return TouchResult(success=True, timestamp=time.time(), details=f"swipe({start}->{end})")

    def long_press(self, pos: tuple[float, float], duration: float = 1.0) -> TouchResult:
        self._log("long_press", pos=pos, duration=duration)
        self.advance()
        return TouchResult(success=True, timestamp=time.time(), details=f"long_press({pos})")

    def wait_for_node(self, poco_path: str, timeout: float = 10.0) -> bool:
        self._log("wait_for_node", poco_path=poco_path, timeout=timeout)
        return self.current_screen.node_existence.get(poco_path, True)

    def node_exists(self, poco_path: str) -> bool:
        self._log("node_exists", poco_path=poco_path)
        return self.current_screen.node_existence.get(poco_path, False)

    def get_poco_tree(self) -> list[PocoNode]:
        self._log("get_poco_tree")
        return list(self.current_screen.poco_tree)

    def screenshot(self) -> bytes:
        self._log("screenshot")
        return self.current_screen.screenshot_bytes

    def get_screen_size(self) -> tuple[int, int]:
        self._log("get_screen_size")
        return self.current_screen.screen_size

    def dump_hierarchy(self) -> dict[str, Any] | None:
        self._log("dump_hierarchy")
        tree = self.current_screen.poco_tree
        if not tree:
            return None
        children = []
        for node in tree:
            children.append({
                "name": node.name,
                "payload": {
                    "name": node.name,
                    "type": node.type,
                    "text": node.text,
                    "visible": node.visible,
                    "pos": list(node.pos),
                    "size": list(node.size),
                    "clickable": node.type == "Button",
                },
                "children": [],
            })
        return {
            "name": "Root",
            "payload": {"name": "Root", "type": "Node", "visible": True},
            "children": children,
        }

    def press_back(self) -> None:
        self._log("press_back")
        self.advance()

    def app_is_running(self, package_name: str) -> bool:
        self._log("app_is_running", package_name=package_name)
        return True

    def force_stop(self, package_name: str) -> None:
        self._log("force_stop", package_name=package_name)

    def start_app(self, activity: str) -> None:
        self._log("start_app", activity=activity)

    def snapshot_to_file(self, save_path: str) -> bool:
        self._log("snapshot_to_file", save_path=save_path)
        try:
            with open(save_path, "wb") as f:
                f.write(self.current_screen.screenshot_bytes)
            return True
        except OSError:
            return False
