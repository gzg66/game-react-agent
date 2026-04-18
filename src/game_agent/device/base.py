"""Abstract device controller interface and core data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PocoNode:
    """Flattened representation of a single Poco UI tree node."""

    name: str
    type: str
    text: str | None = None
    visible: bool = True
    pos: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float] = (0.0, 0.0)
    children_count: int = 0
    payload: dict = field(default_factory=dict)
    poco_path: str = ""


@dataclass
class TouchResult:
    """Result of a touch/gesture operation."""

    success: bool
    timestamp: float
    details: str = ""


class DeviceController(ABC):
    """Abstract interface for all device interactions.

    Every tool and perception call goes through this.
    Real impl delegates to Airtest/Poco; mock returns scripted data.
    """

    @abstractmethod
    def click(self, pos: tuple[float, float]) -> TouchResult:
        """Click at relative screen coordinates."""

    @abstractmethod
    def click_poco(self, poco_path: str) -> TouchResult:
        """Click a UI element by Poco path."""

    @abstractmethod
    def swipe(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        duration: float = 0.5,
    ) -> TouchResult:
        """Swipe from start to end coordinates."""

    @abstractmethod
    def long_press(self, pos: tuple[float, float], duration: float = 1.0) -> TouchResult:
        """Long press at position."""

    @abstractmethod
    def wait_for_node(self, poco_path: str, timeout: float = 10.0) -> bool:
        """Wait until a Poco node appears. Returns True if found."""

    @abstractmethod
    def node_exists(self, poco_path: str) -> bool:
        """Check if a Poco node currently exists."""

    @abstractmethod
    def get_poco_tree(self) -> list[PocoNode]:
        """Return the current flattened Poco UI tree."""

    @abstractmethod
    def screenshot(self) -> bytes:
        """Return PNG bytes of the current screen."""

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) in pixels."""

    @abstractmethod
    def dump_hierarchy(self) -> dict[str, Any] | None:
        """Return the raw Poco hierarchy dict, or None on failure."""

    @abstractmethod
    def press_back(self) -> None:
        """Press the Android back button."""

    @abstractmethod
    def app_is_running(self, package_name: str) -> bool:
        """Check if an app process is alive."""

    @abstractmethod
    def force_stop(self, package_name: str) -> None:
        """Force-stop an app."""

    @abstractmethod
    def start_app(self, activity: str) -> None:
        """Launch an app by activity name."""

    @abstractmethod
    def snapshot_to_file(self, save_path: str) -> bool:
        """Take a screenshot and save to a local file. Returns True on success."""
