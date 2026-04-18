"""Abstract perception provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from game_agent.perception.state import L1Perception, L2Perception, PerceptionState, UIDiff


class PerceptionProvider(ABC):
    """Interface for capturing and comparing game screen states."""

    @abstractmethod
    def capture_l1(self) -> L1Perception:
        """L1: text-only Poco tree extraction (fast, low cost)."""

    @abstractmethod
    def capture_l2(self) -> L2Perception:
        """L2: full multi-modal with screenshot (slow, high cost)."""

    @abstractmethod
    def get_current_page_hash(self) -> str:
        """Return the page hash of the current screen."""

    @abstractmethod
    def compute_diff(self, prev: PerceptionState, curr: PerceptionState) -> UIDiff:
        """Compare two perception snapshots."""

    @abstractmethod
    def to_state(self, perception: L1Perception) -> PerceptionState:
        """Convert a perception capture to a minimal state for diff tracking."""
