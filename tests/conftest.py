"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_agent.config import (
    AppConfig,
    ContextConfig,
    DeviceConfig,
    GeminiConfig,
    GraphConfig,
    PerceptionConfig,
)
from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.graph.models import UIStateGraph
from game_agent.graph.storage import GraphStorage

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_device():
    return MockDevice()


@pytest.fixture
def sample_poco_nodes() -> list[PocoNode]:
    with open(FIXTURES_DIR / "sample_poco_tree.json", encoding="utf-8") as f:
        data = json.load(f)
    return [PocoNode(**node) for node in data]


@pytest.fixture
def mock_device_with_nodes(mock_device, sample_poco_nodes):
    screen = MockScreen(
        poco_tree=sample_poco_nodes,
        node_existence={n.name: True for n in sample_poco_nodes if n.visible},
    )
    mock_device.load_scenario([screen])
    return mock_device


@pytest.fixture
def app_config() -> AppConfig:
    return AppConfig(
        device=DeviceConfig(use_mock=True),
        gemini=GeminiConfig(api_key="test-key"),
        perception=PerceptionConfig(),
        graph=GraphConfig(db_path=":memory:"),
        context=ContextConfig(),
    )


@pytest.fixture
def in_memory_storage() -> GraphStorage:
    return GraphStorage(":memory:")


@pytest.fixture
def sample_graph(in_memory_storage) -> UIStateGraph:
    return in_memory_storage.import_json(str(FIXTURES_DIR / "sample_graph.json"))
