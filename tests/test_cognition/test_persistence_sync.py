"""Tests for immediate persistence of runtime caches."""

from __future__ import annotations

import json

from game_agent.cognition.navigation_memory import NavigationMemory
from game_agent.cognition.page_cache import PageKnowledge, PageKnowledgeCache


def test_navigation_memory_record_autosaves_after_load(tmp_path):
    path = tmp_path / "nav_memory.json"
    memory = NavigationMemory()
    memory.load(path)

    memory.record("page_a", "btn_start", "page_b")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["page_a|btn_start"]["result_page_hash"] == "page_b"
    assert data["page_a|btn_start"]["effective"] is True


def test_page_cache_put_autosaves_after_load(tmp_path):
    path = tmp_path / "page_cache.json"
    cache = PageKnowledgeCache()
    cache.load(path)

    cache.put(
        "page_a",
        PageKnowledge(
            page_name="主城",
            page_description="游戏主界面",
            page_category="导航",
            key_buttons=["开始"],
        ),
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["page_a"]["page_name"] == "主城"
    assert data["page_a"]["page_category"] == "导航"
