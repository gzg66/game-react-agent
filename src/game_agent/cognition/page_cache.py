"""Persistent page knowledge cache — caches LLM page annotations per page hash."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PageKnowledge:
    """Cached semantic knowledge about a game page."""

    page_name: str
    page_description: str
    page_category: str
    key_buttons: list[str]


class PageKnowledgeCache:
    """Caches page annotations by page_hash to avoid repeated LLM calls.

    On the first encounter of a page_hash, the annotator is called and the
    result is stored.  On subsequent encounters the cached result is returned
    directly — zero LLM cost.
    """

    def __init__(self) -> None:
        self._cache: dict[str, PageKnowledge] = {}

    def get(self, page_hash: str) -> PageKnowledge | None:
        return self._cache.get(page_hash)

    def put(self, page_hash: str, knowledge: PageKnowledge) -> None:
        self._cache[page_hash] = knowledge

    def has(self, page_hash: str) -> bool:
        return page_hash in self._cache

    def to_prompt_text(self, page_hash: str) -> str:
        k = self._cache.get(page_hash)
        if k is None:
            return ""
        return (
            f"当前页面：「{k.page_name}」（{k.page_category}）— {k.page_description}"
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for page_hash, knowledge in self._cache.items():
            data[page_hash] = asdict(knowledge)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("页面知识缓存已保存到 %s（%d 个页面）", path, len(data))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            logger.info("页面知识缓存文件不存在，从空状态开始：%s", path)
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for page_hash, val in raw.items():
                self._cache[page_hash] = PageKnowledge(**val)
            logger.info("已加载页面知识缓存：%d 个页面", len(self._cache))
        except Exception as exc:
            logger.warning("加载页面知识缓存失败，从空状态开始：%s", exc)

    def __len__(self) -> int:
        return len(self._cache)
