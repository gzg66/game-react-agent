"""Persistent navigation memory — tracks button click outcomes per page."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NavigationRecord:
    """Outcome of clicking a button on a specific page."""

    result_page_hash: str
    times_tried: int = 1
    effective: bool = False
    last_tried: float = 0.0


class NavigationMemory:
    """Tracks (page_hash, button) -> navigation outcome across runs.

    Persists to a JSON file so the agent doesn't repeat known-ineffective
    clicks in future runs.
    """

    def __init__(self, ineffective_threshold: int = 2) -> None:
        self._records: dict[str, NavigationRecord] = {}
        self._ineffective_threshold = ineffective_threshold
        self._path: Path | None = None

    @staticmethod
    def _key(page_hash: str, button_name: str) -> str:
        return f"{page_hash}|{button_name}"

    def record(
        self, page_hash: str, button_name: str, result_page_hash: str
    ) -> None:
        key = self._key(page_hash, button_name)
        effective = page_hash != result_page_hash

        if key in self._records:
            rec = self._records[key]
            rec.times_tried += 1
            rec.result_page_hash = result_page_hash
            rec.effective = effective
            rec.last_tried = time.time()
        else:
            self._records[key] = NavigationRecord(
                result_page_hash=result_page_hash,
                times_tried=1,
                effective=effective,
                last_tried=time.time(),
            )
        self._autosave()

    def get_ineffective_buttons(self, page_hash: str) -> list[str]:
        prefix = f"{page_hash}|"
        result = []
        for key, rec in self._records.items():
            if (
                key.startswith(prefix)
                and not rec.effective
                and rec.times_tried >= self._ineffective_threshold
            ):
                result.append(key[len(prefix):])
        return result

    def get_known_transitions(self, page_hash: str) -> dict[str, str]:
        prefix = f"{page_hash}|"
        result = {}
        for key, rec in self._records.items():
            if key.startswith(prefix) and rec.effective:
                button = key[len(prefix):]
                result[button] = rec.result_page_hash
        return result

    def to_prompt_text(self, page_hash: str) -> str:
        ineffective = self.get_ineffective_buttons(page_hash)
        transitions = self.get_known_transitions(page_hash)

        if not ineffective and not transitions:
            return "（当前页面无历史导航记录）"

        lines = []
        for btn, target in transitions.items():
            lines.append(f"- {btn} → 跳转到页面 {target[:8]}（有效）")
        for btn in ineffective:
            key = self._key(page_hash, btn)
            rec = self._records[key]
            lines.append(
                f"- {btn}：已尝试{rec.times_tried}次，页面无变化（**无效，请勿再点**）"
            )

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._path = path
        self._write_to_path(path)
        logger.info("导航记忆已保存到 %s（%d 条记录）", path, len(self._records))

    def _write_to_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for key, rec in self._records.items():
            data[key] = asdict(rec)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _autosave(self) -> None:
        if self._path is None:
            return
        self._write_to_path(self._path)
        logger.debug("导航记忆已同步写入 %s", self._path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._path = path
        if not path.exists():
            logger.info("导航记忆文件不存在，从空状态开始：%s", path)
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key, val in raw.items():
                self._records[key] = NavigationRecord(**val)
            logger.info("已加载导航记忆：%d 条记录", len(self._records))
        except Exception as exc:
            logger.warning("加载导航记忆失败，从空状态开始：%s", exc)

    def __len__(self) -> int:
        return len(self._records)
