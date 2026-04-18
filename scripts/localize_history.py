"""将历史导出的 JSON / SQLite 图数据中的常见英文内容迁移为中文。"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXACT_TEXT_MAP = {
    "completed": "已完成",
    "unknown": "未知",
    "unknown_page": "未知页面",
    "navigation": "导航",
    "combat": "战斗",
    "inventory": "背包",
    "shop": "商店",
    "settings": "设置",
    "dialogue": "对话",
    "dialog": "对话",
    "loading": "加载",
    "login": "登录",
    "lobby": "大厅",
    "guide": "引导",
    "reward": "奖励",
    "battle_prepare": "战前准备",
    "battle_running": "战斗中",
    "battle_result": "战斗结算",
    "hero": "英雄",
    "Heroes": "英雄",
    "Shop": "商店",
    "Battle": "战斗",
    "Back": "返回",
    "Upgrade": "升级",
    "Buy Gems": "购买宝石",
    "Backpack": "背包",
    "Warrior": "战士",
}


def ensure_utf8_console() -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def localize_text(text: str) -> str:
    if text in EXACT_TEXT_MAP:
        return EXACT_TEXT_MAP[text]

    page_match = re.fullmatch(r"page_([0-9a-zA-Z_]+)", text)
    if page_match:
        return f"页面_{page_match.group(1)}"

    auto_discovered = re.fullmatch(r"Auto-discovered at step (\d+)", text)
    if auto_discovered:
        return f"在第 {auto_discovered.group(1)} 步自动发现"

    failed_annotate = re.fullmatch(r"Auto-discovered \(annotation failed\)", text)
    if failed_annotate:
        return "自动发现（标注失败）"

    return text


def localize_obj(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: localize_obj(value) for key, value in data.items()}
    if isinstance(data, list):
        return [localize_obj(item) for item in data]
    if isinstance(data, str):
        return localize_text(data)
    return data


def migrate_json_file(path: Path) -> bool:
    try:
        original = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[跳过] 无法读取 JSON：{path}，原因：{exc}")
        return False

    localized = localize_obj(original)
    if localized == original:
        return False

    path.write_text(
        json.dumps(localized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def has_nodes_table(conn: sqlite3.Connection) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'nodes'"
    )
    return cursor.fetchone() is not None


def migrate_db_file(path: Path) -> int:
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error as exc:
        print(f"[跳过] 无法打开数据库：{path}，原因：{exc}")
        return 0

    updated_rows = 0
    try:
        if not has_nodes_table(conn):
            return 0

        rows = conn.execute(
            "SELECT node_id, page_name, page_description, page_category FROM nodes"
        ).fetchall()
        for node_id, page_name, page_description, page_category in rows:
            new_page_name = localize_text(page_name or "")
            new_description = localize_text(page_description or "")
            new_category = localize_text(page_category or "")

            if (
                new_page_name != (page_name or "")
                or new_description != (page_description or "")
                or new_category != (page_category or "")
            ):
                conn.execute(
                    """
                    UPDATE nodes
                    SET page_name = ?, page_description = ?, page_category = ?
                    WHERE node_id = ?
                    """,
                    (new_page_name, new_description, new_category, node_id),
                )
                updated_rows += 1

        conn.commit()
        return updated_rows
    finally:
        conn.close()


def discover_json_files() -> list[Path]:
    output_root = PROJECT_ROOT / "outputs"
    if not output_root.exists():
        return []
    return sorted(output_root.rglob("*.json"))


def discover_db_files() -> list[Path]:
    candidates: list[Path] = []

    data_db = PROJECT_ROOT / "data" / "graph.db"
    if data_db.exists():
        candidates.append(data_db)

    output_root = PROJECT_ROOT / "outputs"
    if output_root.exists():
        candidates.extend(sorted(output_root.rglob("*.db")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def main() -> None:
    ensure_utf8_console()

    json_files = discover_json_files()
    db_files = discover_db_files()

    if not json_files and not db_files:
        print("未发现可迁移的历史 outputs JSON 或 graph.db，已跳过。")
        return

    updated_json = 0
    for path in json_files:
        if migrate_json_file(path):
            updated_json += 1
            print(f"[JSON] 已迁移：{path.relative_to(PROJECT_ROOT)}")

    updated_db_rows = 0
    touched_db_files = 0
    for path in db_files:
        updated_rows = migrate_db_file(path)
        if updated_rows > 0:
            touched_db_files += 1
            updated_db_rows += updated_rows
            print(
                f"[DB] 已迁移：{path.relative_to(PROJECT_ROOT)}（更新 {updated_rows} 条 nodes 记录）"
            )

    if updated_json == 0 and updated_db_rows == 0:
        print("已扫描历史 JSON / graph.db，但没有发现需要迁移的英文内容。")
        return

    print(
        "迁移完成："
        f"已更新 {updated_json} 个 JSON 文件，"
        f"{touched_db_files} 个数据库文件，"
        f"共 {updated_db_rows} 条节点记录。"
    )


if __name__ == "__main__":
    main()
