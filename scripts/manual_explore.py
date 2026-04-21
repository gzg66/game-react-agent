"""Interactive manual cold-start explorer.

The user clicks buttons on the device while this tool records page hashes
and directed edges into the SQLite graph cache.

Usage:
    python scripts/manual_explore.py --config config/xttc_poco.yaml
    python scripts/manual_explore.py --config config/xttc_poco.yaml --action-wait 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCREENSHOT_DIR = SCRIPT_DIR / "page_screenshots"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_agent.cognition.gemini_client import GeminiClient  # noqa: E402
from game_agent.cognition.navigation_memory import NavigationMemory  # noqa: E402
from game_agent.cognition.page_cache import PageKnowledge, PageKnowledgeCache  # noqa: E402
from game_agent.config import load_config, load_env_file  # noqa: E402
from game_agent.device.airtest_device import AirtestDevice  # noqa: E402
from game_agent.device.base import PocoNode  # noqa: E402
from game_agent.graph.annotator import PageAnnotator  # noqa: E402
from game_agent.graph.models import EdgeAction, GraphEdge, GraphNode, UIStateGraph  # noqa: E402
from game_agent.graph.storage import GraphStorage  # noqa: E402
from game_agent.logging_setup import setup_logging  # noqa: E402
from game_agent.perception.poco_tree import PocoTreeExtractor  # noqa: E402
from game_agent.perception.state import L1Perception  # noqa: E402

logger = logging.getLogger(__name__)

_POPUP_MIN_NODES = 3


def _ensure_utf8_console() -> None:
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="手动冷启动探索：人工点击 + 程序记录页面转场图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="./config/xttc_poco.yaml", help="配置 YAML 路径")
    parser.add_argument("--device-uri", default=None, help="显式指定 Airtest 设备 URI")
    parser.add_argument("--serial", default=None, help="显式指定设备串号")
    parser.add_argument(
        "--screenshot-dir",
        default=str(DEFAULT_SCREENSHOT_DIR),
        help="截图保存目录",
    )
    parser.add_argument(
        "--action-wait",
        type=float,
        default=2.0,
        help="点击后等待页面变化的秒数",
    )
    return parser


def _normalize_platform(platform: str) -> str:
    normalized = platform.strip().lower()
    if normalized == "android":
        return "Android"
    if normalized == "ios":
        return "iOS"
    return platform.strip() or "Android"


def _resolve_device_config(config, args) -> None:
    if args.device_uri:
        config.game.device_uri = args.device_uri.strip()

    serial = (
        (args.serial or "").strip()
        or (config.game.device_serial or "").strip()
        or ((config.device.serial or "").strip() if config.device.serial else "")
    )
    if serial:
        config.game.device_serial = serial

    if not (config.game.device_uri or "").strip():
        if not serial:
            raise SystemExit(
                "未配置设备连接信息：请通过 --config 提供 game.device_uri，"
                "或传入 --device-uri Android:///127.0.0.1:16384"
            )
        platform = _normalize_platform(config.device.platform)
        config.game.device_uri = f"{platform}:///{serial}"


# ---------------------------------------------------------------------------
# Popup detection — tree-path based
# ---------------------------------------------------------------------------
#
# In CocosJS/Unity Poco trees the popup is rendered LAST, so its nodes
# appear at the END of the DFS-ordered list.  We detect a popup by
# finding a branch-point in the poco_path where the interactive nodes
# split into two groups.  The last group (by DFS order) is the popup.


def _detect_popup_nodes(
    interactive: list[PocoNode],
) -> tuple[list[PocoNode], list[PocoNode]]:
    """Split interactive nodes into (popup, main) by Poco-tree branching.

    Returns ``(popup_nodes, main_nodes)``.
    If no popup is detected, returns ``(all_nodes, [])``.
    """
    if len(interactive) < _POPUP_MIN_NODES:
        return interactive, []

    segments = [n.poco_path.split(" > ") for n in interactive]
    min_depth = min(len(s) for s in segments)

    # Find deepest common prefix
    common = 0
    for d in range(min_depth):
        if len({s[d] for s in segments}) == 1:
            common = d + 1
        else:
            break

    # Walk outward from the common prefix looking for a branch split
    for depth in range(common, min(common + 3, min_depth)):
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(segments):
            branch = s[depth] if depth < len(s) else "__leaf__"
            groups.setdefault(branch, []).append(i)

        if len(groups) < 2:
            continue

        # Reject leaf-level splits where every node is its own group —
        # a real popup branch should hold ≥2 nodes.
        largest_group = max(len(v) for v in groups.values())
        if largest_group < 2:
            continue

        # Last group by max DFS index → topmost (popup) layer
        sorted_groups = sorted(groups.values(), key=lambda idxs: max(idxs))
        popup_idxs = sorted_groups[-1]
        main_idxs = [i for g in sorted_groups[:-1] for i in g]

        # Popup must be the minority and contain ≥2 nodes
        if len(popup_idxs) < 2 or len(popup_idxs) >= len(main_idxs):
            continue

        popup = [interactive[i] for i in sorted(popup_idxs)]
        main = [interactive[i] for i in sorted(main_idxs)]
        return popup, main

    return interactive, []


def _reorder_by_popup(
    perception: L1Perception,
) -> tuple[list[PocoNode], int]:
    """Returns ``(display_nodes, top_count)``.

    When a popup is detected the first *top_count* entries are the popup
    buttons; the rest are background.  Otherwise *top_count* equals the
    total count (no popup).
    """
    popup, main = _detect_popup_nodes(perception.interactive_nodes)
    if main:
        return popup + main, len(popup)
    return popup, len(popup)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _node_label(node: PocoNode) -> str:
    text_part = f' "{node.text}"' if node.text else ""
    return f"{node.type} {node.name}{text_part}"


def _btn_desc_map(page_cache: PageKnowledgeCache, page_hash: str) -> dict[str, str]:
    """Build node_name → Chinese description map from LLM annotation."""
    cached = page_cache.get(page_hash)
    if not cached or not cached.key_buttons:
        return {}
    result: dict[str, str] = {}
    for kb in cached.key_buttons:
        if isinstance(kb, dict):
            name = kb.get("name", "")
            desc = kb.get("desc", "")
            if name and desc:
                result[name] = desc
    return result


def _format_node_line(
    idx: int,
    node: PocoNode,
    desc_map: dict[str, str],
    prefix: str = "  ",
) -> str:
    desc = desc_map.get(node.name, "")
    text_part = f' "{desc}"' if desc else (f' "{node.text}"' if node.text else "")
    return (
        f"  [{idx:2d}]{prefix}{node.type} {node.name}{text_part}"
        f"  pos=({node.pos[0]:.2f}, {node.pos[1]:.2f})"
    )


def _show_page(
    page_hash: str,
    display_nodes: list[PocoNode],
    top_count: int,
    graph: UIStateGraph,
    page_cache: PageKnowledgeCache,
    show_all: bool = False,
) -> None:
    gnode = graph.get_node(page_hash)
    cached = page_cache.get(page_hash)
    page_name = (
        (gnode.page_name if gnode else None)
        or (cached.page_name if cached else None)
        or "未命名"
    )

    has_popup = top_count < len(display_nodes)
    desc_map = _btn_desc_map(page_cache, page_hash)

    print(f"\n{'=' * 55}")
    tag = "  [弹窗]" if has_popup else ""
    print(f"  页面: {page_hash}  ({page_name}){tag}")
    if cached and cached.page_description:
        print(f"  {cached.page_category} | {cached.page_description}")
    print(f"{'=' * 55}")

    if not display_nodes:
        print("  (无可交互节点)")
    elif has_popup and not show_all:
        print(f"最上层按钮 ({top_count} 个):")
        for i, n in enumerate(display_nodes[:top_count], 1):
            print(_format_node_line(i, n, desc_map))
        bg_count = len(display_nodes) - top_count
        print(f"  --- 底层 {bg_count} 个按钮被遮挡 (输入 all 查看) ---")
    else:
        if has_popup:
            print(f"最上层按钮 ({top_count} 个):")
        else:
            print(f"可交互节点 ({len(display_nodes)} 个):")
        for i, n in enumerate(display_nodes, 1):
            prefix = "  " if not has_popup or i <= top_count else "* "
            print(_format_node_line(i, n, desc_map, prefix))
            if has_popup and i == top_count:
                print(f"  --- 底层 ({len(display_nodes) - top_count} 个) ---")

    print(f"\n图状态: {graph.node_count()} 节点, {graph.edge_count()} 边")
    cmds = "<编号> | h | name | back | tap <x> <y> | close"
    if has_popup:
        cmds += " | all"
    cmds += " | info | undo | export | q"
    print(f"命令: {cmds}")


def _show_info(graph: UIStateGraph) -> None:
    print(f"\n--- 图统计 ---")
    print(f"节点数: {graph.node_count()}")
    print(f"边数:   {graph.edge_count()}")
    if graph.all_nodes():
        print("已知页面:")
        for node in graph.all_nodes():
            edges_out = len(graph.get_edges_from(node.node_id))
            print(f"  {node.node_id[:8]}  {node.page_name}  (出边={edges_out}, 访问={node.visit_count})")


# ---------------------------------------------------------------------------
# Core snapshot
# ---------------------------------------------------------------------------

def _snapshot(extractor: PocoTreeExtractor) -> L1Perception:
    return extractor.extract()


def _do_snapshot(
    extractor: PocoTreeExtractor,
    graph: UIStateGraph,
    storage: GraphStorage,
    device: AirtestDevice,
    screenshot_dir: Path,
    annotator: PageAnnotator | None = None,
    page_cache: PageKnowledgeCache | None = None,
) -> tuple[L1Perception, str, list[PocoNode], int]:
    """Snapshot + ensure node + annotate + reorder. Returns (perception, hash, display_nodes, top_count)."""
    perception = _snapshot(extractor)
    page_hash = perception.page_hash
    is_new = _ensure_current_node(page_hash, perception, graph, storage)
    _save_screenshot(device, page_hash, screenshot_dir)
    if is_new and page_cache is not None:
        _annotate_page(page_hash, perception, annotator, graph, storage, page_cache)
    display_nodes, top_count = _reorder_by_popup(perception)
    return perception, page_hash, display_nodes, top_count


def _ensure_current_node(
    page_hash: str,
    perception: L1Perception,
    graph: UIStateGraph,
    storage: GraphStorage,
) -> bool:
    """Returns True if this is a newly created node."""
    if not graph.has_node(page_hash):
        node = GraphNode(
            node_id=page_hash,
            page_name=f"页面_{page_hash[:8]}",
            poco_tree_snapshot=perception.poco_tree_markdown or "",
            discovered_at=time.time(),
            visit_count=1,
        )
        graph.add_node(node)
        storage.save_node(node)
        print(f"  [新页面] 已自动创建节点 {page_hash[:8]}")
        return True
    else:
        existing = graph.get_node(page_hash)
        if existing:
            existing.visit_count += 1
        return False


def _annotate_page(
    page_hash: str,
    perception: L1Perception,
    annotator: PageAnnotator | None,
    graph: UIStateGraph,
    storage: GraphStorage,
    page_cache: PageKnowledgeCache,
) -> None:
    if page_cache.has(page_hash):
        return
    if annotator is None:
        return
    try:
        print("  [LLM] 正在标注页面 ...")
        annotation = annotator.annotate(perception)
        node = graph.get_node(page_hash)
        if node:
            node.page_name = annotation.page_name
            node.page_description = annotation.page_description
            node.page_category = annotation.page_category
            node.key_buttons = annotation.key_buttons
            storage.save_node(node)
        page_cache.put(
            page_hash,
            PageKnowledge(
                page_name=annotation.page_name,
                page_description=annotation.page_description,
                page_category=annotation.page_category,
                key_buttons=annotation.key_buttons,
            ),
        )
        print(f"  [LLM] {annotation.page_name} ({annotation.page_category})")
        if annotation.key_buttons:
            for kb in annotation.key_buttons:
                if isinstance(kb, dict):
                    print(f"         - {kb.get('name', '?')}: {kb.get('desc', '')}")
    except Exception as exc:
        logger.warning("LLM 页面标注失败: %s", exc)
        print(f"  [LLM] 标注失败: {exc}")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _do_click_node(device: AirtestDevice, node: PocoNode) -> None:
    try:
        result = device.click_poco(node.poco_path)
        if not result.success:
            device.click(node.pos)
    except Exception:
        device.click(node.pos)


def _record_edge_and_show(
    source_hash: str,
    new_hash: str,
    perception: L1Perception,
    action: EdgeAction,
    action_label: str,
    graph: UIStateGraph,
    storage: GraphStorage,
    device: AirtestDevice,
    screenshot_dir: Path,
    annotator: PageAnnotator | None = None,
    page_cache: PageKnowledgeCache | None = None,
) -> None:
    edge = GraphEdge(
        source_id=source_hash,
        target_id=new_hash,
        actions=[action],
        discovered_at=time.time(),
    )
    graph.add_edge(edge)
    storage.save_edge(edge)

    is_new = _ensure_current_node(new_hash, perception, graph, storage)
    _save_screenshot(device, new_hash, screenshot_dir)

    if is_new and page_cache is not None:
        _annotate_page(new_hash, perception, annotator, graph, storage, page_cache)

    node = graph.get_node(new_hash)
    name = node.page_name if node else "?"
    if is_new:
        print(f"  >>> 发现新页面 {new_hash[:8]} ({name})")
    else:
        print(f"  >>> 跳转到已知页面 {new_hash[:8]} ({name})")

    print(f"  边已记录: {source_hash[:8]} --[{action_label}]--> {new_hash[:8]}")


def _handle_click(
    idx: int,
    display_nodes: list[PocoNode],
    current_hash: str,
    extractor: PocoTreeExtractor,
    graph: UIStateGraph,
    storage: GraphStorage,
    nav_memory: NavigationMemory,
    screenshot_dir: Path,
    device: AirtestDevice,
    action_wait: float,
    annotator: PageAnnotator | None = None,
    page_cache: PageKnowledgeCache | None = None,
) -> tuple[str, list[PocoNode], int, L1Perception]:
    if idx < 1 or idx > len(display_nodes):
        print(f"  无效编号，请输入 1-{len(display_nodes)}")
        perception = _snapshot(extractor)
        display_nodes, top_count = _reorder_by_popup(perception)
        return current_hash, display_nodes, top_count, perception

    clicked = display_nodes[idx - 1]
    print(f"  点击: [{idx}] {_node_label(clicked)}")
    _do_click_node(device, clicked)

    print(f"  等待 {action_wait}s ...")
    time.sleep(action_wait)

    perception = _snapshot(extractor)
    new_hash = perception.page_hash
    display_nodes_new, top_count = _reorder_by_popup(perception)

    nav_memory.record(current_hash, clicked.name, new_hash)

    if new_hash != current_hash:
        action = EdgeAction(
            action_type="poco_click",
            target=clicked.poco_path,
            params={
                "node_name": clicked.name,
                "node_text": clicked.text or "",
                "pos": list(clicked.pos),
            },
        )
        _record_edge_and_show(
            current_hash, new_hash, perception, action, clicked.name,
            graph, storage, device, screenshot_dir,
            annotator, page_cache,
        )
        return new_hash, display_nodes_new, top_count, perception
    else:
        print("  页面未变化")
        return current_hash, display_nodes_new, top_count, perception


def _handle_tap(
    x: float,
    y: float,
    current_hash: str,
    extractor: PocoTreeExtractor,
    graph: UIStateGraph,
    storage: GraphStorage,
    nav_memory: NavigationMemory,
    screenshot_dir: Path,
    device: AirtestDevice,
    action_wait: float,
    annotator: PageAnnotator | None = None,
    page_cache: PageKnowledgeCache | None = None,
) -> tuple[str, list[PocoNode], int, L1Perception]:
    print(f"  点击坐标: ({x:.2f}, {y:.2f})")
    device.click((x, y))

    print(f"  等待 {action_wait}s ...")
    time.sleep(action_wait)

    perception = _snapshot(extractor)
    new_hash = perception.page_hash
    display_nodes, top_count = _reorder_by_popup(perception)

    label = f"tap({x:.2f},{y:.2f})"
    nav_memory.record(current_hash, label, new_hash)

    if new_hash != current_hash:
        action = EdgeAction(
            action_type="touch_pos",
            target=f"{x:.4f},{y:.4f}",
            params={"pos": [x, y]},
        )
        _record_edge_and_show(
            current_hash, new_hash, perception, action, label,
            graph, storage, device, screenshot_dir,
            annotator, page_cache,
        )
        return new_hash, display_nodes, top_count, perception
    else:
        print("  页面未变化")
        return current_hash, display_nodes, top_count, perception


def _handle_back(
    current_hash: str,
    extractor: PocoTreeExtractor,
    graph: UIStateGraph,
    storage: GraphStorage,
    nav_memory: NavigationMemory,
    screenshot_dir: Path,
    device: AirtestDevice,
    action_wait: float,
    annotator: PageAnnotator | None = None,
    page_cache: PageKnowledgeCache | None = None,
) -> tuple[str, list[PocoNode], int, L1Perception]:
    print("  执行 press_back ...")
    device.press_back()
    print(f"  等待 {action_wait}s ...")
    time.sleep(action_wait)

    perception = _snapshot(extractor)
    new_hash = perception.page_hash
    display_nodes, top_count = _reorder_by_popup(perception)

    nav_memory.record(current_hash, "BACK", new_hash)

    if new_hash != current_hash:
        action = EdgeAction("press_back", "BACK", {})
        _record_edge_and_show(
            current_hash, new_hash, perception, action, "BACK",
            graph, storage, device, screenshot_dir,
            annotator, page_cache,
        )
        return new_hash, display_nodes, top_count, perception
    else:
        print("  页面未变化（可能已在最顶层）")
        return current_hash, display_nodes, top_count, perception


def _handle_name(
    name: str,
    current_hash: str,
    graph: UIStateGraph,
    storage: GraphStorage,
    page_cache: PageKnowledgeCache,
) -> None:
    node = graph.get_node(current_hash)
    if node:
        node.page_name = name
        storage.save_node(node)

    page_cache.put(
        current_hash,
        PageKnowledge(
            page_name=name,
            page_description="",
            page_category="手动标注",
            key_buttons=[],
        ),
    )
    print(f"  已标注: {current_hash[:8]} = \"{name}\"")


def _handle_undo(graph: UIStateGraph, storage: GraphStorage) -> None:
    removed = storage.delete_last_edge()
    if removed:
        print(
            f"  已撤销: {removed.source_id[:8]} --> {removed.target_id[:8]}"
            f"  (actions={[a.target for a in removed.actions]})"
        )
    else:
        print("  无可撤销的边")


def _handle_export(storage: GraphStorage, graph: UIStateGraph) -> None:
    output_dir = Path("outputs/manual_explore")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "state_graph.json"
    storage.export_json(str(out_path))
    print(f"  已导出: {out_path} ({graph.node_count()} 节点, {graph.edge_count()} 边)")


def _save_screenshot(device: AirtestDevice, page_hash: str, screenshot_dir: Path) -> None:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    path = screenshot_dir / f"{page_hash}.png"
    try:
        data = device.screenshot()
        if data:
            path.write_bytes(data)
    except Exception as exc:
        logger.debug("截图保存失败: %s", exc)


def _parse_tap_args(raw: str) -> tuple[float, float] | None:
    """Parse 'tap 0.5 0.3' or 'tap 0.5,0.3'. Returns normalized coords or None."""
    parts = raw[3:].strip().replace(",", " ").split()
    if len(parts) != 2:
        return None
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if x > 1.0 or y > 1.0:
        print(f"  坐标看起来是像素值，请使用 0-1 的归一化坐标")
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        print(f"  坐标超出范围，请使用 0-1 的归一化坐标")
        return None
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    _ensure_utf8_console()
    load_env_file(PROJECT_ROOT / ".env")
    args = _build_arg_parser().parse_args()
    config = load_config(args.config)
    config.device.use_mock = False
    setup_logging(level_override=config.log_level)

    action_wait = args.action_wait
    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Connect device
    _resolve_device_config(config, args)
    print("[manual] 正在连接设备 ...")
    device = AirtestDevice(config.game)
    device.connect()
    print(f"[manual] 已连接: {config.game.device_uri}")

    # Load existing graph
    storage = GraphStorage(config.graph.db_path)
    graph = storage.load_graph()
    print(f"[manual] 已加载图: {graph.node_count()} 节点, {graph.edge_count()} 边")

    # Load caches
    nav_memory = NavigationMemory()
    nav_memory.load(config.navigation_memory.file_path)

    page_cache = PageKnowledgeCache()
    page_cache.load(config.page_cache.file_path)

    # Initialize LLM annotator (optional — works without API key)
    # Disable forced JSON response_mime_type — the prompt already asks for JSON,
    # and JSON mode can cause empty responses with some Gemini models.
    annotator: PageAnnotator | None = None
    try:
        config.gemini.response_mime_type = ""
        gemini = GeminiClient(config.gemini)
        if gemini._client is not None:
            annotator = PageAnnotator(gemini)
            print(f"[manual] LLM 标注已启用 (model={config.gemini.model_name})")
        else:
            print("[manual] LLM 标注不可用（未配置 API key）")
    except Exception as exc:
        print(f"[manual] LLM 标注不可用: {exc}")

    # Initial snapshot
    extractor = PocoTreeExtractor(device, config.perception)
    perception, current_hash, display_nodes, top_count = _do_snapshot(
        extractor, graph, storage, device, screenshot_dir,
        annotator, page_cache,
    )

    _show_page(current_hash, display_nodes, top_count, graph, page_cache)

    # Interactive loop
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        if not raw:
            continue

        # Quit
        if raw.lower() in ("q", "quit", "exit"):
            nav_memory.save(config.navigation_memory.file_path)
            page_cache.save(config.page_cache.file_path)
            storage.close()
            print("[manual] 已保存，再见！")
            return 0

        # Refresh hash
        if raw.lower() in ("h", "hash", "refresh"):
            perception, current_hash, display_nodes, top_count = _do_snapshot(
                extractor, graph, storage, device, screenshot_dir,
                annotator, page_cache,
            )
            _show_page(current_hash, display_nodes, top_count, graph, page_cache)
            continue

        # Show all nodes (including background behind popup)
        if raw.lower() == "all":
            _show_page(current_hash, display_nodes, top_count, graph, page_cache, show_all=True)
            continue

        # Debug: show poco_path tree for each interactive node
        if raw.lower() == "tree":
            print(f"\n--- poco_path (共 {len(display_nodes)} 个交互节点) ---")
            for i, n in enumerate(display_nodes, 1):
                tag = " [弹窗]" if i <= top_count < len(display_nodes) else ""
                print(f"  [{i:2d}]{tag} {n.name}  path={n.poco_path}")
            continue

        # Name current page
        if raw.lower().startswith("name "):
            name = raw[5:].strip()
            if name:
                _handle_name(name, current_hash, graph, storage, page_cache)
            else:
                print("  用法: name <页面名称>")
            continue

        # Press back
        if raw.lower() in ("back", "b"):
            current_hash, display_nodes, top_count, perception = _handle_back(
                current_hash, extractor, graph, storage,
                nav_memory, screenshot_dir, device, action_wait,
                annotator, page_cache,
            )
            _show_page(current_hash, display_nodes, top_count, graph, page_cache)
            continue

        # Tap arbitrary coordinates
        if raw.lower().startswith("tap "):
            coords = _parse_tap_args(raw)
            if coords:
                current_hash, display_nodes, top_count, perception = _handle_tap(
                    coords[0], coords[1], current_hash,
                    extractor, graph, storage,
                    nav_memory, screenshot_dir, device, action_wait,
                    annotator, page_cache,
                )
                _show_page(current_hash, display_nodes, top_count, graph, page_cache)
            else:
                print("  用法: tap <x> <y>  (归一化坐标 0-1，例如 tap 0.5 0.3)")
            continue

        # Close popup (tap bottom-center where "点击任意区域关闭" typically is)
        if raw.lower() == "close":
            print("  点击屏幕底部关闭弹窗 ...")
            current_hash, display_nodes, top_count, perception = _handle_tap(
                0.50, 0.95, current_hash,
                extractor, graph, storage,
                nav_memory, screenshot_dir, device, action_wait,
                annotator, page_cache,
            )
            _show_page(current_hash, display_nodes, top_count, graph, page_cache)
            continue

        # Graph info
        if raw.lower() in ("info", "i"):
            _show_info(graph)
            continue

        # Undo last edge
        if raw.lower() == "undo":
            _handle_undo(graph, storage)
            graph = storage.load_graph()
            continue

        # Export
        if raw.lower() == "export":
            _handle_export(storage, graph)
            continue

        # Number → click
        if raw.isdigit():
            idx = int(raw)
            current_hash, display_nodes, top_count, perception = _handle_click(
                idx, display_nodes, current_hash,
                extractor, graph, storage,
                nav_memory, screenshot_dir, device, action_wait,
                annotator, page_cache,
            )
            _show_page(current_hash, display_nodes, top_count, graph, page_cache)
            continue

        print(f"  未知命令: {raw}")
        print("  可用: <编号> | h | name | back | tap <x> <y> | close | all | tree | info | undo | export | q")


if __name__ == "__main__":
    raise SystemExit(main())
