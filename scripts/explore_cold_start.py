"""CLI entry point for cold-start UI exploration.

Supports both mock mode (for testing) and real device mode.

Usage:
    # Mock mode (default)
    python scripts/explore_cold_start.py --mock

    # Real device with game config
    python scripts/explore_cold_start.py --config config/xttc_poco.yaml

    # Override max steps
    python scripts/explore_cold_start.py --config config/xttc_poco.yaml --max-steps 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game_agent.cognition.gemini_client import GeminiClient
from game_agent.cognition.vision_client import VisionLLMClient
from game_agent.config import (
    GameConnectionConfig,
    load_config,
    load_env_file,
)
from game_agent.device.base import PocoNode
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.graph.annotator import PageAnnotator
from game_agent.graph.explorer import GraphExplorer
from game_agent.graph.hasher import PageHasher
from game_agent.graph.models import UIStateGraph
from game_agent.graph.storage import GraphStorage
from game_agent.logging_setup import setup_logging
from game_agent.perception.poco_tree import PocoTreeExtractor

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# ADB utilities (for device detection before Airtest connects)
# ---------------------------------------------------------------------------

def _find_adb() -> str:
    try:
        import airtest
        adb_exe = (
            Path(airtest.__file__).resolve().parent
            / "core" / "android" / "static" / "adb" / "windows" / "adb.exe"
        )
        if adb_exe.exists():
            return str(adb_exe)
    except Exception:
        pass
    return "adb"


def _adb_global(*args: str) -> str:
    result = subprocess.run(
        [_find_adb(), *args],
        check=False, capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
    )
    return (result.stdout or result.stderr or "").strip()


def _adb_device(serial: str, *args: str) -> str:
    result = subprocess.run(
        [_find_adb(), "-s", serial, *args],
        check=False, capture_output=True, text=True,
        encoding="utf-8", errors="ignore",
    )
    return (result.stdout or result.stderr or "").strip()


def _is_device_ready(serial: str) -> bool:
    devices_output = _adb_global("devices")
    return f"{serial}\tdevice" in devices_output


def _is_device_offline(serial: str) -> bool:
    devices_output = _adb_global("devices")
    return f"{serial}\toffline" in devices_output


def _try_auto_connect(serial: str) -> bool:
    if ":" not in serial:
        return False
    output = _adb_global("connect", serial)
    normalized = output.lower()
    return "connected to" in normalized or "already connected" in normalized


def _detect_runtime(config: GameConnectionConfig) -> None:
    """Detect and validate device connection, engine type, and Poco port."""
    serial = config.device_serial
    if not _is_device_ready(serial):
        _try_auto_connect(serial)
        import time
        time.sleep(2)
        if not _is_device_ready(serial):
            if _is_device_offline(serial):
                print(f"[detect] 警告：设备 {serial} 当前离线，模拟器可能需要重启。")
                print("[detect] 将继续尝试，Airtest 可能会自行恢复连接...")
            else:
                devices = _adb_global("devices")
                raise RuntimeError(
                    f"设备未连接：{serial}。\n"
                    f"  adb devices 输出：\n{devices}\n"
                    f"  请确认模拟器已启动，并且 adb 已成功连接。"
                )

    device_online = _is_device_ready(serial)
    if device_online:
        package_path = _adb_device(serial, "shell", "pm", "path", config.package_name)
        if not package_path.startswith("package:"):
            raise RuntimeError(f"设备中未安装目标包：{config.package_name}")
    else:
        print("[detect] 设备离线，跳过安装包检查")

    pid = "未知"
    if device_online:
        listening = _adb_device(serial, "shell", "ss", "-ltn")
        if "5003" in listening and config.engine_type != "cocos2dx_js":
            print("[detect] 检测到 5003 端口，已将引擎自动修正为 cocos2dx_js")
            config.engine_type = "cocos2dx_js"
            config.poco_port = 5003

        pid = _adb_device(serial, "shell", "pidof", config.package_name) or "未运行"
        focus = _adb_device(serial, "shell", "dumpsys", "window")
        focus_lines = [
            line.strip() for line in focus.splitlines()
            if "mCurrentFocus" in line or "mFocusedApp" in line
        ]
    else:
        focus_lines = []

    print("[detect] 当前设备与游戏配置：")
    print(f"  设备串号：{serial}（{'在线' if device_online else '离线'}）")
    print(f"  包名：    {config.package_name}")
    print(f"  引擎：    {config.engine_type}")
    print(f"  Poco：    {config.poco_host}:{config.effective_poco_port()}")
    print(f"  进程 PID：{pid}")
    if focus_lines:
        print("  焦点窗口：")
        for line in focus_lines:
            print(f"    {line}")


# ---------------------------------------------------------------------------
# Page annotation prompt
# ---------------------------------------------------------------------------

PAGE_ANNOTATION_PROMPT = """请结合这张游戏截图和下方 UI 树信息，返回一个 JSON 对象：

{
    "page_name": "简短中文页面名",
    "page_description": "用一句中文简要描述当前页面",
    "page_category": "以下之一：登录、大厅、对话、引导、奖励、战前准备、战斗中、战斗结算、商店、英雄、设置、未知",
    "key_buttons": ["重要", "按钮", "名称"],
    "recommended_actions": ["建议按顺序点击的按钮文案或节点名"],
    "is_popup": false,
    "is_high_risk": false
}

要求：
- 所有字段值都使用中文。
- 仅返回 JSON，不要添加额外说明。
- key_buttons 中填写页面上的重要按钮文案（即用户看到的可见文字）。
- recommended_actions 中按推荐的点击顺序排列，排除高风险按钮（充值/支付/删除）。
- 重点描述当前可见 UI 元素。"""


# ---------------------------------------------------------------------------
# Node-label alignment: match LLM labels to Poco nodes
# ---------------------------------------------------------------------------

def _node_display_label(node: PocoNode) -> str:
    """Human-readable label for a Poco node: 'name (text)' or just 'name'."""
    text = (node.text or "").strip()
    name = (node.name or "").strip()
    if text and text.lower() != name.lower():
        return f"{name}（{text}）"
    return text or name


def _match_label_to_node(label: str, nodes: list[PocoNode]) -> PocoNode | None:
    """Match an LLM-returned label (e.g. '登录') to a Poco node.

    Match strategy (first wins):
    1. Exact match on node.text
    2. Exact match on node.name
    3. label is a substring of node.text or vice versa
    4. label is a substring of node.name or vice versa (case-insensitive)
    """
    label_lower = label.strip().lower()
    if not label_lower:
        return None

    for node in nodes:
        text = (node.text or "").strip()
        if text == label:
            return node

    for node in nodes:
        name = (node.name or "").strip()
        if name == label:
            return node

    for node in nodes:
        text = (node.text or "").strip().lower()
        if text and (label_lower in text or text in label_lower):
            return node

    for node in nodes:
        name = (node.name or "").strip().lower()
        if name and (label_lower in name or name in label_lower):
            return node

    return None


def _prioritize_by_llm(
    interactive_nodes: list[PocoNode],
    annotation: dict | None,
) -> list[tuple[PocoNode, str]]:
    """Reorder interactive_nodes: LLM-recommended first, then remaining.

    Returns list of (node, reason) tuples where reason indicates why.
    """
    if not annotation:
        return [(n, "规则") for n in interactive_nodes]

    recommended = annotation.get("recommended_actions") or annotation.get("key_buttons") or []
    if not recommended:
        return [(n, "规则") for n in interactive_nodes]

    ordered: list[tuple[PocoNode, str]] = []
    used: set[str] = set()

    for label in recommended:
        matched = _match_label_to_node(label, interactive_nodes)
        if matched and matched.poco_path not in used:
            ordered.append((matched, f"LLM推荐「{label}」"))
            used.add(matched.poco_path)

    for node in interactive_nodes:
        if node.poco_path not in used:
            ordered.append((node, "规则补充"))

    return ordered


# ---------------------------------------------------------------------------
# Cold-start exploration loop (real device)
# ---------------------------------------------------------------------------

async def run_real_exploration(config, vision_client: VisionLLMClient | None) -> None:
    """Run cold-start exploration on a real device."""
    from game_agent.device.airtest_device import AirtestDevice

    game_config = config.game
    exploration = config.exploration
    safety = config.safety

    output_dir = Path(exploration.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_dir = output_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Connect to device
    print("\n[explore] 正在连接设备...")
    device = AirtestDevice(game_config)

    # Restart game
    print("[explore] 正在重启游戏...")
    device.force_stop(game_config.package_name)
    time.sleep(1)
    device.start_app(game_config.activity_name)
    print(f"[explore] 等待 {exploration.boot_wait_s} 秒让游戏完成启动...")
    time.sleep(exploration.boot_wait_s)

    # Connect Poco
    device.connect()
    screen_size = device.get_screen_size()
    print(f"[explore] 屏幕分辨率：{screen_size}")

    # Step 2: Exploration state
    graph = UIStateGraph()
    storage = GraphStorage(config.graph.db_path)

    step_counter = 0
    pages_found = 0
    executions: list[dict] = []
    page_annotations: dict[str, dict] = {}

    dangerous_kw = set(safety.dangerous_keywords)

    print("\n[explore] ========== 开始探索 ==========\n")

    while step_counter < exploration.max_steps and pages_found < exploration.max_pages:
        # Capture current page
        hierarchy = device.dump_hierarchy(retries=5, wait_s=2.0)
        if hierarchy is None:
            print("[explore] 获取 UI 层级失败，尝试重新连接...")
            try:
                device.reconnect()
                hierarchy = device.dump_hierarchy()
            except Exception as exc:
                print(f"[explore] 重新连接失败：{exc}")
                break
            if hierarchy is None:
                print("[explore] 重连后仍无法获取层级信息，停止探索。")
                break

        # Compute hash from the current perception snapshot
        tree_extractor = PocoTreeExtractor(device, config.perception)
        perception = tree_extractor.extract()
        page_hash = perception.page_hash

        # Take screenshot
        screenshot_path = str(screenshot_dir / f"step_{step_counter:03d}.png")
        device.snapshot_to_file(screenshot_path)

        # Check if new page
        is_new = not graph.has_node(page_hash)
        if is_new:
            pages_found += 1
            print(f"[explore] [{step_counter}] 发现新页面（hash={page_hash[:8]}，累计={pages_found}）")

            # Annotate with Gemini vision
            annotation = None
            if vision_client and vision_client.available:
                tree_text = perception.poco_tree_markdown[:2000]
                prompt = PAGE_ANNOTATION_PROMPT + f"\n\nUI 树：\n{tree_text}"
                screenshot_name = Path(screenshot_path).name
                print(f"  -> 正在调用视觉 LLM 分析 {screenshot_name}")
                annotation, raw_text = vision_client.analyze_page_with_raw(
                    screenshot_path, prompt
                )
                print(f"  -> 视觉 LLM 返回原始 JSON: {raw_text}")
                if annotation:
                    print(f"  -> 页面名={annotation.get('page_name', '?')}，"
                          f"分类={annotation.get('page_category', '?')}")
            else:
                print(
                    "  -> 本次未调用视觉 LLM，因为 --no-vision / API key 缺失 / client unavailable"
                )

            if not annotation:
                annotation = {
                    "page_name": f"页面_{page_hash[:8]}",
                    "page_category": "未知",
                    "page_description": f"在第 {step_counter} 步自动发现",
                    "key_buttons": [n.name for n in perception.interactive_nodes[:5]],
                }

            page_annotations[page_hash] = annotation

            from game_agent.graph.models import GraphNode
            node = GraphNode(
                node_id=page_hash,
                page_name=annotation.get("page_name", f"页面_{page_hash[:8]}"),
                page_description=annotation.get("page_description", ""),
                page_category=annotation.get("page_category", "未知"),
                key_buttons=annotation.get("key_buttons", []),
                poco_tree_snapshot=perception.poco_tree_markdown,
                discovered_at=time.time(),
            )
            graph.add_node(node)
            storage.save_node(node)
        else:
            existing = graph.get_node(page_hash)
            if existing:
                existing.visit_count += 1
            print(f"[explore] [{step_counter}] 重访页面（hash={page_hash[:8]}）")
            print("  -> 本次未调用视觉 LLM，因为页面是重访页")

        # Find clickable buttons, filter dangerous ones
        current_annotation = page_annotations.get(page_hash)
        interactive = [
            n for n in perception.interactive_nodes
            if n.visible and not any(
                kw in (n.name + " " + (n.text or "")).lower() for kw in dangerous_kw
            )
        ]

        if not interactive:
            print(f"[explore] [{step_counter}] 没有安全的可交互节点，执行返回...")
            device.press_back()
            time.sleep(exploration.action_wait_s)
            step_counter += 1
            continue

        # Prioritize buttons by LLM recommendation
        prioritized = _prioritize_by_llm(interactive, current_annotation)
        llm_count = sum(1 for _, reason in prioritized if reason.startswith("LLM"))
        if llm_count > 0:
            print(f"[explore] 动作排序：{llm_count} 个 LLM 推荐 + "
                  f"{len(prioritized) - llm_count} 个规则补充")

        # Try each button in priority order
        acted = False
        for button, reason in prioritized[:exploration.max_actions_per_page]:
            if step_counter >= exploration.max_steps:
                break

            step_counter += 1
            label = _node_display_label(button)
            print(f"[explore] [{step_counter}] 点击：{label}  [{reason}]")

            try:
                result = device.click_poco(button.poco_path)
                if not result.success:
                    print("  -> 点击失败")
                    continue
            except Exception:
                ok, info = device.click_node(
                    button.name,
                    list(button.pos) if button.pos else None,
                    screen_size,
                )
                if not ok:
                    print(f"  -> 兜底点击失败：{info}")
                    continue

            time.sleep(exploration.action_wait_s)

            # Check if app crashed
            if not device.app_is_running(game_config.package_name):
                print("[explore] 应用疑似崩溃，正在重启...")
                device.start_app(game_config.activity_name)
                time.sleep(exploration.boot_wait_s)
                device.reconnect()
                break

            # Check for page change
            new_hierarchy = device.dump_hierarchy()
            if new_hierarchy:
                new_perception = PocoTreeExtractor(device, config.perception).extract()
                new_hash = new_perception.page_hash

                execution = {
                    "step": step_counter,
                    "button_name": button.name,
                    "button_text": button.text or "",
                    "button_label": label,
                    "action_reason": reason,
                    "page_before": page_hash,
                    "page_after": new_hash,
                    "page_changed": new_hash != page_hash,
                    "timestamp": _utc_iso(),
                }
                executions.append(execution)

                if new_hash != page_hash:
                    print(f"  -> 页面已变化（新 hash={new_hash[:8]}）")
                    from game_agent.graph.models import EdgeAction, GraphEdge
                    edge = GraphEdge(
                        source_id=page_hash,
                        target_id=new_hash,
                        actions=[EdgeAction("poco_click", button.poco_path, {})],
                        discovered_at=time.time(),
                    )
                    graph.add_edge(edge)
                    storage.save_edge(edge)
                    acted = True
                    break
                else:
                    print("  -> 页面未变化")

        if not acted:
            print("[explore] 未检测到页面跳转，执行返回...")
            device.press_back()
            time.sleep(exploration.action_wait_s)
            step_counter += 1

    # Save results
    print("\n[explore] ========== 探索完成 ==========")
    print(f"  总步数：    {step_counter}")
    print(f"  页面数量：  {pages_found}")
    print(f"  图节点数：  {graph.node_count()}")
    print(f"  图边数：    {graph.edge_count()}")

    storage.export_json(str(output_dir / "state_graph.json"))

    (output_dir / "executions.json").write_text(
        json.dumps(executions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "status": "已完成",
        "total_steps": step_counter,
        "pages_found": pages_found,
        "graph_nodes": graph.node_count(),
        "graph_edges": graph.edge_count(),
        "page_annotations": page_annotations,
        "started_at": _utc_iso(),
    }
    (output_dir / "exploration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[explore] 输出已保存到：{output_dir}")
    print(f"  状态图：    {output_dir / 'state_graph.json'}")
    print(f"  执行记录：  {output_dir / 'executions.json'}")
    print(f"  汇总：      {output_dir / 'exploration_summary.json'}")
    print(f"  截图目录：  {screenshot_dir}")

    storage.close()


# ---------------------------------------------------------------------------
# Mock exploration (for testing)
# ---------------------------------------------------------------------------

def create_mock_scenario() -> list[MockScreen]:
    main_nodes = [
        PocoNode(name="btn_hero", type="Button", text="英雄", visible=True,
                 pos=(0.1, 0.9), poco_path="Root > Main > btn_hero"),
        PocoNode(name="btn_shop", type="Button", text="商店", visible=True,
                 pos=(0.3, 0.9), poco_path="Root > Main > btn_shop"),
        PocoNode(name="btn_battle", type="Button", text="战斗", visible=True,
                 pos=(0.5, 0.9), poco_path="Root > Main > btn_battle"),
    ]
    hero_nodes = [
        PocoNode(name="btn_back", type="Button", text="返回", visible=True,
                 pos=(0.05, 0.05), poco_path="Root > Hero > btn_back"),
        PocoNode(name="btn_upgrade", type="Button", text="升级", visible=True,
                 pos=(0.5, 0.8), poco_path="Root > Hero > btn_upgrade"),
    ]
    shop_nodes = [
        PocoNode(name="btn_back", type="Button", text="返回", visible=True,
                 pos=(0.05, 0.05), poco_path="Root > Shop > btn_back"),
        PocoNode(name="btn_buy_gems", type="Button", text="购买宝石", visible=True,
                 pos=(0.5, 0.5), poco_path="Root > Shop > btn_buy_gems"),
    ]
    return [
        MockScreen(poco_tree=main_nodes),
        MockScreen(poco_tree=hero_nodes),
        MockScreen(poco_tree=main_nodes),
        MockScreen(poco_tree=shop_nodes),
        MockScreen(poco_tree=main_nodes),
    ]


async def run_mock_exploration(config) -> None:
    device = MockDevice()
    device.load_scenario(create_mock_scenario())
    logger.info("使用模拟设备")

    tree_extractor = PocoTreeExtractor(device, config.perception)
    hasher = PageHasher(config.graph.hash_algorithm)
    storage = GraphStorage(config.graph.db_path)
    graph = storage.load_graph()
    gemini = GeminiClient(config.gemini)
    annotator = PageAnnotator(gemini)

    explorer = GraphExplorer(
        device=device,
        tree_extractor=tree_extractor,
        hasher=hasher,
        annotator=annotator,
        graph=graph,
        storage=storage,
        config=config.graph,
    )

    print("开始模拟冷启动探索...")
    report = await explorer.explore(max_depth=config.graph.exploration_max_depth)

    print("\n--- 探索报告 ---")
    print(f"发现节点：    {report.nodes_discovered}")
    print(f"发现边：      {report.edges_discovered}")
    print(f"重复访问节点：{report.nodes_revisited}")
    print(f"最大深度：    {report.max_depth_reached}")
    if report.errors:
        print(f"错误数：      {len(report.errors)}")
    print(f"\n图总计：{graph.node_count()} 个节点，{graph.edge_count()} 条边")
    storage.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    _ensure_utf8_console()
    load_env_file(_PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="冷启动 UI 探索器")
    parser.add_argument("--config", default=None, help="配置 YAML 路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟设备")
    parser.add_argument("--max-steps", type=int, default=None, help="最大探索步数")
    parser.add_argument("--max-pages", type=int, default=None, help="最多发现页面数")
    parser.add_argument("--export", default=None, help="导出图数据到 JSON 文件")
    parser.add_argument("--no-vision", action="store_true", help="禁用视觉 LLM")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mock:
        config.device.use_mock = True
    if args.max_steps:
        config.exploration.max_steps = args.max_steps
    if args.max_pages:
        config.exploration.max_pages = args.max_pages

    setup_logging(level_override=config.log_level)

    if config.device.use_mock:
        await run_mock_exploration(config)
        return

    # Real device mode
    print(f"[cold-start] 配置：{args.config or '默认'}")
    print(f"[cold-start] 游戏：{config.game.project_name or config.game.package_name}")

    _detect_runtime(config.game)

    # Initialize vision client
    vision_client = None
    if config.vision.enabled and not args.no_vision:
        api_key = config.vision.api_key
        if api_key:
            print(f"[cold-start] 已启用视觉 LLM（模型={config.vision.model_name}，密钥={api_key[:5]}***）")
            vision_client = VisionLLMClient(config.vision)
        else:
            print("[cold-start] 未提供视觉 LLM 的 API Key，将仅使用规则策略")
    else:
        print("[cold-start] 视觉 LLM 已禁用")

    await run_real_exploration(config, vision_client)


if __name__ == "__main__":
    asyncio.run(main())
