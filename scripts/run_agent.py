"""CLI entry point for the interactive ReAct agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from game_agent.cognition.context import ContextManager
from game_agent.cognition.gemini_client import GeminiClient
from game_agent.cognition.react_loop import ReActLoop
from game_agent.config import load_config, load_env_file
from game_agent.device.mock_device import MockDevice, MockScreen
from game_agent.device.base import PocoNode
from game_agent.graph.hasher import PageHasher
from game_agent.graph.models import UIStateGraph
from game_agent.graph.navigator import GraphNavigator
from game_agent.graph.storage import GraphStorage
from game_agent.logging_setup import setup_logging
from game_agent.perception.provider import DefaultPerceptionProvider
from game_agent.perception.ui_diff import UIDiffCalculator
from game_agent.tools.atomic import build_atomic_tools
from game_agent.tools.macros import build_macro_tools
from game_agent.tools.registry import ToolRegistry

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


def create_mock_scenario() -> list[MockScreen]:
    """Create a simple mock scenario for demo/testing."""
    main_city_nodes = [
        PocoNode(name="btn_hero", type="Button", text="英雄", visible=True,
                 pos=(0.1, 0.9), poco_path="Root > MainPanel > btn_hero"),
        PocoNode(name="btn_shop", type="Button", text="商店", visible=True,
                 pos=(0.3, 0.9), poco_path="Root > MainPanel > btn_shop"),
        PocoNode(name="btn_battle", type="Button", text="战斗", visible=True,
                 pos=(0.5, 0.9), poco_path="Root > MainPanel > btn_battle"),
        PocoNode(name="btn_backpack", type="Button", text="背包", visible=True,
                 pos=(0.7, 0.9), poco_path="Root > MainPanel > btn_backpack"),
        PocoNode(name="txt_gold", type="Text", text="1000", visible=True,
                 pos=(0.9, 0.05), poco_path="Root > TopBar > txt_gold"),
    ]

    hero_nodes = [
        PocoNode(name="btn_back", type="Button", text="返回", visible=True,
                 pos=(0.05, 0.05), poco_path="Root > HeroPanel > btn_back"),
        PocoNode(name="btn_upgrade", type="Button", text="升级", visible=True,
                 pos=(0.5, 0.8), poco_path="Root > HeroPanel > btn_upgrade"),
        PocoNode(name="txt_hero_name", type="Text", text="战士", visible=True,
                 pos=(0.5, 0.2), poco_path="Root > HeroPanel > txt_hero_name"),
    ]

    return [
        MockScreen(poco_tree=main_city_nodes, node_existence={"btn_hero": True}),
        MockScreen(poco_tree=hero_nodes, node_existence={"btn_back": True}),
        MockScreen(poco_tree=main_city_nodes),
    ]


async def main() -> None:
    _ensure_utf8_console()
    load_env_file(_PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="游戏 ReAct 智能体")
    parser.add_argument("--config", default=None, help="配置 YAML 路径")
    parser.add_argument("--mock", action="store_true", help="使用模拟设备")
    parser.add_argument(
        "--no-restart-game",
        action="store_true",
        help="真实设备模式下启动前不重启游戏",
    )
    parser.add_argument("--task", default=None, help="要执行的任务")
    parser.add_argument("--max-steps", type=int, default=50, help="ReAct 最大步数")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mock:
        config.device.use_mock = True
    setup_logging(level_override=config.log_level)

    if config.device.use_mock:
        device = MockDevice()
        device.load_scenario(create_mock_scenario())
        logger.info("使用模拟设备，共加载 %d 个页面", len(device._screens))
    else:
        from game_agent.device.airtest_device import AirtestDevice
        device = AirtestDevice(config.game)
        if not args.no_restart_game:
            logger.info("正在重启游戏：%s", config.game.package_name)
            device.force_stop(config.game.package_name)
            await asyncio.sleep(1.0)
            device.start_app(config.game.activity_name)
            logger.info(
                "游戏已启动，等待 %.1f 秒完成加载",
                config.exploration.boot_wait_s,
            )
            await asyncio.sleep(config.exploration.boot_wait_s)
        device.connect()
        logger.info("使用真实设备，已建立 Poco 连接：%s", config.game.device_serial)

    perception = DefaultPerceptionProvider(device, config.perception)
    ui_diff = UIDiffCalculator(config.perception)

    storage = GraphStorage(config.graph.db_path)
    graph = storage.load_graph()
    navigator = GraphNavigator(graph)

    registry = ToolRegistry()
    for name, (fn, schema, desc) in build_atomic_tools(device).items():
        registry.register(name, fn, schema, desc)
    for name, (fn, schema, desc) in build_macro_tools(
        device, graph, navigator, perception.tree_extractor
    ).items():
        registry.register(name, fn, schema, desc)

    context = ContextManager(window_size=config.context.window_size)

    gemini = GeminiClient(config.gemini)
    react_loop = ReActLoop(gemini, registry, perception, context, ui_diff, config)

    logger.info("智能体已初始化，可用工具：%s", registry.list_tools())

    if args.task:
        result = await react_loop.run(args.task, max_steps=args.max_steps)
        print(f"\n结果：{'成功' if result.success else '失败'}")
        print(f"步数：{result.total_steps}")
        print(f"说明：{result.final_message}")
    else:
        print("游戏 ReAct 智能体 - 交互模式")
        print("请输入任务，输入 `退出` 可结束；同时兼容 `quit`、`exit`、`q`。\n")
        while True:
            try:
                task = input("任务> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not task or task.lower() in ("quit", "exit", "q") or task == "退出":
                break
            result = await react_loop.run(task, max_steps=args.max_steps)
            print(f"\n{'成功' if result.success else '失败'}：{result.final_message}")
            print(f"（共 {result.total_steps} 步）\n")

    storage.close()


if __name__ == "__main__":
    asyncio.run(main())
