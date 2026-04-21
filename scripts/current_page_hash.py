"""Capture the current game page and print its page hash."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCREENSHOT_DIR = SCRIPT_DIR / "page_screenshots"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from game_agent.config import load_config, load_env_file  # noqa: E402
from game_agent.device.airtest_device import AirtestDevice  # noqa: E402
from game_agent.device.mock_device import MockDevice  # noqa: E402
from game_agent.logging_setup import setup_logging  # noqa: E402
from game_agent.perception.provider import DefaultPerceptionProvider  # noqa: E402

logger = logging.getLogger(__name__)


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
        description="获取当前游戏页面的 page hash",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="./config/xttc_poco.yaml", help="配置 YAML 路径")
    parser.add_argument("--mock", action="store_true", default=False, help="使用模拟设备")
    parser.add_argument("--device-uri", default=None, help="显式指定 Airtest 设备 URI")
    parser.add_argument("--serial", default=None, help="显式指定设备串号")
    parser.add_argument(
        "--screenshot-dir",
        default=str(DEFAULT_SCREENSHOT_DIR),
        help="截图保存目录，默认保存在 scripts/page_screenshots",
    )
    parser.add_argument(
        "--show-markdown",
        action="store_true",
        default=True,
        help="额外输出当前页面的交互节点 Markdown",
    )
    parser.add_argument(
        "--show-nodes",
        action="store_true",
        default=True,
        help="额外输出当前页面的交互节点明细",
    )
    parser.add_argument(
        "--hide-markdown",
        action="store_false",
        dest="show_markdown",
        help="不输出当前页面的交互节点 Markdown",
    )
    parser.add_argument(
        "--hide-nodes",
        action="store_false",
        dest="show_nodes",
        help="不输出当前页面的交互节点明细",
    )
    return parser


def _normalize_platform(platform: str) -> str:
    normalized = platform.strip().lower()
    if normalized == "android":
        return "Android"
    if normalized == "ios":
        return "iOS"
    return platform.strip() or "Android"


def _resolve_real_device_config(config, args) -> None:
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
                "未配置设备连接信息：请通过 `--config` 提供包含 `game.device_uri` 的配置，"
                "或直接传入 `--device-uri Android:///127.0.0.1:16384`。"
            )
        platform = _normalize_platform(config.device.platform)
        config.game.device_uri = f"{platform}:///{serial}"

    logger.info(
        "当前设备连接配置：device_uri=%s, device_serial=%s",
        config.game.device_uri,
        config.game.device_serial or "（空）",
    )


def _save_page_screenshot(device, page_hash: str, screenshot_dir: str) -> Path:
    output_dir = Path(screenshot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / f"{page_hash}.png"
    screenshot_bytes = device.screenshot()
    if not screenshot_bytes:
        raise SystemExit("截图失败：设备未返回有效 PNG 数据。")
    screenshot_path.write_bytes(screenshot_bytes)
    logger.info("当前页面截图已保存：%s", screenshot_path)
    return screenshot_path


def main() -> int:
    _ensure_utf8_console()
    load_env_file(PROJECT_ROOT / ".env")
    args = _build_arg_parser().parse_args()
    config = load_config(args.config)
    # 该脚本默认面向真实设备当前页面；如需模拟环境，请显式传入 --mock。
    config.device.use_mock = False
    if args.mock:
        config.device.use_mock = True
    setup_logging(level_override=config.log_level)

    if config.device.use_mock:
        device = MockDevice()
        logger.info("使用模拟设备获取当前页面 hash")
    else:
        _resolve_real_device_config(config, args)
        device = AirtestDevice(config.game)
        device.connect()
        logger.info("已连接真实设备，开始采集当前页面 hash")

    perception = DefaultPerceptionProvider(device, config.perception)
    snapshot = perception.capture_l1()
    screenshot_path = _save_page_screenshot(
        device,
        snapshot.page_hash,
        args.screenshot_dir,
    )

    print(f"page_hash={snapshot.page_hash}")
    print(f"interactive_nodes={len(snapshot.interactive_nodes)}")
    print(f"screenshot={screenshot_path}")

    if args.show_nodes:
        print("nodes:")
        for node in snapshot.interactive_nodes:
            label = f' text="{node.text}"' if node.text else ""
            print(f"- {node.type} {node.name}{label} path={node.poco_path}")

    if args.show_markdown:
        print("\nmarkdown:")
        print(snapshot.poco_tree_markdown or "（空）")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
