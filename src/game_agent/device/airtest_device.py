"""Real Airtest/Poco device implementation with multi-engine support."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

from game_agent.config import (
    ENGINE_ANDROID_UIAUTOMATION,
    ENGINE_COCOS2DX_JS,
    ENGINE_COCOS2DX_LUA,
    ENGINE_COCOS_CREATOR,
    ENGINE_UNITY3D,
    GameConnectionConfig,
)
from game_agent.device.base import DeviceController, PocoNode, TouchResult
from game_agent.exceptions import DeviceNotConnectedError, PerceptionError, PocoNodeNotFoundError

logger = logging.getLogger(__name__)

AIRTEST_IMPORT_ERROR: Exception | None = None

try:
    from airtest.core.api import connect_device, snapshot, swipe as airtest_swipe, touch

    HAS_AIRTEST = True
except Exception as exc:
    HAS_AIRTEST = False
    AIRTEST_IMPORT_ERROR = exc


class AirtestDevice(DeviceController):
    """DeviceController backed by real Airtest + Poco libraries.

    Supports Unity, CocosJS, CocosLua, and Android UIAutomation engines.
    """

    def __init__(self, config: GameConnectionConfig) -> None:
        if not HAS_AIRTEST:
            detail = f"（导入异常：{AIRTEST_IMPORT_ERROR}）" if AIRTEST_IMPORT_ERROR else ""
            raise DeviceNotConnectedError(
                "Airtest/Poco 不可用，请先确认已安装且依赖兼容："
                " pip install airtest pocoui"
                f"{detail}"
            )
        self._config = config
        self._device: Any = None
        self._poco: Any = None
        self._adb_path: str | None = None

    def connect(self) -> None:
        """Connect to device and initialize Poco."""
        from airtest.core.api import connect_device as airtest_connect

        self._device = airtest_connect(self._config.device_uri)
        self._ensure_poco_forward()
        self._poco = self._create_poco()
        logger.info(
            "已连接设备 %s（引擎=%s，Poco=%s:%d）",
            self._config.device_serial,
            self._config.engine_type,
            self._config.poco_host,
            self._config.effective_poco_port(),
        )
        self._warmup_touch()

    def reconnect(self) -> None:
        """Reconnect after RPC failure."""
        from airtest.core.api import connect_device as airtest_connect

        self._device = airtest_connect(self._config.device_uri)
        self._ensure_poco_forward()
        self._poco = self._create_poco()
        logger.info("已重新连接到 %s", self._config.device_serial)
        self._warmup_touch()

    def _create_poco(self) -> Any:
        engine = self._config.engine_type
        host = self._config.poco_host
        port = self._config.effective_poco_port()

        if engine == ENGINE_UNITY3D:
            from poco.drivers.unity3d import UnityPoco
            return UnityPoco(addr=(host, port), device=self._device)

        if engine in {ENGINE_COCOS_CREATOR, ENGINE_COCOS2DX_JS}:
            from poco.drivers.cocosjs import CocosJsPoco
            return CocosJsPoco(addr=(host, port), device=self._device)

        if engine == ENGINE_COCOS2DX_LUA:
            from poco.drivers.std import StdPoco
            return StdPoco(port=port, device=self._device, use_airtest_input=True)

        if engine == ENGINE_ANDROID_UIAUTOMATION:
            from poco.drivers.android.uiautomation import AndroidUiautomationPoco
            return AndroidUiautomationPoco(
                use_airtest_input=True, screenshot_each_action=False,
            )

        from poco.drivers.android.uiautomation import AndroidUiautomationPoco
        return AndroidUiautomationPoco(
            use_airtest_input=True, screenshot_each_action=False,
        )

    # ---- ADB helpers ----

    def _get_adb(self) -> str:
        if self._adb_path:
            return self._adb_path
        try:
            import airtest
            adb_exe = (
                Path(airtest.__file__).resolve().parent
                / "core" / "android" / "static" / "adb" / "windows" / "adb.exe"
            )
            if adb_exe.exists():
                self._adb_path = str(adb_exe)
                return self._adb_path
        except Exception:
            pass
        self._adb_path = "adb"
        return self._adb_path

    def _adb_cmd(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self._get_adb(), "-s", self._config.device_serial, *args],
            check=False, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )

    def _adb_output(self, *args: str) -> str:
        result = self._adb_cmd(*args)
        return (result.stdout or result.stderr or "").strip()

    def _ensure_poco_forward(self) -> None:
        port = self._config.effective_poco_port()
        if port <= 0 or self._config.engine_type == ENGINE_ANDROID_UIAUTOMATION:
            return
        self._adb_cmd("forward", f"tcp:{port}", f"tcp:{port}")

    def _warmup_touch(self) -> None:
        """Eagerly initialize Airtest touch subsystem (rotationwatcher + maxtouch)."""
        dev = self._device
        if dev is None:
            return
        try:
            if hasattr(dev, "display_info"):
                _ = dev.display_info
            if hasattr(dev, "minitouch"):
                _ = dev.minitouch
            elif hasattr(dev, "touch_proxy"):
                _ = dev.touch_proxy
            logger.info("Airtest 触控子系统预热完成")
        except Exception as exc:
            logger.debug("Airtest 触控预热跳过：%s", exc)

    # ---- DeviceController implementation ----

    def click(self, pos: tuple[float, float]) -> TouchResult:
        touch(pos)
        return TouchResult(success=True, timestamp=time.time())

    def click_poco(self, poco_path: str) -> TouchResult:
        if self._poco is None:
            raise DeviceNotConnectedError("Poco 尚未连接，请先调用 connect()")
        try:
            node = self._poco(poco_path)
            node.click()
            return TouchResult(success=True, timestamp=time.time())
        except Exception as exc:
            raise PocoNodeNotFoundError(f"未找到节点：{poco_path}") from exc

    def swipe(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        duration: float = 0.5,
    ) -> TouchResult:
        airtest_swipe(start, end, duration=duration)
        return TouchResult(success=True, timestamp=time.time())

    def long_press(self, pos: tuple[float, float], duration: float = 1.0) -> TouchResult:
        touch(pos, duration=duration)
        return TouchResult(success=True, timestamp=time.time())

    def wait_for_node(self, poco_path: str, timeout: float = 10.0) -> bool:
        if self._poco is None:
            return False
        try:
            self._poco(poco_path).wait_for_appearance(timeout=timeout)
            return True
        except Exception:
            return False

    def node_exists(self, poco_path: str) -> bool:
        if self._poco is None:
            return False
        try:
            return self._poco(poco_path).exists()
        except Exception:
            return False

    def get_poco_tree(self) -> list[PocoNode]:
        hierarchy = self.dump_hierarchy()
        if hierarchy is None:
            return []
        return self._flatten_tree(hierarchy)

    def screenshot(self) -> bytes:
        import io
        from PIL import Image

        screen = snapshot(filename=None)
        if screen is None:
            logger.warning("Airtest snapshot 返回空数据，尝试使用 ADB screencap 兜底")
            fallback = self._adb_screenshot()
            if fallback:
                return fallback
            raise PerceptionError("设备截图失败：Airtest snapshot 返回空数据")
        buf = io.BytesIO()
        if isinstance(screen, Image.Image):
            screen.save(buf, format="PNG")
        elif isinstance(screen, (bytes, bytearray)):
            buf.write(screen)
        else:
            logger.warning("Airtest snapshot 返回了未知类型 %s，尝试使用 ADB screencap 兜底", type(screen).__name__)
            fallback = self._adb_screenshot()
            if fallback:
                return fallback
            raise PerceptionError(f"设备截图失败：snapshot 返回未知类型 {type(screen).__name__}")
        return buf.getvalue()

    def _adb_screenshot(self) -> bytes | None:
        result = subprocess.run(
            [self._get_adb(), "-s", self._config.device_serial, "exec-out", "screencap", "-p"],
            check=False,
            capture_output=True,
        )
        data = bytes(result.stdout or b"")
        if result.returncode == 0 and data.startswith(b"\x89PNG"):
            return data

        temp_path = "/sdcard/temp_screen_agent.png"
        self._adb_cmd("shell", "rm", "-f", temp_path)
        capture = self._adb_cmd("shell", "screencap", "-p", temp_path)
        if capture.returncode != 0:
            return None

        local_path = Path("data") / "temp_screen_agent.png"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        pull = subprocess.run(
            [self._get_adb(), "-s", self._config.device_serial, "pull", temp_path, str(local_path)],
            check=False,
            capture_output=True,
        )
        if pull.returncode != 0 or not local_path.exists():
            return None

        try:
            data = local_path.read_bytes()
            return data if data.startswith(b"\x89PNG") else None
        finally:
            try:
                local_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._adb_cmd("shell", "rm", "-f", temp_path)

    def get_screen_size(self) -> tuple[int, int]:
        result = self._adb_cmd("shell", "wm", "size")
        output = result.stdout.strip()
        for token in output.replace("Physical size:", "").split():
            if "x" in token:
                w, h = token.split("x", 1)
                if w.isdigit() and h.isdigit():
                    return int(w), int(h)
        return 1440, 2560

    def dump_hierarchy(self, retries: int = 3, wait_s: float = 1.0) -> dict[str, Any] | None:
        if self._poco is None:
            return None
        for _ in range(retries):
            try:
                return self._poco.freeze().agent.hierarchy.dump()
            except Exception:
                time.sleep(wait_s)
        return None

    def press_back(self) -> None:
        self._adb_cmd("shell", "input", "keyevent", "4")

    def app_is_running(self, package_name: str) -> bool:
        result = self._adb_cmd("shell", "pidof", package_name)
        return bool(result.stdout.strip())

    def force_stop(self, package_name: str) -> None:
        self._adb_cmd("shell", "am", "force-stop", package_name)

    def start_app(self, activity: str) -> None:
        self._adb_cmd("shell", "am", "start", "-n", activity)

    def snapshot_to_file(self, save_path: str) -> bool:
        try:
            temp_path = "/sdcard/temp_screen_agent.png"
            self._adb_cmd("shell", "screencap", "-p", temp_path)
            subprocess.run(
                [self._get_adb(), "-s", self._config.device_serial, "pull", temp_path, save_path],
                check=False, capture_output=True,
            )
            return Path(save_path).exists()
        except Exception:
            return False

    def click_node(
        self, name: str, pos: list[float] | None, screen_size: tuple[int, int],
    ) -> tuple[bool, str]:
        """Click a node by Poco name, falling back to ADB tap by position."""
        if name and self._poco is not None:
            try:
                node = self._poco(name)
                if node.exists():
                    node.click()
                    return True, f"poco:{name}"
            except Exception:
                pass

        if pos and isinstance(pos, list) and len(pos) == 2:
            x, y = pos
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                px = max(1, min(screen_size[0] - 1, int(x * screen_size[0])))
                py = max(1, min(screen_size[1] - 1, int(y * screen_size[1])))
                result = self._adb_cmd("shell", "input", "tap", str(px), str(py))
                if result.returncode == 0:
                    return True, f"tap:{px},{py}"
                return False, "adb_tap_failed"

        return False, "no_valid_target"

    def _flatten_tree(
        self, node: dict, parent_path: str = "", result: list[PocoNode] | None = None,
    ) -> list[PocoNode]:
        if result is None:
            result = []
        payload = node.get("payload", {})
        name = payload.get("name", "")
        current_path = f"{parent_path} > {name}" if parent_path else name
        children = node.get("children", [])
        result.append(
            PocoNode(
                name=name,
                type=payload.get("type", ""),
                text=payload.get("text"),
                visible=payload.get("visible", True),
                pos=tuple(payload.get("pos", [0.0, 0.0])),
                size=tuple(payload.get("size", [0.0, 0.0])),
                children_count=len(children),
                payload=payload,
                poco_path=current_path,
            )
        )
        for child in children:
            self._flatten_tree(child, current_path, result)
        return result
