"""Application configuration using Pydantic models with YAML + env var support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from game_agent.exceptions import ConfigError

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "default.yaml"

ENGINE_UNITY3D = "unity3d"
ENGINE_COCOS_CREATOR = "cocos_creator"
ENGINE_COCOS2DX_JS = "cocos2dx_js"
ENGINE_COCOS2DX_LUA = "cocos2dx_lua"
ENGINE_ANDROID_UIAUTOMATION = "android_uiautomation"

DEFAULT_POCO_PORTS: dict[str, int] = {
    ENGINE_UNITY3D: 5001,
    ENGINE_COCOS_CREATOR: 5003,
    ENGINE_COCOS2DX_JS: 5003,
    ENGINE_COCOS2DX_LUA: 15004,
    ENGINE_ANDROID_UIAUTOMATION: 0,
}


class DeviceConfig(BaseModel):
    platform: Literal["android", "ios"] = "android"
    serial: str | None = None
    use_mock: bool = False


class GameConnectionConfig(BaseModel):
    """Game-specific connection parameters."""

    project_name: str = ""
    engine_type: str = ENGINE_COCOS2DX_JS
    package_name: str = ""
    activity_name: str = ""
    device_uri: str = "Android:///127.0.0.1:16384"
    device_serial: str = "127.0.0.1:16384"
    poco_host: str = "127.0.0.1"
    poco_port: int = 5003

    def effective_poco_port(self) -> int:
        if self.poco_port > 0:
            return self.poco_port
        return DEFAULT_POCO_PORTS.get(self.engine_type, 5001)


class ExplorationConfig(BaseModel):
    """Cold-start exploration parameters."""

    max_steps: int = 200
    max_pages: int = 30
    max_actions_per_page: int = 20
    boot_wait_s: float = 10.0
    action_wait_s: float = 2.0
    no_new_page_limit: int = 10
    output_dir: str = "outputs/cold_start"


class SafetyConfig(BaseModel):
    """Safety guardrails for automated exploration."""

    dangerous_keywords: list[str] = Field(default_factory=lambda: [
        "充值", "支付", "购买", "删除", "删除账号", "退出登录",
        "recharge", "pay", "purchase", "delete",
    ])
    safe_priority_keywords: list[str] = Field(default_factory=lambda: [
        "关闭", "确认", "确定", "下一步", "开始", "进入", "领取", "跳过", "返回",
        "大厅", "冒险", "出战", "挑战", "自动", "结算",
        "close", "confirm", "next", "start", "enter", "claim", "skip", "back", "ok",
    ])


class VisionConfig(BaseModel):
    """Vision LLM configuration."""

    enabled: bool = True
    api_key: str = Field(default="")
    model_name: str = "gemini-3-flash-preview"
    mode: str = "vision_first"
    max_candidates: int = 16
    min_confidence: float = 0.55


class GeminiConfig(BaseModel):
    api_key: str = Field(default="")
    model_name: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    response_mime_type: str = "application/json"


class PerceptionConfig(BaseModel):
    diff_threshold: float = 0.05
    poco_tree_max_depth: int = 8
    screenshot_resize: tuple[int, int] = (720, 1280)


class GraphConfig(BaseModel):
    db_path: str = "data/graph.db"
    exploration_max_depth: int = 20
    hash_algorithm: str = "sha256"


class ContextConfig(BaseModel):
    window_size: int = 10


class NavigationMemoryConfig(BaseModel):
    file_path: str = "data/nav_memory.json"
    ineffective_threshold: int = 2
    staleness_threshold: int = 5


class PageCacheConfig(BaseModel):
    file_path: str = "data/page_cache.json"


class AppConfig(BaseModel):
    device: DeviceConfig = DeviceConfig()
    game: GameConnectionConfig = GameConnectionConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    safety: SafetyConfig = SafetyConfig()
    vision: VisionConfig = VisionConfig()
    gemini: GeminiConfig = GeminiConfig()
    perception: PerceptionConfig = PerceptionConfig()
    graph: GraphConfig = GraphConfig()
    context: ContextConfig = ContextConfig()
    navigation_memory: NavigationMemoryConfig = NavigationMemoryConfig()
    page_cache: PageCacheConfig = PageCacheConfig()
    log_level: str = "INFO"


def load_env_file(env_path: Path | str | None = None) -> None:
    """Load environment variables from a .env file without overriding existing."""
    path = Path(env_path) if env_path else _PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _flatten_game_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML sections into the game connection config."""
    game: dict[str, Any] = raw.pop("game", {}) or {}
    if "device" in raw and isinstance(raw["device"], dict):
        dev = raw["device"]
        if "uri" in dev:
            game.setdefault("device_uri", dev.pop("uri"))
        if "serial" in dev and "device_serial" not in game:
            game.setdefault("device_serial", dev["serial"])
    if "connection" in raw and isinstance(raw["connection"], dict):
        conn = raw.pop("connection")
        game.setdefault("poco_host", conn.get("host", "127.0.0.1"))
        game.setdefault("poco_port", conn.get("port", 0))
    for key in ("project_name", "engine_type", "package_name", "activity_name"):
        if key in raw:
            game.setdefault(key, raw.pop(key))
    return game


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Load configuration from YAML file with environment variable overrides."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigError(f"未找到配置文件：{path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    game_data = _flatten_game_config(raw)
    if game_data:
        raw["game"] = game_data

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if "gemini" in raw:
        raw["gemini"]["api_key"] = api_key
    else:
        raw["gemini"] = {"api_key": api_key}

    if "vision" not in raw:
        raw["vision"] = {}
    raw["vision"]["api_key"] = os.environ.get("LLM_API_KEY") or api_key

    env_log_level = os.environ.get("LOG_LEVEL")
    if env_log_level:
        raw["log_level"] = env_log_level

    return AppConfig(**raw)
