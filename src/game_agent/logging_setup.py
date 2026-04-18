"""Configure stdlib logging from YAML config."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOGGING_CONFIG = _PROJECT_ROOT / "config" / "logging.yaml"


def setup_logging(
    config_path: Path | str | None = None,
    level_override: str | None = None,
) -> None:
    """Initialize logging from YAML config file."""
    path = Path(config_path) if config_path else DEFAULT_LOGGING_CONFIG

    if path.exists():
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        log_dir = Path(config.get("handlers", {}).get("file", {}).get("filename", "data/agent.log")).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, encoding="utf-8")

    if level_override:
        logging.getLogger("game_agent").setLevel(getattr(logging, level_override.upper(), logging.INFO))
