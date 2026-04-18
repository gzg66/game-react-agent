"""Central tool registry mapping tool names to implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from game_agent.tools.schemas import ToolResult, pydantic_to_gemini_declaration

logger = logging.getLogger(__name__)


@dataclass
class ToolEntry:
    """A registered tool."""

    name: str
    fn: Callable[..., ToolResult]
    schema: type[BaseModel]
    description: str


class ToolRegistry:
    """Maps tool names to (callable, schema, description).

    Used by the ReAct loop to dispatch actions and by the Gemini client
    to build the tools parameter.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolEntry] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., ToolResult],
        schema: type[BaseModel],
        description: str = "",
    ) -> None:
        desc = description or schema.__doc__ or name
        self._tools[name] = ToolEntry(name=name, fn=fn, schema=schema, description=desc)
        logger.debug("已注册工具：%s", name)

    def execute(self, name: str, params: dict[str, Any]) -> ToolResult:
        if name not in self._tools:
            return ToolResult(success=False, message=f"未知工具：{name}")
        entry = self._tools[name]
        try:
            validated = entry.schema(**params)
            return entry.fn(validated)
        except Exception as exc:
            logger.exception("工具 %s 执行失败", name)
            return ToolResult(success=False, message=f"工具 {name} 执行失败：{exc}")

    def get_gemini_tools(self) -> list[dict]:
        declarations = [
            pydantic_to_gemini_declaration(
                entry.schema,
                entry.name,
                entry.description,
            )
            for entry in self._tools.values()
        ]
        if not declarations:
            return []
        return [{"function_declarations": declarations}]

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> ToolEntry | None:
        return self._tools.get(name)
