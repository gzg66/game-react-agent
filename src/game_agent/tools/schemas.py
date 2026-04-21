"""Pydantic models for tool inputs/outputs and Gemini FunctionDeclaration generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


# --- Tool result ---

@dataclass
class ToolResult:
    """Standardized output from any tool execution."""

    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


# --- Atomic tool input schemas ---

class PocoClickInput(BaseModel):
    """按节点名称点击界面元素。"""

    node_name: str = Field(
        description="目标按钮或节点名称（如 btnLogin、btnStartGame），无需完整路径"
    )


class AirtestTouchPosInput(BaseModel):
    """按相对坐标点击指定屏幕位置。"""

    x: float = Field(ge=0.0, le=1.0, description="相对 X 坐标（0-1）")
    y: float = Field(ge=0.0, le=1.0, description="相对 Y 坐标（0-1）")


class SwipeInput(BaseModel):
    """从一个位置滑动到另一个位置。"""

    start_x: float = Field(ge=0.0, le=1.0)
    start_y: float = Field(ge=0.0, le=1.0)
    end_x: float = Field(ge=0.0, le=1.0)
    end_y: float = Field(ge=0.0, le=1.0)
    duration: float = Field(default=0.5, ge=0.1, le=5.0, description="滑动持续时间（秒）")


class WaitForNodeInput(BaseModel):
    """等待 Poco 节点出现在屏幕上。"""

    poco_path: str = Field(description="需要等待的 Poco 路径")
    timeout: float = Field(default=10.0, ge=1.0, le=60.0, description="最长等待时间（秒）")


class MapsToInput(BaseModel):
    """使用缓存图路径导航到目标页面。"""

    target_node_id: str = Field(description="要导航到的图节点 ID（页面哈希）")


class ClearAllPopupsInput(BaseModel):
    """关闭所有可见弹窗或对话框。"""

    max_attempts: int = Field(default=5, ge=1, le=20, description="最多尝试关闭弹窗的次数")


# --- Gemini FunctionDeclaration converter ---

def pydantic_to_gemini_declaration(
    model_cls: type[BaseModel],
    fn_name: str,
    description: str | None = None,
) -> dict:
    """Convert a Pydantic model to a Gemini-compatible function declaration dict."""
    schema = model_cls.model_json_schema()
    properties = {}
    required = schema.get("required", [])

    for prop_name, prop_schema in schema.get("properties", {}).items():
        prop_type = prop_schema.get("type", "string")
        gemini_type = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
        }.get(prop_type, "STRING")

        properties[prop_name] = {
            "type": gemini_type,
            "description": prop_schema.get("description", ""),
        }

    return {
        "name": fn_name,
        "description": description or model_cls.__doc__ or fn_name,
        "parameters": {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
        },
    }
