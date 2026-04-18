"""ReAct reasoning loop — the core Thought -> Action -> Observation orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from game_agent.cognition import prompts
from game_agent.cognition.context import ContextManager, ReActStep
from game_agent.cognition.gemini_client import GeminiClient
from game_agent.config import AppConfig
from game_agent.perception.base import PerceptionProvider
from game_agent.perception.state import PerceptionState
from game_agent.perception.ui_diff import UIDiffCalculator
from game_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

L2_ESCALATION_FAILURE_THRESHOLD = 3


@dataclass
class AgentResult:
    """Final result of an agent task execution."""

    success: bool
    steps: list[ReActStep] = field(default_factory=list)
    final_message: str = ""
    total_steps: int = 0


class ReActLoop:
    """Orchestrates the ReAct cycle: perceive -> think -> act -> observe."""

    def __init__(
        self,
        gemini: GeminiClient,
        tool_registry: ToolRegistry,
        perception: PerceptionProvider,
        context: ContextManager,
        ui_diff: UIDiffCalculator,
        config: AppConfig,
    ) -> None:
        self._gemini = gemini
        self._tools = tool_registry
        self._perception = perception
        self._context = context
        self._ui_diff = ui_diff
        self._config = config

    @staticmethod
    def _format_thought_for_log(raw_text: str | None) -> str:
        if not raw_text:
            return "（模型未返回显式思考文本）"

        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
        candidates = [fenced.group(1).strip()] if fenced else []
        candidates.append(text)

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            thought = payload.get("thought") or payload.get("思考")
            if isinstance(thought, str) and thought.strip():
                return thought.strip()

        compact = re.sub(r"\s+", " ", text)
        return compact[:400] + ("..." if len(compact) > 400 else "")

    async def run(self, task: str, max_steps: int = 50) -> AgentResult:
        """Execute a task through iterative ReAct reasoning."""
        self._context.clear()
        self._gemini.start_chat(
            system_prompt=prompts.SYSTEM_PROMPT,
            tools=self._tools.get_gemini_tools(),
        )

        prev_state: PerceptionState | None = None
        action_count = 0
        loading_wait_count = 0

        while action_count < max_steps:
            perception_data = self._perception.capture_l1()
            curr_state = self._perception.to_state(perception_data)
            logger.info(
                "步骤 %d：页面=%s，可交互节点=%d",
                action_count,
                curr_state.page_hash[:8],
                len(perception_data.interactive_nodes),
            )

            if prev_state is not None:
                diff = self._perception.compute_diff(prev_state, curr_state)
                if diff.is_loading:
                    loading_wait_count += 1
                    logger.info(
                        "界面疑似仍在加载，等待后重试（页面=%s，连续等待=%d）",
                        curr_state.page_hash[:8],
                        loading_wait_count,
                    )
                    await asyncio.sleep(1.0)
                    continue
                if loading_wait_count:
                    logger.info("界面加载等待结束，继续规划动作")
                    loading_wait_count = 0

            use_l2 = self._context.consecutive_failures() >= L2_ESCALATION_FAILURE_THRESHOLD

            prompt_text = prompts.SYSTEM_PROMPT.format(
                perception_text=perception_data.poco_tree_markdown,
                context_window=self._context.to_prompt_text(),
                task_description=task,
            )

            if use_l2:
                logger.info("连续失败次数过多，升级到 L2 感知")
                l2_data = self._perception.capture_l2()
                response = self._gemini.send_multimodal(
                    prompt_text, l2_data.screenshot_b64
                )
            else:
                response = self._gemini.send_message(prompt_text)

            logger.info(
                "步骤 %d：思考=%s",
                action_count,
                self._format_thought_for_log(response.text),
            )

            if not response.function_calls:
                logger.info("代理已完成任务：%s", response.text)
                return AgentResult(
                    success=True,
                    steps=list(self._context._steps),
                    final_message=response.text or "任务已完成",
                    total_steps=action_count,
                )

            fc = response.function_calls[0]
            logger.info("步骤 %d：执行 %s(%s)", action_count, fc.name, fc.args)
            tool_result = self._tools.execute(fc.name, fc.args)
            logger.info(
                "步骤 %d：动作结果 success=%s，observation=%s",
                action_count,
                tool_result.success,
                tool_result.message,
            )

            step = ReActStep(
                step_number=action_count,
                thought=response.text or "",
                action_name=fc.name,
                action_params=fc.args,
                observation=tool_result.message,
                timestamp=time.time(),
            )
            self._context.add_step(step)
            prev_state = curr_state
            action_count += 1

            await asyncio.sleep(0.3)

        return AgentResult(
            success=False,
            steps=list(self._context._steps),
            final_message=f"已达到最大步数（{max_steps}）",
            total_steps=action_count,
        )
