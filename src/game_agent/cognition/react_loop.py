"""ReAct reasoning loop — the core Thought -> Action -> Observation orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from game_agent.cognition import prompts
from game_agent.cognition.context import ContextManager, ReActStep
from game_agent.cognition.gemini_client import GeminiClient
from game_agent.cognition.navigation_memory import NavigationMemory
from game_agent.cognition.page_cache import PageKnowledge, PageKnowledgeCache
from game_agent.config import AppConfig
from game_agent.graph.annotator import PageAnnotator
from game_agent.perception.base import PerceptionProvider
from game_agent.perception.state import PerceptionState
from game_agent.perception.ui_diff import UIDiffCalculator
from game_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

L2_ESCALATION_FAILURE_THRESHOLD = 3
PAGE_STALENESS_THRESHOLD = 5
CYCLE_DETECTION_MIN_REPETITIONS = 3
POST_ACTION_LOADING_RETRIES = 3
POST_ACTION_LOADING_WAIT_S = 1.0


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
        navigation_memory: NavigationMemory | None = None,
        page_cache: PageKnowledgeCache | None = None,
    ) -> None:
        self._gemini = gemini
        self._tools = tool_registry
        self._perception = perception
        self._context = context
        self._ui_diff = ui_diff
        self._config = config
        self._nav_memory = navigation_memory if navigation_memory is not None else NavigationMemory()
        self._page_cache = page_cache if page_cache is not None else PageKnowledgeCache()
        self._annotator = PageAnnotator(gemini)
        self._staleness_threshold = getattr(
            config, "navigation_memory", None
        )
        if self._staleness_threshold is not None:
            self._staleness_threshold = self._staleness_threshold.staleness_threshold
        else:
            self._staleness_threshold = PAGE_STALENESS_THRESHOLD

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

    def _extract_button_name(self, action_name: str, action_params: dict) -> str | None:
        """Extract the button/target name from an action for navigation tracking."""
        if action_name == "poco_click":
            return action_params.get("poco_path")
        if action_name == "airtest_touch_pos":
            x = action_params.get("x", 0)
            y = action_params.get("y", 0)
            return f"touch({x:.2f},{y:.2f})"
        return None

    def _get_page_context(self, page_hash: str, perception_data: Any) -> str:
        """Get cached page annotation or create one via LLM."""
        if self._page_cache.has(page_hash):
            return self._page_cache.to_prompt_text(page_hash)
        try:
            annotation = self._annotator.annotate(perception_data)
            knowledge = PageKnowledge(
                page_name=annotation.page_name,
                page_description=annotation.page_description,
                page_category=annotation.page_category,
                key_buttons=annotation.key_buttons,
            )
            self._page_cache.put(page_hash, knowledge)
            logger.info(
                "页面标注完成：%s → 「%s」（%s）",
                page_hash[:8],
                knowledge.page_name,
                knowledge.page_category,
            )
            return self._page_cache.to_prompt_text(page_hash)
        except Exception as exc:
            logger.warning("页面标注失败：%s", exc)
            return "（页面识别不可用）"

    def _decompose_task(self, task: str, perception_text: str) -> str:
        """Decompose a high-level task into ordered sub-goals via LLM."""
        try:
            prompt = prompts.TASK_DECOMPOSITION_PROMPT.format(
                task=task,
                game_state=perception_text[:500],
            )
            response = self._gemini.single_prompt(prompt)
            if response.text:
                sub_goals = json.loads(
                    response.text.strip().strip("`").removeprefix("json").strip()
                )
                if isinstance(sub_goals, list) and sub_goals:
                    formatted = "\n".join(
                        f"{i+1}. {g}" for i, g in enumerate(sub_goals)
                    )
                    logger.info("任务已分解为 %d 个子目标", len(sub_goals))
                    return formatted
        except Exception as exc:
            logger.warning("任务分解失败：%s", exc)
        return "（未分解，直接执行整体任务）"

    def _format_cycling_buttons(self) -> str:
        """List buttons from recent steps that form the detected cycle."""
        recent = self._context.last_n(6)
        seen: list[str] = []
        for step in recent:
            btn = self._extract_button_name(step.action_name, step.action_params)
            if btn and btn not in seen:
                seen.append(btn)
        return "\n".join(f"- {b}" for b in seen) if seen else "（未知按钮）"

    def _annotate_ineffective_buttons(
        self, markdown: str, page_hash: str, *, hide: bool = False
    ) -> str:
        """Mark or hide known-ineffective buttons in the perception markdown.

        When *hide* is True (page stale), ineffective button lines are removed
        entirely so the LLM cannot select them.
        """
        ineffective = set(self._nav_memory.get_ineffective_buttons(page_hash))
        if not ineffective:
            return markdown
        lines = []
        for line in markdown.split("\n"):
            matched = False
            for btn in ineffective:
                if btn in line:
                    matched = True
                    break
            if matched:
                if not hide:
                    lines.append(f"{line}  [❌ 禁止点击 - 已验证无效]")
            else:
                lines.append(line)
        return "\n".join(lines)

    async def run(self, task: str, max_steps: int = 50) -> AgentResult:
        """Execute a task through iterative ReAct reasoning."""
        self._context.clear()
        self._gemini.start_chat(
            system_prompt=prompts.STATIC_SYSTEM_INSTRUCTION,
            tools=self._tools.get_gemini_tools(),
        )

        initial_perception = self._perception.capture_l1()
        sub_goals = self._decompose_task(
            task, initial_perception.poco_tree_markdown
        )

        prev_page_hash: str = ""
        action_count = 0

        while action_count < max_steps:
            perception_data = self._perception.capture_l1()
            curr_state = self._perception.to_state(perception_data)
            curr_page_hash = curr_state.page_hash[:16]
            logger.info(
                "步骤 %d：页面=%s，可交互节点=%d",
                action_count,
                curr_page_hash[:8],
                len(perception_data.interactive_nodes),
            )

            stale_count = self._context.page_stale_count()
            is_cycling = self._context.cycle_detected(
                min_cycle_length=2, min_repetitions=CYCLE_DETECTION_MIN_REPETITIONS,
            )
            use_l2 = (
                self._context.consecutive_failures() >= L2_ESCALATION_FAILURE_THRESHOLD
                or stale_count >= self._staleness_threshold
                or is_cycling
            )

            if is_cycling:
                logger.warning(
                    "导航死循环检测：%s，强制升级策略",
                    self._context.cycle_description(),
                )

            if stale_count >= self._staleness_threshold:
                logger.info(
                    "页面停滞检测：连续 %d 步停留在页面 %s，升级到 L2 感知",
                    stale_count,
                    curr_page_hash[:8],
                )

            hide_ineffective = stale_count >= self._staleness_threshold
            annotated_markdown = self._annotate_ineffective_buttons(
                perception_data.poco_tree_markdown, curr_page_hash,
                hide=hide_ineffective,
            )
            nav_knowledge = self._nav_memory.to_prompt_text(curr_page_hash)
            page_context = self._get_page_context(curr_page_hash, perception_data)

            prompt_text = prompts.SYSTEM_PROMPT.format(
                perception_text=annotated_markdown,
                context_window=self._context.to_prompt_text(),
                navigation_knowledge=nav_knowledge,
                page_context=page_context,
                task_description=task,
                sub_goals=sub_goals,
            )

            if is_cycling:
                cycling_buttons = self._format_cycling_buttons()
                prompt_text += (
                    "\n\n## ⚠️ 系统警告：检测到导航死循环\n"
                    f"{self._context.cycle_description()}\n"
                    "你正在两个页面之间无限反复跳转。以下按钮会导致死循环，**严禁继续点击**：\n"
                    f"{cycling_buttons}\n"
                    "必须立即尝试完全不同的操作策略：滑动、使用返回键、或点击从未尝试过的按钮。"
                )

            if use_l2:
                if self._context.consecutive_failures() >= L2_ESCALATION_FAILURE_THRESHOLD:
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
                if response.text is None:
                    logger.warning(
                        "步骤 %d：模型未返回文本和工具调用（疑似解析失败），跳过本轮",
                        action_count,
                    )
                    action_count += 1
                    await asyncio.sleep(1.0)
                    continue
                logger.info("代理已完成任务：%s", response.text)
                return AgentResult(
                    success=True,
                    steps=list(self._context._steps),
                    final_message=response.text,
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

            # Post-action loading wait: for click actions, if page hash unchanged,
            # give the game a moment to finish animating/loading.
            post_state = self._perception.to_state(self._perception.capture_l1())
            if (
                fc.name in ("poco_click", "airtest_touch_pos")
                and post_state.page_hash[:16] == curr_page_hash
            ):
                for _retry in range(POST_ACTION_LOADING_RETRIES):
                    logger.info(
                        "点击后页面未变化，等待 %.1f 秒后重新采样（第%d次）",
                        POST_ACTION_LOADING_WAIT_S,
                        _retry + 1,
                    )
                    await asyncio.sleep(POST_ACTION_LOADING_WAIT_S)
                    post_state = self._perception.to_state(self._perception.capture_l1())
                    if post_state.page_hash[:16] != curr_page_hash:
                        logger.info("UI 已完成变化，继续规划")
                        break

            button_name = self._extract_button_name(fc.name, fc.args)

            step = ReActStep(
                step_number=action_count,
                thought=response.text or "",
                action_name=fc.name,
                action_params=fc.args,
                observation=tool_result.message,
                timestamp=time.time(),
                page_hash=curr_page_hash,
            )
            self._context.add_step(step)

            # Record navigation: reuse post_state from loading check above
            if button_name:
                post_page_hash = post_state.page_hash[:16]
                self._nav_memory.record(curr_page_hash, button_name, post_page_hash)
                if curr_page_hash == post_page_hash:
                    logger.info(
                        "导航记忆：%s 在页面 %s 上无效（页面未变化）",
                        button_name,
                        curr_page_hash[:8],
                    )

            prev_page_hash = curr_page_hash
            action_count += 1

            await asyncio.sleep(0.3)

        return AgentResult(
            success=False,
            steps=list(self._context._steps),
            final_message=f"已达到最大步数（{max_steps}）",
            total_steps=action_count,
        )
