"""ReAct reasoning loop — the core Thought -> Action -> Observation orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field

from game_agent.cognition import prompts
from game_agent.cognition.context import ContextManager, ReActStep
from game_agent.cognition.gemini_client import GeminiClient
from game_agent.cognition.navigation_memory import NavigationMemory
from game_agent.cognition.page_cache import PageKnowledge, PageKnowledgeCache
from game_agent.config import AppConfig
from game_agent.exceptions import PerceptionError
from game_agent.graph.annotator import PageAnnotator
from game_agent.perception.base import PerceptionProvider
from game_agent.perception.state import L1Perception, L2Perception
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

    @staticmethod
    def _format_action_for_log(response) -> str:
        if not response.function_calls:
            return "finish"
        action_lines = []
        for call in response.function_calls:
            action_lines.append(
                f"{call.name}({json.dumps(call.args, ensure_ascii=False, sort_keys=True)})"
            )
        return "\n".join(action_lines)

    def _log_react_output(
        self,
        step_number: int,
        response,
        observation: str,
    ) -> None:
        thought = response.thought or response.text
        logger.info(
            "步骤 %d：ReAct输出\n思考:\n%s\n动作:\n%s\n观察:\n%s",
            step_number,
            self._format_thought_for_log(thought),
            self._format_action_for_log(response),
            observation.strip() if observation.strip() else "（空）",
        )

    def _extract_button_name(self, action_name: str, action_params: dict) -> str | None:
        """Extract the button/target name from an action for navigation tracking."""
        if action_name == "poco_click":
            return action_params.get("poco_path")
        if action_name == "airtest_touch_pos":
            x = action_params.get("x", 0)
            y = action_params.get("y", 0)
            return f"touch({x:.2f},{y:.2f})"
        return None

    def _get_page_context(
        self,
        page_hash: str,
        perception_data: L1Perception | L2Perception,
    ) -> str:
        """Get cached page annotation or create one via LLM."""
        if self._page_cache.has(page_hash):
            return self._page_cache.to_prompt_text(page_hash)
        try:
            if isinstance(perception_data, L2Perception):
                annotation = self._annotator.annotate_with_screenshot(perception_data)
            else:
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
            response = self._gemini.single_prompt(
                prompt,
                log_context="task_decomposition",
            )
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

    # ---- Fast navigation (skip LLM when cached graph has a path) ----

    def _find_goal_pages(self, task: str, exclude_hash: str) -> list[str]:
        """Find candidate goal pages by matching task text against cached page names.

        Only returns the pages with the *best* overlap score to avoid BFS
        stopping at a weakly-matching intermediate page.
        """
        candidates: list[tuple[int, str]] = []
        for page_hash, knowledge in self._page_cache.items():
            if page_hash == exclude_hash:
                continue
            name = knowledge.page_name
            overlap = sum(1 for c in name if c in task and c not in "的了在是和与")
            if overlap >= 2:
                candidates.append((overlap, page_hash))
        if not candidates:
            return []
        candidates.sort(reverse=True)
        best_score = candidates[0][0]
        return [h for score, h in candidates if score == best_score]

    def _find_nav_path(
        self, start_hash: str, goal_hashes: set[str],
    ) -> list[tuple[str, str]] | None:
        """BFS shortest path from *start_hash* to any page in *goal_hashes*.

        Returns list of (button_name, target_page_hash), or None if unreachable.
        """
        if start_hash in goal_hashes:
            return []
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(start_hash, [])])
        visited = {start_hash}
        while queue:
            current, path = queue.popleft()
            for button, target in self._nav_memory.get_known_transitions(current).items():
                if button.startswith("touch("):
                    continue
                new_path = path + [(button, target)]
                if target in goal_hashes:
                    return new_path
                if target not in visited:
                    visited.add(target)
                    queue.append((target, new_path))
        return None

    async def _try_fast_path(self, task: str, start_hash: str) -> AgentResult | None:
        """Navigate to the goal using the cached nav graph — zero LLM calls."""
        goal_pages = self._find_goal_pages(task, start_hash)
        if not goal_pages:
            return None
        path = self._find_nav_path(start_hash, set(goal_pages))
        if not path:
            return None

        goal_knowledge = self._page_cache.get(path[-1][1])
        goal_name = goal_knowledge.page_name if goal_knowledge else path[-1][1][:8]
        logger.info(
            "快速导航：发现 %d 步缓存路径 → 「%s」", len(path), goal_name,
        )

        current_hash = start_hash
        for i, (button, expected_target) in enumerate(path):
            tool_result = self._tools.execute("poco_click", {"poco_path": button})
            if not tool_result.success:
                logger.warning("快速导航中断（步骤 %d）：点击 %s 失败", i, button)
                return None

            await asyncio.sleep(0.3)

            l1 = self._perception.capture_l1()
            actual_hash = self._perception.to_state(l1).page_hash[:16]

            if actual_hash != expected_target:
                for _ in range(POST_ACTION_LOADING_RETRIES):
                    await asyncio.sleep(POST_ACTION_LOADING_WAIT_S)
                    l1 = self._perception.capture_l1()
                    actual_hash = self._perception.to_state(l1).page_hash[:16]
                    if actual_hash == expected_target:
                        break

            self._nav_memory.record(current_hash, button, actual_hash)

            if actual_hash != expected_target:
                logger.warning(
                    "快速导航中断（步骤 %d）：页面不匹配（预期=%s 实际=%s）",
                    i, expected_target[:8], actual_hash[:8],
                )
                return None

            logger.info("快速导航步骤 %d：%s → %s ✓", i, button, actual_hash[:8])
            current_hash = actual_hash

        final_msg = f"快速导航完成：已到达「{goal_name}」"
        logger.info(final_msg)
        return AgentResult(
            success=True,
            steps=[],
            final_message=final_msg,
            total_steps=len(path),
        )

    # ---- Main ReAct loop ----

    async def run(self, task: str, max_steps: int = 50) -> AgentResult:
        """Execute a task through iterative ReAct reasoning."""
        self._context.clear()

        initial_perception = self._perception.capture_l1()
        start_hash = self._perception.to_state(initial_perception).page_hash[:16]

        fast_result = await self._try_fast_path(task, start_hash)
        if fast_result is not None:
            return fast_result

        # Fast path unavailable — fall back to full ReAct loop.
        # Re-capture in case fast path partially changed state.
        initial_perception = self._perception.capture_l1()

        self._gemini.start_chat(
            system_prompt=prompts.STATIC_SYSTEM_INSTRUCTION,
            tools=self._tools.get_gemini_tools(),
        )
        sub_goals = self._decompose_task(
            task, initial_perception.poco_tree_markdown
        )

        action_count = 0
        force_l2_reason: str | None = None

        while action_count < max_steps:
            l1_perception = self._perception.capture_l1()
            curr_state = self._perception.to_state(l1_perception)
            curr_page_hash = curr_state.page_hash[:16]
            logger.info(
                "步骤 %d：页面=%s，可交互节点=%d",
                action_count,
                curr_page_hash[:8],
                len(l1_perception.interactive_nodes),
            )

            stale_count = self._context.page_stale_count()
            is_cycling = self._context.cycle_detected(
                min_cycle_length=2, min_repetitions=CYCLE_DETECTION_MIN_REPETITIONS,
            )
            use_l2 = (
                self._context.consecutive_failures() >= L2_ESCALATION_FAILURE_THRESHOLD
                or stale_count >= self._staleness_threshold
                or is_cycling
                or force_l2_reason is not None
            )

            if use_l2:
                try:
                    perception_data = self._perception.capture_l2()
                    curr_state = self._perception.to_state(perception_data)
                    curr_page_hash = curr_state.page_hash[:16]
                except PerceptionError as exc:
                    logger.warning("步骤 %d：L2 感知失败，回退到 L1：%s", action_count, exc)
                    perception_data = l1_perception
                    use_l2 = False
            else:
                perception_data = l1_perception

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
            page_context = self._get_page_context(
                curr_page_hash,
                perception_data,
            )

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

            if force_l2_reason:
                logger.info("步骤 %d：强制使用 L2 感知，原因=%s", action_count, force_l2_reason)
                prompt_text += (
                    "\n\n## 视觉优先指令\n"
                    f"{force_l2_reason}\n"
                    "本轮必须优先根据截图判断是否存在新手引导手指、箭头、聚光高亮或被遮罩强调的目标区域；"
                    "若存在，应优先使用对应的点击操作跟随引导，而不是继续探索普通按钮。"
                )
                force_l2_reason = None

            if use_l2:
                if self._context.consecutive_failures() >= L2_ESCALATION_FAILURE_THRESHOLD:
                    logger.info("连续失败次数过多，升级到 L2 感知")
                response = self._gemini.send_multimodal(
                    prompt_text,
                    perception_data.screenshot_b64,
                    log_context="react",
                    log_output=False,
                )
            else:
                response = self._gemini.send_message(
                    prompt_text,
                    log_context="react",
                    log_output=False,
                )

            logger.info(
                "步骤 %d：模型思考摘要=%s",
                action_count,
                self._format_thought_for_log(response.thought or response.text),
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
                self._log_react_output(
                    action_count,
                    response,
                    f"任务完成，最终输出：{response.text}",
                )
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
            self._log_react_output(
                action_count,
                response,
                tool_result.message,
            )

            # Post-action loading wait: for click actions, if page hash unchanged,
            # give the game a moment to finish animating/loading.
            post_perception = self._perception.capture_l1()
            post_state = self._perception.to_state(post_perception)
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
                    post_perception = self._perception.capture_l1()
                    post_state = self._perception.to_state(post_perception)
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
                    if not self._page_cache.has(curr_page_hash):
                        force_l2_reason = (
                            "上一步点击后页面结构未发生变化，但游戏内可能出现了纯视觉引导。"
                            "请结合截图重点检查是否有手指动画、箭头、高亮描边、聚光遮罩或被教程强调的点击目标。"
                        )
                    else:
                        logger.info(
                            "页面 %s 已在缓存中，跳过视觉引导检测",
                            curr_page_hash[:8],
                        )
                elif fc.name in ("poco_click", "airtest_touch_pos"):
                    if not self._page_cache.has(post_page_hash):
                        force_l2_reason = (
                            "上一步点击后页面结构已变化。请结合截图确认新页面是否处于新手教程或强引导状态，"
                            "若有手指、箭头或高亮提示，优先跟随其指向的区域。"
                        )
                    else:
                        logger.info(
                            "目标页面 %s 已在缓存中，跳过强制 L2 感知",
                            post_page_hash[:8],
                        )

            action_count += 1

            await asyncio.sleep(0.3)

        return AgentResult(
            success=False,
            steps=list(self._context._steps),
            final_message=f"已达到最大步数（{max_steps}）",
            total_steps=action_count,
        )
