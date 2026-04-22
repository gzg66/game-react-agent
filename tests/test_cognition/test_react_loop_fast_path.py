import asyncio
from types import SimpleNamespace

from game_agent.cognition.context import ContextManager
from game_agent.cognition.gemini_client import FunctionCall, GeminiResponse
from game_agent.cognition.page_cache import PageKnowledge, PageKnowledgeCache
from game_agent.cognition.react_loop import AgentResult, ReActLoop
from game_agent.device.base import PocoNode
from game_agent.perception.base import PerceptionProvider
from game_agent.perception.state import L1Perception, PerceptionState
from game_agent.tools.registry import ToolRegistry
from game_agent.tools.schemas import AirtestTouchPosInput, PocoClickInput, ToolResult


class _FakePerception(PerceptionProvider):
    def __init__(self, snapshots: list[L1Perception]) -> None:
        self._snapshots = iter(snapshots)

    def capture_l1(self) -> L1Perception:
        return next(self._snapshots)

    def capture_l2(self):
        raise AssertionError("本测试不应进入 L2 感知")

    def get_current_page_hash(self) -> str:
        raise NotImplementedError

    def compute_diff(self, prev, curr):
        raise NotImplementedError

    def to_state(self, perception: L1Perception) -> PerceptionState:
        return PerceptionState(page_hash=perception.page_hash)


class _FakeGemini:
    def __init__(self) -> None:
        self.start_chat_calls = 0
        self.single_prompt_calls = 0
        self.send_message_calls = 0
        self.send_multimodal_calls = 0

    def start_chat(self, system_prompt: str, tools: list[dict] | None = None) -> None:
        self.start_chat_calls += 1
        self.system_prompt = system_prompt
        self.tools = tools

    def single_prompt(self, prompt: str, log_context: str) -> GeminiResponse:
        self.single_prompt_calls += 1
        return GeminiResponse(text='["进入下一页"]')

    def send_message(self, prompt: str, log_context: str, log_output: bool = False) -> GeminiResponse:
        self.send_message_calls += 1
        if self.send_message_calls > 1:
            raise AssertionError("进入新页面后应优先命中缓存，不应再次调用 LLM")
        return GeminiResponse(
            thought="先执行一步常规点击",
            function_calls=[FunctionCall(name="poco_click", args={"node_name": "btnNext"})],
        )

    def send_multimodal(self, *args, **kwargs):
        self.send_multimodal_calls += 1
        raise AssertionError("本测试不应进入多模态推理")


class _FakeUITreeStore:
    def update(self, page_hash: str, nodes: list) -> None:
        self.last_page_hash = page_hash
        self.last_nodes = nodes


def _make_node(
    name: str,
    *,
    text: str | None = None,
    poco_path: str = "",
    pos: tuple[float, float] = (0.5, 0.5),
    node_type: str = "Button",
) -> PocoNode:
    return PocoNode(
        name=name,
        type=node_type,
        text=text,
        pos=pos,
        poco_path=poco_path or name,
    )


def _make_l1(
    page_hash: str,
    *,
    visible_nodes: list[PocoNode] | None = None,
    interactive_nodes: list[PocoNode] | None = None,
    guide_node: PocoNode | None = None,
) -> L1Perception:
    visible_nodes = visible_nodes or []
    return L1Perception(
        timestamp=0.0,
        poco_tree_markdown=f"- page={page_hash}",
        interactive_nodes=interactive_nodes or [],
        page_hash=page_hash,
        all_visible_nodes=visible_nodes,
        guide_node=guide_node,
    )


def test_run_retries_fast_path_after_page_change(app_config):
    tool_registry = ToolRegistry()
    tool_registry.register(
        "poco_click",
        lambda params: ToolResult(success=True, message=f"已点击节点：{params.node_name}"),
        PocoClickInput,
        "点击节点",
    )

    page_cache = PageKnowledgeCache()
    for page_hash, page_name in (("page-a", "起始页"), ("page-b", "目标中转页")):
        page_cache.put(
            page_hash,
            PageKnowledge(
                page_name=page_name,
                page_description=f"{page_name}描述",
                page_category="测试页面",
                key_buttons=[],
            ),
        )

    loop = ReActLoop(
        gemini=_FakeGemini(),
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a"),  # run() 初始页面
                _make_l1("page-a"),  # 初始 fast path 失败后重新感知
                _make_l1("page-a"),  # while 第 0 轮
                _make_l1("page-b"),  # 点击后的 post-action 页面
                _make_l1("page-b"),  # while 第 1 轮，先尝试 fast path
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        page_cache=page_cache,
        ui_tree_store=_FakeUITreeStore(),
    )

    attempted_hashes: list[str] = []

    async def _fake_try_fast_path(task: str, start_hash: str) -> AgentResult | None:
        attempted_hashes.append(start_hash)
        if start_hash == "page-b":
            return AgentResult(
                success=True,
                final_message="快速导航完成：已到达缓存目标",
                total_steps=2,
            )
        return None

    loop._try_fast_path = _fake_try_fast_path  # type: ignore[method-assign]

    result = asyncio.run(loop.run("进入缓存目标", max_steps=5))

    assert attempted_hashes == ["page-a", "page-b"]
    assert result.success is True
    assert result.final_message == "快速导航完成：已到达缓存目标"
    assert result.total_steps == 3
    assert [step.action_name for step in result.steps] == ["poco_click"]


def test_run_continues_after_subgoal_fast_path_hit(app_config):
    page_cache = PageKnowledgeCache()
    page_cache.put(
        "page-a",
        PageKnowledge(
            page_name="登录页",
            page_description="登录页描述",
            page_category="测试页面",
            key_buttons=[],
        ),
    )
    page_cache.put(
        "page-b",
        PageKnowledge(
            page_name="下一页",
            page_description="下一页描述",
            page_category="测试页面",
            key_buttons=[],
        ),
    )

    gemini = _FakeGemini()
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=ToolRegistry(),
        perception=_FakePerception(
            [
                _make_l1("page-a"),
                _make_l1("page-a"),
                _make_l1("page-b"),
                _make_l1("page-b"),
                _make_l1("page-c"),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        page_cache=page_cache,
        ui_tree_store=_FakeUITreeStore(),
    )
    loop._tools.register(
        "poco_click",
        lambda params: ToolResult(success=True, message=f"已点击节点：{params.node_name}"),
        PocoClickInput,
        "点击节点",
    )

    attempted_queries: list[tuple[str, str]] = []

    async def _fake_try_fast_path(task: str, start_hash: str) -> AgentResult | None:
        attempted_queries.append((task, start_hash))
        if task == "进入下一页" and start_hash == "page-a":
            return AgentResult(
                success=True,
                final_message="快速导航完成：已到达缓存目标",
                total_steps=1,
            )
        return None

    loop._try_fast_path = _fake_try_fast_path  # type: ignore[method-assign]

    result = asyncio.run(loop.run("升一级", max_steps=2))

    assert attempted_queries == [
        ("升一级", "page-a"),
        ("进入下一页", "page-a"),
        ("升一级", "page-b"),
    ]
    assert result.success is False
    assert result.final_message == "已达到最大步数（2）"
    assert result.total_steps == 2
    assert [step.action_name for step in result.steps] == ["poco_click"]
    assert gemini.start_chat_calls == 1
    assert gemini.single_prompt_calls == 1
    assert gemini.send_message_calls == 1
    assert gemini.send_multimodal_calls == 0


def test_run_uses_l1_gloader3d_shortcut_without_llm(app_config, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("game_agent.cognition.react_loop.asyncio.sleep", _fake_sleep)

    tool_registry = ToolRegistry()
    tool_registry.register(
        "airtest_touch_pos",
        lambda params: ToolResult(
            success=True,
            message=f"已点击坐标：({params.x:.2f}, {params.y:.2f})",
        ),
        AirtestTouchPosInput,
        "点击坐标",
    )

    gemini = _FakeGemini()
    guide_node = _make_node(
        "GLoader3D",
        poco_path="Scene > GRoot > Container > Guide > Container > SoftGuideView > Container > GLoader3D",
        pos=(0.31, 0.42),
    )
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a", visible_nodes=[guide_node], guide_node=guide_node),
                _make_l1("page-a", visible_nodes=[guide_node], guide_node=guide_node),
                _make_l1("page-a", visible_nodes=[guide_node], guide_node=guide_node),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        ui_tree_store=_FakeUITreeStore(),
    )

    result = asyncio.run(loop.run("跟随引导", max_steps=1))

    assert result.success is False
    assert result.total_steps == 1
    assert [step.action_name for step in result.steps] == ["gloader3d_click"]
    assert gemini.start_chat_calls == 0
    assert gemini.single_prompt_calls == 0
    assert gemini.send_message_calls == 0
    assert gemini.send_multimodal_calls == 0
    assert sleep_calls == [0.3]


def test_run_does_not_use_gloader3d_shortcut_for_click_effect_path(app_config, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("game_agent.cognition.react_loop.asyncio.sleep", _fake_sleep)

    tool_registry = ToolRegistry()
    tool_registry.register(
        "poco_click",
        lambda params: ToolResult(success=True, message=f"已点击节点：{params.node_name}"),
        PocoClickInput,
        "点击节点",
    )

    gemini = _FakeGemini()
    click_effect_loader = _make_node(
        "GLoader3D",
        poco_path="Scene > GRoot > Container > ClickEffect > Container > GLoader3D",
        pos=(0.49, 0.81),
    )
    regular_button = _make_node(
        "btnNext",
        text="下一步",
        poco_path="Root > btnNext",
        pos=(0.66, 0.84),
    )
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a"),
                _make_l1(
                    "page-a",
                    visible_nodes=[click_effect_loader, regular_button],
                    interactive_nodes=[regular_button],
                ),
                _make_l1("page-b"),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        ui_tree_store=_FakeUITreeStore(),
    )

    result = asyncio.run(loop.run("进入下一页", max_steps=1))

    assert result.success is False
    assert result.total_steps == 1
    assert [step.action_name for step in result.steps] == ["poco_click"]
    assert gemini.start_chat_calls == 1
    assert gemini.single_prompt_calls >= 1
    assert gemini.send_message_calls == 1
    assert gemini.send_multimodal_calls == 0
    assert sleep_calls == [0.3]


def test_run_does_not_use_gloader3d_shortcut_without_explicit_ui_node(app_config, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("game_agent.cognition.react_loop.asyncio.sleep", _fake_sleep)

    tool_registry = ToolRegistry()
    tool_registry.register(
        "poco_click",
        lambda params: ToolResult(success=True, message=f"已点击节点：{params.node_name}"),
        PocoClickInput,
        "点击节点",
    )

    gemini = _FakeGemini()
    inferred_guide_node = _make_node(
        "GuideHint",
        poco_path="Root > Guide > GuideHint",
        pos=(0.31, 0.42),
    )
    regular_button = _make_node(
        "btnNext",
        text="下一步",
        poco_path="Root > btnNext",
        pos=(0.66, 0.84),
    )
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a"),
                _make_l1(
                    "page-a",
                    visible_nodes=[regular_button],
                    interactive_nodes=[regular_button],
                    guide_node=inferred_guide_node,
                ),
                _make_l1("page-b"),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        ui_tree_store=_FakeUITreeStore(),
    )

    result = asyncio.run(loop.run("进入下一页", max_steps=1))

    assert result.success is False
    assert result.total_steps == 1
    assert [step.action_name for step in result.steps] == ["poco_click"]
    assert gemini.start_chat_calls == 1
    assert gemini.single_prompt_calls >= 1
    assert gemini.send_message_calls == 1
    assert gemini.send_multimodal_calls == 0
    assert sleep_calls == [0.3]


def test_run_closes_popup_with_l1_shortcut_without_llm(app_config, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("game_agent.cognition.react_loop.asyncio.sleep", _fake_sleep)

    tool_registry = ToolRegistry()
    tool_registry.register(
        "airtest_touch_pos",
        lambda params: ToolResult(
            success=True,
            message=f"已点击坐标：({params.x:.2f}, {params.y:.2f})",
        ),
        AirtestTouchPosInput,
        "点击坐标",
    )

    gemini = _FakeGemini()
    popup_node = _make_node(
        "txtCloseHint",
        text="点击任意区域关闭",
        poco_path="Root > Modal > txtCloseHint",
        pos=(0.50, 0.88),
    )
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a", visible_nodes=[popup_node]),
                _make_l1("page-a", visible_nodes=[popup_node]),
                _make_l1("page-a", visible_nodes=[popup_node]),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        ui_tree_store=_FakeUITreeStore(),
    )

    result = asyncio.run(loop.run("关闭弹窗", max_steps=1))

    assert result.success is False
    assert result.total_steps == 1
    assert [step.action_name for step in result.steps] == ["popup_close_click"]
    assert gemini.start_chat_calls == 0
    assert gemini.single_prompt_calls == 0
    assert gemini.send_message_calls == 0
    assert gemini.send_multimodal_calls == 0
    assert sleep_calls == [0.3]


def test_run_skips_dialog_with_l1_shortcut_without_llm(app_config, monkeypatch):
    sleep_calls: list[float] = []

    async def _fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("game_agent.cognition.react_loop.asyncio.sleep", _fake_sleep)

    tool_registry = ToolRegistry()
    tool_registry.register(
        "airtest_touch_pos",
        lambda params: ToolResult(
            success=True,
            message=f"已点击坐标：({params.x:.2f}, {params.y:.2f})",
        ),
        AirtestTouchPosInput,
        "点击坐标",
    )

    gemini = _FakeGemini()
    dialog_content = _make_node(
        "DialogContent",
        text="对话内容",
        poco_path="Root > Dialogue > DialogContent",
        pos=(0.50, 0.75),
        node_type="Label",
    )
    skip_node = _make_node(
        "txtSkip",
        text="跳过",
        poco_path="Root > Dialogue > txtSkip",
        pos=(0.92, 0.08),
        node_type="Label",
    )
    loop = ReActLoop(
        gemini=gemini,
        tool_registry=tool_registry,
        perception=_FakePerception(
            [
                _make_l1("page-a", visible_nodes=[dialog_content, skip_node]),
                _make_l1("page-a", visible_nodes=[dialog_content, skip_node]),
                _make_l1("page-a", visible_nodes=[dialog_content, skip_node]),
            ]
        ),
        context=ContextManager(window_size=5),
        ui_diff=SimpleNamespace(),
        config=app_config,
        ui_tree_store=_FakeUITreeStore(),
    )

    result = asyncio.run(loop.run("跳过剧情", max_steps=1))

    assert result.success is False
    assert result.total_steps == 1
    assert [step.action_name for step in result.steps] == ["dialog_skip_click"]
    assert gemini.start_chat_calls == 0
    assert gemini.single_prompt_calls == 0
    assert gemini.send_message_calls == 0
    assert gemini.send_multimodal_calls == 0
    assert sleep_calls == [2.0]
