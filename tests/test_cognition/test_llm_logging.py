import logging
from types import SimpleNamespace

from game_agent.cognition.gemini_client import (
    FunctionCall,
    GeminiClient,
    GeminiResponse,
)
from game_agent.cognition.react_loop import ReActLoop
from game_agent.config import GeminiConfig


def _fake_gemini_response(*parts, function_calls=None):
    return SimpleNamespace(
        function_calls=function_calls or [],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=list(parts)),
            )
        ],
    )


def _text_part(text: str, *, thought: bool = False):
    return SimpleNamespace(text=text, thought=thought)


def _function_call_part(name: str, args: dict):
    return SimpleNamespace(
        function_call=SimpleNamespace(name=name, args=args),
    )


def test_send_message_logs_input_and_output(caplog):
    client = GeminiClient(GeminiConfig())
    client._system_prompt = "系统提示"
    client._tools = None
    client._generate_content = lambda *args, **kwargs: _fake_gemini_response(  # type: ignore[method-assign]
        _text_part("先分析局面", thought=True),
        _text_part("输出说明"),
        _function_call_part("poco_click", {"poco_path": "开始"}),
    )

    with caplog.at_level(logging.INFO):
        response = client.send_message("用户输入", log_context="task_decomposition")

    assert response.thought == "先分析局面"
    assert response.text == "输出说明"
    assert "LLM调用[task_decomposition] 输入" in caplog.text
    assert "system_prompt:\n系统提示" in caplog.text
    assert "user_prompt:\n用户输入" in caplog.text
    assert "LLM调用[task_decomposition] 输出" in caplog.text
    assert "thought:\n先分析局面" in caplog.text
    assert "tool_calls:\n- poco_click" in caplog.text


def test_react_output_log_uses_thought_action_observation_sections(caplog):
    loop = ReActLoop.__new__(ReActLoop)
    response = GeminiResponse(
        text="点击开始按钮",
        thought="当前位于主界面，应该尝试进入游戏。",
        function_calls=[FunctionCall(name="poco_click", args={"poco_path": "开始"})],
    )

    with caplog.at_level(logging.INFO):
        loop._log_react_output(3, response, "点击成功，页面跳转到关卡选择")

    assert "步骤 3：ReAct输出" in caplog.text
    assert "思考:\n当前位于主界面，应该尝试进入游戏。" in caplog.text
    assert '动作:\npoco_click({"poco_path": "开始"})' in caplog.text
    assert "观察:\n点击成功，页面跳转到关卡选择" in caplog.text
