"""
測試重點：
- 正常情況下依照 LLM 回傳的 JSON 決定 next_agent。
- LLM 回傳格式錯誤（非 JSON）時，安全 fallback 到 synthesizer。
- 已執行過的 agent 不會被重複路由。
- 達到 max_steps 時強制路由到 synthesizer，避免無限迴圈。
- route_after_orchestrator 正確讀出 state 裡的 next_agent。
"""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage

from agents import orchestrator


def _mock_llm(content: str):
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = AIMessage(content=content)
    return fake_llm


@pytest.mark.asyncio
async def test_orchestrator_routes_based_on_llm_decision():
    state = {
        "user_input": "幫我查一下台新金控的電解銅貿易融資新聞",
        "completed_agents": [],
        "agent_outputs": {},
        "step_count": 0,
        "max_steps": 8,
    }
    fake_llm = _mock_llm('{"next_agent": "research", "reason": "需要查詢外部資料"}')

    with patch.object(orchestrator, "_build_llm", return_value=fake_llm):
        result = await orchestrator.orchestrator_node(state)

    assert result["next_agent"] == "research"
    assert result["step_count"] == 1


@pytest.mark.asyncio
async def test_orchestrator_fallback_on_invalid_json():
    state = {
        "user_input": "隨便問點什麼",
        "completed_agents": [],
        "agent_outputs": {},
        "step_count": 0,
        "max_steps": 8,
    }
    fake_llm = _mock_llm("這不是 JSON，只是隨便講講")

    with patch.object(orchestrator, "_build_llm", return_value=fake_llm):
        result = await orchestrator.orchestrator_node(state)

    assert result["next_agent"] == "synthesizer"


@pytest.mark.asyncio
async def test_orchestrator_avoids_repeating_completed_agent():
    state = {
        "user_input": "再查一次",
        "completed_agents": ["research"],
        "agent_outputs": {"research": {"summary": "已有結果"}},
        "step_count": 1,
        "max_steps": 8,
    }
    # LLM 又想選 research，但 research 已經跑過，應被導去 synthesizer
    fake_llm = _mock_llm('{"next_agent": "research", "reason": "重複建議"}')

    with patch.object(orchestrator, "_build_llm", return_value=fake_llm):
        result = await orchestrator.orchestrator_node(state)

    assert result["next_agent"] == "synthesizer"


@pytest.mark.asyncio
async def test_orchestrator_forces_synthesizer_at_max_steps():
    state = {
        "user_input": "任何問題",
        "completed_agents": ["research", "finance"],
        "agent_outputs": {},
        "step_count": 8,
        "max_steps": 8,
    }
    # 即使不 mock LLM，也不應該被呼叫到（max_steps 邏輯應在呼叫 LLM 前就短路）
    with patch.object(orchestrator, "_build_llm") as mock_build:
        result = await orchestrator.orchestrator_node(state)
        mock_build.assert_not_called()

    assert result["next_agent"] == "synthesizer"
    assert result["step_count"] == 9


def test_route_after_orchestrator_reads_next_agent():
    assert orchestrator.route_after_orchestrator({"next_agent": "finance"}) == "finance"
    assert orchestrator.route_after_orchestrator({}) == "synthesizer"
