"""
測試重點：
- 正常情況下回傳 final_answer 並把 next_agent 設為 "end"。
- 會把 agent_outputs 與 errors 一起放進送給 LLM 的內容裡。
- completed_agents 正確標記 "synthesizer"。
"""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage

from agents import synthesizer


def _mock_llm(content: str):
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = AIMessage(content=content)
    return fake_llm


@pytest.mark.asyncio
async def test_synthesizer_returns_final_answer_and_ends():
    state = {
        "user_input": "幫我整理一下狀況",
        "agent_outputs": {"research": {"summary": "查到相關新聞"}},
        "errors": [],
    }
    fake_llm = _mock_llm("根據查到的資料，目前進度如下：...")

    with patch.object(synthesizer, "_build_llm", return_value=fake_llm):
        result = await synthesizer.synthesizer_node(state)

    assert result["final_answer"] == "根據查到的資料，目前進度如下：..."
    assert result["next_agent"] == "end"
    assert result["completed_agents"] == ["synthesizer"]


@pytest.mark.asyncio
async def test_synthesizer_includes_errors_in_payload():
    state = {
        "user_input": "問題",
        "agent_outputs": {},
        "errors": ["查詢被 circuit breaker 擋下"],
    }
    fake_llm = _mock_llm("部分資料暫時無法取得，但根據現有資訊...")

    with patch.object(synthesizer, "_build_llm", return_value=fake_llm):
        await synthesizer.synthesizer_node(state)

    call_args = fake_llm.ainvoke.call_args
    sent_messages = call_args.args[0]
    combined_text = " ".join(getattr(m, "content", "") for m in sent_messages)
    assert "circuit breaker" in combined_text
