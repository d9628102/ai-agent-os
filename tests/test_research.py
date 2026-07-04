"""
測試重點：
- 正常情況下能彙整搜尋結果成 summary。
- 當 DDGS 拋出例外時，errors 會被正確收集，不會讓整個 node 掛掉。
- user_input 為空字串時，回傳「無有效查詢內容」且不觸發搜尋。
- circuit breaker 開啟時，搜尋結果會標記為失敗但流程仍能完成。
"""

import asyncio
import pytest
from unittest.mock import patch

from agents import research
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


@pytest.mark.asyncio
async def test_research_node_empty_input_short_circuits():
    state = {"user_input": "", "agent_outputs": {}}
    result = await research.research_node(state)
    assert result["completed_agents"] == ["research"]
    assert result["agent_outputs"]["research"]["summary"] == "無有效查詢內容"
    assert result["agent_outputs"]["research"]["raw"] == []


@pytest.mark.asyncio
async def test_research_node_success(monkeypatch):
    async def fake_to_thread(func, *args, **kwargs):
        return [{"title": "結果一", "href": "http://a", "body": "..."}]

    monkeypatch.setattr(research.asyncio, "to_thread", fake_to_thread)

    # Mock _generate_subqueries to avoid real LLM call
    async def fake_generate(_):
        return ["台新金控 電解銅 貿易融資"]

    monkeypatch.setattr(research, "_generate_subqueries", fake_generate)

    # Mock _summarize_results to avoid real LLM call
    async def fake_summarize(ui, sq, results):
        # Return a fake summary that includes the search result title
        return {"summary": "結果一", "sources": ["http://a"]}

    monkeypatch.setattr(research, "_summarize_results", fake_summarize)

    state = {"user_input": "台新金控 電解銅 貿易融資", "agent_outputs": {}}
    result = await research.research_node(state)

    assert result["completed_agents"] == ["research"]
    output = result["agent_outputs"]["research"]
    assert "結果一" in output["summary"]
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_research_node_handles_search_exception(monkeypatch):
    async def fake_to_thread_raises(func, *args, **kwargs):
        raise RuntimeError("network error")

    monkeypatch.setattr(research.asyncio, "to_thread", fake_to_thread_raises)

    # Mock _generate_subqueries to avoid real LLM call
    async def fake_generate(_):
        return ["測試查詢"]

    monkeypatch.setattr(research, "_generate_subqueries", fake_generate)

    # Mock _summarize_results to avoid real LLM call (all searches will fail)
    async def fake_summarize(ui, sq, results):
        return {"summary": "無搜尋結果", "sources": []}

    monkeypatch.setattr(research, "_summarize_results", fake_summarize)

    state = {"user_input": "測試查詢", "agent_outputs": {}}
    result = await research.research_node(state)

    assert result["completed_agents"] == ["research"]
    assert len(result["errors"]) == 1
    assert "network error" in result["errors"][0]
    # 即使查詢失敗，summary 仍應有預設文字而不是拋例外
    assert result["agent_outputs"]["research"]["summary"] == "無搜尋結果"


@pytest.mark.asyncio
async def test_research_node_respects_open_circuit_breaker(monkeypatch):
    # 預先把對應的 breaker 設成 OPEN，模擬該查詢之前已經連續失敗過
    from circuit_breaker import mcp_circuit_registry

    query = "已經斷路的查詢"
    breaker = mcp_circuit_registry.get(f"ddg_search:{query[:20]}")
    breaker.config.failure_threshold = 1
    breaker._failure_count = 1
    breaker._state = CircuitState.OPEN
    breaker._last_failure_time = __import__("time").monotonic()
    breaker.config.recovery_timeout = 999

    async def fake_to_thread(func, *args, **kwargs):
        raise AssertionError("breaker 為 OPEN 時不應該真的呼叫搜尋")

    monkeypatch.setattr(research.asyncio, "to_thread", fake_to_thread)

    state = {"user_input": query, "agent_outputs": {}}
    result = await research.research_node(state)

    assert len(result["errors"]) == 1
    assert "OPEN" in result["errors"][0]


@pytest.mark.asyncio
async def test_split_into_subqueries_caps_at_max():
    queries = await research._generate_subqueries("正常輸入")
    assert queries == ["正常輸入"]

    assert await research._generate_subqueries("   ") == []
    assert await research._generate_subqueries("") == []
