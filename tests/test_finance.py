"""
測試重點：
- calc_position_levels 的數字計算正確（這是「確定性計算」，必須精確）。
- 當沒有 position_size 時不應出現 position_value 欄位。
- finance_node 在有 finance_params 時會呼叫計算函式並把結果放進 state。
- finance_node 在計算參數錯誤時，回傳 error 而不是讓整個流程掛掉。
- LLM 回傳非 JSON 時，安全 fallback 成純文字 interpretation。
"""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage

from agents import finance


def _mock_llm(content: str):
    fake_llm = AsyncMock()
    fake_llm.ainvoke.return_value = AIMessage(content=content)
    return fake_llm


def test_calc_position_levels_basic_numbers():
    # 對應力成(6239)案例：233.5 進場、280 壓力、243 支撐
    result = finance.calc_position_levels(
        entry_price=233.5, resistance=280, support=243
    )
    assert result["upside_pct"] == pytest.approx(19.91, rel=1e-3)
    assert result["downside_pct"] == pytest.approx(4.07, rel=1e-3)
    assert result["risk_reward_ratio"] == pytest.approx(4.89, rel=1e-2)
    assert result["exit_tiers"]["stop_loss"] == 243
    assert result["exit_tiers"]["tier_2_exit"] == 280
    assert "position_value" not in result


def test_calc_position_levels_with_position_size():
    result = finance.calc_position_levels(
        entry_price=100, resistance=120, support=90, position_size=10
    )
    assert result["position_value"] == 1000


def test_calc_position_levels_zero_downside_gives_infinite_ratio():
    result = finance.calc_position_levels(entry_price=100, resistance=120, support=100)
    assert result["risk_reward_ratio"] == float("inf")


@pytest.mark.asyncio
async def test_finance_node_with_valid_params():
    state = {
        "user_input": "幫我分析力成的進出場位置",
        "agent_outputs": {
            "finance_params": {
                "entry_price": 233.5,
                "resistance": 280,
                "support": 243,
            }
        },
    }
    fake_llm = _mock_llm(
        '{"interpretation": "風險報酬比約4.9，屬於不錯的進場點。", '
        '"risk_notes": ["留意大盤系統性風險"]}'
    )

    with patch.object(finance, "_build_llm", return_value=fake_llm):
        result = await finance.finance_node(state)

    output = result["agent_outputs"]["finance"]
    assert output["numbers"]["exit_tiers"]["stop_loss"] == 243
    assert "風險報酬比" in output["summary"]
    assert output["risk_notes"] == ["留意大盤系統性風險"]


@pytest.mark.asyncio
async def test_finance_node_with_invalid_params_returns_error_not_crash():
    state = {
        "user_input": "分析一下",
        "agent_outputs": {
            "finance_params": {"entry_price": 100}  # 缺少 resistance/support
        },
    }
    fake_llm = _mock_llm('{"interpretation": "略", "risk_notes": []}')

    with patch.object(finance, "_build_llm", return_value=fake_llm):
        result = await finance.finance_node(state)

    output = result["agent_outputs"]["finance"]
    assert "error" in output["numbers"]


@pytest.mark.asyncio
async def test_finance_node_without_params_skips_calculation():
    state = {"user_input": "純粹問問題", "agent_outputs": {}}
    fake_llm = _mock_llm('{"interpretation": "沒有具體數字可分析", "risk_notes": []}')

    with patch.object(finance, "_build_llm", return_value=fake_llm), \
         patch.object(finance, "extract_finance_params", return_value=None):
        result = await finance.finance_node(state)

    output = result["agent_outputs"]["finance"]
    assert output["numbers"] == {}


@pytest.mark.asyncio
async def test_finance_node_llm_fallback_on_invalid_json():
    state = {"user_input": "問題", "agent_outputs": {}}
    fake_llm = _mock_llm("純文字回覆，不是 JSON")

    with patch.object(finance, "_build_llm", return_value=fake_llm), \
         patch.object(finance, "extract_finance_params", return_value=None):
        result = await finance.finance_node(state)

    output = result["agent_outputs"]["finance"]
    assert output["summary"] == "純文字回覆，不是 JSON"
    assert output["risk_notes"] == []
