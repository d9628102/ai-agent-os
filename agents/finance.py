"""
finance.py
強化模組 6/7：Finance Agent

職責：
- 處理財務分析、估值、SBLC融資結構、股票進出場分析等數字導向任務。
- 將計算邏輯與 LLM 判讀分離：先用 Python 做確定性計算，再交給 LLM 產生解讀文字，
  避免 LLM 在數字上「幻覺」。
"""

import json
import logging
from typing import Any, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from llm_retry import llm_ainvoke

from state import HermesState

logger = logging.getLogger("hermes.finance")

FINANCE_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Finance Agent，負責財務分析與解讀。
你會收到已經計算好的數字結果（不要自己重新計算或猜測數字），
你的任務是針對這些數字提供精簡、專業的繁體中文解讀與風險提示。
請回傳 JSON：
{"interpretation": "解讀文字", "risk_notes": ["風險提示1", "風險提示2"]}
只回傳 JSON，不要其他文字。
"""

EXTRACTION_SYSTEM_PROMPT = """你是參數抽取器，負責從使用者輸入中找出股票/部位的進場價、壓力價、
支撐價、持股數量等數字。

規則：
- 只有當輸入裡明確提到「進場價/成本價/買進價」以及至少一個「壓力/阻力/目標價」或
  「支撐/停損價」時，才視為找到有效參數。
- 找不到足夠資訊時，回傳 {"found": false}。
- 找到時，回傳：
  {"found": true, "entry_price": 數字, "resistance": 數字或null, "support": 數字或null, "position_size": 數字或null}
- 只回傳 JSON，不要其他文字，不要 markdown code block。
"""


def _build_extraction_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=300, temperature=0)


async def extract_finance_params(user_input: str) -> Optional[dict[str, Any]]:
    """
    從使用者原始輸入中抽取進場價/壓力/支撐等數字，供 calc_position_levels 使用。
    回傳 None 表示沒有找到足夠資訊（例如使用者只是純問問題，沒有提供具體數字）。
    """
    if not user_input or not user_input.strip():
        return None

    llm = _build_extraction_llm()
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        AIMessage(content=user_input),
    ]

    try:
        response = await llm_ainvoke(llm, messages)
        parsed = json.loads(response.content.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(f"finance_params 抽取失敗，視為未找到: {exc}")
        return None

    if not parsed.get("found"):
        return None

    entry_price = parsed.get("entry_price")
    resistance = parsed.get("resistance")
    support = parsed.get("support")

    if entry_price is None or (resistance is None and support is None):
        logger.info("抽取結果缺少必要欄位（entry_price 或 resistance/support 都缺），放棄計算")
        return None

    params: dict[str, Any] = {"entry_price": entry_price}
    if resistance is None or support is None:
        logger.info("僅有 resistance 或 support 其中一項，無法計算完整分批出場區間，跳過確定性計算")
        return None

    params["resistance"] = resistance
    params["support"] = support
    if parsed.get("position_size") is not None:
        params["position_size"] = parsed["position_size"]

    return params


def calc_position_levels(
    entry_price: float,
    resistance: float,
    support: float,
    position_size: Optional[float] = None,
) -> dict[str, Any]:
    """
    確定性計算：分批出場價位、停損停利幅度。
    不交給 LLM 算，避免數字錯誤。
    """
    upside_pct = (resistance - entry_price) / entry_price * 100
    downside_pct = abs(entry_price - support) / entry_price * 100
    risk_reward_ratio = upside_pct / downside_pct if downside_pct > 0 else float("inf")

    tiers = {
        "tier_1_exit": round(entry_price + (resistance - entry_price) * 0.5, 2),
        "tier_2_exit": round(resistance, 2),
        "stop_loss": round(support, 2),
    }

    result = {
        "entry_price": entry_price,
        "resistance": resistance,
        "support": support,
        "upside_pct": round(upside_pct, 2),
        "downside_pct": round(downside_pct, 2),
        "risk_reward_ratio": round(risk_reward_ratio, 2),
        "exit_tiers": tiers,
    }
    if position_size is not None:
        result["position_value"] = round(entry_price * position_size, 2)
    return result


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=600, temperature=0)


async def finance_node(state: HermesState) -> dict:
    user_input = state.get("user_input", "")

    # 優先順序：
    # 1) 若上游（例如 Orchestrator 或外部 API 呼叫）已經明確提供 finance_params，直接使用。
    # 2) 否則嘗試從使用者原始輸入自動抽取進場價/壓力/支撐等數字。
    calc_params = state.get("agent_outputs", {}).get("finance_params")
    if not calc_params:
        calc_params = await extract_finance_params(user_input)

    calc_result: dict[str, Any] = {}
    if calc_params:
        try:
            calc_result = calc_position_levels(**calc_params)
        except TypeError as exc:
            logger.error(f"Finance 計算參數錯誤: {exc}")
            calc_result = {"error": f"參數錯誤: {exc}"}

    llm = _build_llm()
    messages = [
        SystemMessage(content=FINANCE_SYSTEM_PROMPT),
        AIMessage(
            content=json.dumps(
                {"user_input": user_input, "calculated_numbers": calc_result},
                ensure_ascii=False,
            )
        ),
    ]

    response = await llm_ainvoke(llm, messages)

    try:
        interpretation = json.loads(response.content.strip())
    except (json.JSONDecodeError, AttributeError):
        interpretation = {
            "interpretation": getattr(response, "content", ""),
            "risk_notes": [],
        }

    return {
        "completed_agents": ["finance"],
        "agent_outputs": {
            "finance": {
                "numbers": calc_result,
                "summary": interpretation.get("interpretation", ""),
                "risk_notes": interpretation.get("risk_notes", []),
            }
        },
    }
