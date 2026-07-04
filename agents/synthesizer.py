"""
synthesizer.py
強化模組 7/7：Synthesizer Agent

職責：
- 彙整 research / document / finance 各 agent 的輸出，
  產生給使用者的最終回覆（純文字，繁體中文）。
- 若有 errors（例如某個查詢被 circuit breaker 擋下），會在回覆中適度註明，
  但不會讓技術性錯誤訊息主導整體回覆語氣。
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from llm_retry import llm_ainvoke

from state import HermesState

logger = logging.getLogger("hermes.synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Synthesizer，負責把多個 agent 的輸出
彙整成一份給使用者的最終繁體中文回覆。

要求：
- 語氣專業、簡潔，符合商業顧問溝通風格。
- 如果有財務數字，務必精確引用，不要自行更動。
- 如果有部分資訊取得失敗（見 errors），用一句話委婉說明，不要過度強調技術細節。
- 直接給結論與重點，不要重複描述「我彙整了以下資訊」這類贅詞。
"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1500, temperature=0.3)


async def synthesizer_node(state: HermesState) -> dict:
    user_input = state.get("user_input", "")
    agent_outputs = state.get("agent_outputs", {})
    errors = state.get("errors", [])

    llm = _build_llm()
    payload = {
        "user_input": user_input,
        "agent_outputs": agent_outputs,
        "errors": errors,
    }

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        AIMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
    ]

    response = await llm_ainvoke(llm, messages)
    final_answer = getattr(response, "content", "").strip()

    logger.info("Synthesizer 完成最終彙整")

    return {
        "completed_agents": ["synthesizer"],
        "final_answer": final_answer,
        "next_agent": "end",
    }
