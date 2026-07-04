"""
orchestrator.py
強化模組 4/7：Orchestrator Agent

職責：
- LangGraph 條件路由中樞，負責判斷下一站該去哪個 agent。
- 依據 user_input、已完成的 agents、agent_outputs 決定 next_agent。
- 內建 max_steps 防禦機制，避免無限迴圈。
"""

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from llm_retry import llm_ainvoke

from state import HermesState

logger = logging.getLogger("hermes.orchestrator")

ORCHESTRATOR_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Orchestrator，負責決定下一步該執行哪個 agent。
可選的 agent：research（需要查資料）、document（產生結構化文件）、finance（財務分析）、synthesizer（彙整所有結果給使用者）

規則：
1. 如果使用者需求還需要查資料，優先選 research。
2. 如果 financial analysis 已經有結果需要彙整，選 finance。
3. 如果需要將既有資料整理成文件，選 document。
4. 如果所有需要的資料都已經收集完畢，選 synthesizer 結束流程。
5. 永遠不要選一個已經在 completed_agents 裡的 agent。
6. 回傳 JSON：{"next_agent": "research|document|finance|synthesizer", "reason": "簡短原因"}
只回傳 JSON，不要 markdown code block，不要其他文字。
"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=300, temperature=0)


async def orchestrator_node(state: HermesState) -> dict:
    user_input = state.get("user_input", "")
    completed = state.get("completed_agents", [])
    agent_outputs = state.get("agent_outputs", {})
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 8)

    # max_steps 防禦：達到上限則強制結束，不再呼叫 LLM
    if step_count >= max_steps:
        logger.warning(
            "orchestrator_max_steps_reached",
            extra={"step_count": step_count, "max_steps": max_steps},
        )
        return {
            "next_agent": "synthesizer",
            "step_count": step_count + 1,
            "_max_steps_forced": True,
        }

    llm = _build_llm()
    payload = {
        "user_input": user_input,
        "completed_agents": completed,
        "agent_outputs": {k: list(v.keys()) for k, v in agent_outputs.items()},
    }

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        AIMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        response = await llm_ainvoke(llm, messages)
        decision = json.loads(response.content.strip())
        next_agent = decision.get("next_agent", "synthesizer")
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error(f"Orchestrator JSON 解析失敗，fallback 到 synthesizer: {exc}")
        next_agent = "synthesizer"

    # 若 LLM 選擇的 agent 已執行過，強制導向 synthesizer
    if next_agent in completed:
        logger.warning("orchestrator_agent_already_completed", extra={"agent": next_agent})
        next_agent = "synthesizer"

    return {
        "next_agent": next_agent,
        "step_count": step_count + 1,
    }


def route_after_orchestrator(state: dict) -> Literal["research", "document", "finance", "synthesizer"]:
    """LangGraph conditional edge 用 — 從 state 讀出 orchestrator 決定的 next_agent。"""
    return state.get("next_agent", "synthesizer")
