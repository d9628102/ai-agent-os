"""
document.py
強化模組 5/7：Document Agent

職責：
- 接收 research/finance 的輸出或使用者提供的原始筆記/逐字稿，
  產生結構化文件內容（例如會議紀錄摘要）。
- 實際 DOCX 輸出建議交由 Node.js docx 套件處理（沿用既有 pipeline），
  這裡負責產生「結構化內容 JSON」，方便後續轉成 DOCX。
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from llm_retry import llm_ainvoke

from state import HermesState

logger = logging.getLogger("hermes.document")

DOCUMENT_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Document Agent，專門產生繁體中文商業文件結構。
輸出格式為 JSON，包含：
{
  "title": "文件標題",
  "summary": "400-500字摘要",
  "sections": [
    {"heading": "段落標題", "bullets": ["要點1", "要點2"]}
  ]
}
只回傳 JSON，不要 markdown code block，不要其他說明文字。
"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2000, temperature=0.3)


async def document_node(state: HermesState) -> dict:
    user_input = state.get("user_input", "")
    agent_outputs = state.get("agent_outputs", {})

    context_parts = [f"使用者需求：{user_input}"]
    if "research" in agent_outputs:
        context_parts.append(f"Research 結果摘要：{agent_outputs['research'].get('summary', '')}")
    if "finance" in agent_outputs:
        context_parts.append(f"Finance 結果摘要：{agent_outputs['finance'].get('summary', '')}")

    llm = _build_llm()
    messages = [
        SystemMessage(content=DOCUMENT_SYSTEM_PROMPT),
        AIMessage(content="\n".join(context_parts)),
    ]

    response = await llm_ainvoke(llm, messages)

    try:
        doc_structure = json.loads(response.content.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        logger.error(f"Document Agent JSON 解析失敗: {exc}")
        doc_structure = {
            "title": "文件產生失敗",
            "summary": "結構化解析失敗，請參考原始回應。",
            "sections": [],
            "raw_response": getattr(response, "content", ""),
        }

    return {
        "completed_agents": ["document"],
        "agent_outputs": {"document": doc_structure},
    }
