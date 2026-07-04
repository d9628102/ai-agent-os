#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  Hermes Agent — v2 底層強化（Agent 模組）                  ║
║  內容: agents/orchestrator.py + research.py + document.py   ║
║        + finance.py + synthesizer.py                        ║
║  餵入方式: 此檔案與 v1-hermes-base.py 放在同一目錄，        ║
║            agents/ 目錄下每個檔案獨立輸出                    ║
╚══════════════════════════════════════════════════════════════╝

注意：
  - 此檔案與 v1-hermes-base.py 必須合在一起餵入 AI。
  - 實際使用時，此檔案的路徑為 ./agents/*.py，v1 的 graph.py 會
    from agents.orchestrator import orchestrator_node 載入這些節點。
"""


# ═══════════════════════════════════════════════════════════════
# Part 1 / agents/orchestrator.py
# ═══════════════════════════════════════════════════════════════

"""
orchestrator.py
強化模組 4/7：Orchestrator Agent

職責：
  - LangGraph 條件路由中樞，決定下一站該去哪個 agent。
  - 內建 max_steps 防禦，避免無限迴圈。
  - 同一 agent 不會被分配兩次。
"""

import json
import logging
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# v1 提供：llm_ainvoke, HermesState
# from llm_retry import llm_ainvoke
# from state import HermesState

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
只回傳 JSON，不要 markdown code block，不要其他文字。"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=300, temperature=0)


async def orchestrator_node(state: dict) -> dict:
    """
    從 state 判斷下一步路由。

    回傳 dict 包含 next_agent（research|document|finance|synthesizer）
    以及 step_count（max_steps 防護用）。
    """
    user_input = state.get("user_input", "")
    completed = state.get("completed_agents", [])
    agent_outputs = state.get("agent_outputs", {})
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 8)

    # max_steps 防禦
    if step_count >= max_steps:
        logger.warning("orchestrator_max_steps_reached",
                       extra={"step_count": step_count, "max_steps": max_steps})
        return {"next_agent": "synthesizer", "step_count": step_count + 1}

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

    if next_agent in completed:
        logger.warning("orchestrator_agent_already_completed", extra={"agent": next_agent})
        next_agent = "synthesizer"

    return {"next_agent": next_agent, "step_count": step_count + 1}


def route_after_orchestrator(state: dict) -> Literal["research", "document", "finance", "synthesizer"]:
    """LangGraph conditional edge 用 — 從 state 讀出 orchestrator 決定的 next_agent。"""
    return state.get("next_agent", "synthesizer")


# ═══════════════════════════════════════════════════════════════
# Part 2 / agents/research.py
# ═══════════════════════════════════════════════════════════════

"""
research.py
強化模組 3/7：Research Agent

職責：
  - 將使用者問題拆成 1~5 條子查詢（sub-query）
  - 平行對 DuckDuckGo 下搜尋，增加召回率
  - 內建 TTL 快取（5 分鐘），相同查詢不重複呼叫引擎
  - 受 circuit breaker 保護，搜尋失敗不整 node 掛掉
"""

import asyncio
import json
import logging
import time
from cachetools import TTLCache
from duckduckgo_search import DDGS
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# from llm_retry import llm_ainvoke
# from circuit_breaker import mcp_circuit_registry
# from state import HermesState

logger = logging.getLogger("hermes.research")

SEARCH_CACHE_TTL = 300  # 5 分鐘
_search_cache: TTLCache = TTLCache(maxsize=128, ttl=SEARCH_CACHE_TTL)

MAX_SUBQUERIES = 5
DEFAULT_NUM_RESULTS = 5

SUBQUERY_SYSTEM_PROMPT = """你是一個搜尋意圖拆解助手。你的工作是將使用者的問題拆成 1~5 條獨立的搜尋查詢。
每一條查詢應該專注於問題的一個面向，以便最大化搜尋引擎的召回率。

回傳格式（JSON 陣列）：["查詢1", "查詢2", ...]
只回傳 JSON 陣列，不要其他文字或 markdown code block。
若使用者輸入為空白或不合理，回傳空陣列 []。"""

RESEARCH_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Research Agent。
你收到的資訊包含使用者原問題、子查詢清單、以及原始搜尋結果。
請總結這些資訊，產出一份客觀、基於事實的繁體中文摘要。

請回傳 JSON：
{"summary": "200-400 字摘要", "sources": ["來源標題1", "來源標題2"]}
只回傳 JSON，不要 markdown code block。"""


def _build_subquery_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=500, temperature=0)


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=800, temperature=0.3)


async def _generate_subqueries(user_input: str) -> list[str]:
    """將使用者輸入拆成多條平行查詢。失敗時 fallback 回原始輸入。"""
    stripped = user_input.strip()
    if not stripped:
        return []

    llm = _build_subquery_llm()
    messages = [
        SystemMessage(content=SUBQUERY_SYSTEM_PROMPT),
        AIMessage(content=stripped),
    ]

    try:
        response = await llm_ainvoke(llm, messages)
        subqueries = json.loads(response.content.strip())
        if not isinstance(subqueries, list):
            raise ValueError("LLM 回傳非陣列格式")
        return [str(q) for q in subqueries[:MAX_SUBQUERIES]]
    except Exception as exc:
        logger.warning(f"子查詢拆解失敗，fallback 回原始輸入: {exc}")
        return [stripped]


def _sync_ddg_text(query: str, num_results: int) -> list[dict]:
    """同步執行 DDGS.text()，TTL 快取保護。"""
    cache_key = (query, num_results)
    cached = _search_cache.get(cache_key)
    if cached is not None:
        logger.debug("search_cache_hit", extra={"query": query[:40]})
        return cached

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))

    _search_cache[cache_key] = results
    return results


async def _search_single_subquery(subquery: str, num_results: int) -> dict:
    """對單條子查詢執行搜尋，受 circuit breaker 保護。"""
    breaker_name = f"ddg_search:{subquery[:20]}"
    breaker = mcp_circuit_registry.get(breaker_name)

    if breaker.state.name == "OPEN":
        logger.warning("search_circuit_open", extra={"breaker": breaker_name})
        return {"query": subquery, "error": "CircuitBreaker OPEN", "results": []}

    try:
        results = await asyncio.to_thread(_sync_ddg_text, subquery, num_results)
        await breaker.call(lambda: None)
        return {"query": subquery, "results": results}
    except Exception as exc:
        logger.error("search_failed", extra={"query": subquery[:40], "error": str(exc)})
        try:
            await breaker.call(lambda: (_ for _ in ()).throw(Exception(str(exc))))
        except Exception:
            pass
        return {"query": subquery, "error": str(exc), "results": []}


async def _summarize_results(user_input: str, subqueries: list[str], results: list[dict]) -> dict:
    """搜尋結果餵給 LLM 產生最終摘要。"""
    all_raw = []
    for r in results:
        for item in r.get("results", []):
            all_raw.append({"title": item.get("title", ""), "body": item.get("body", "")})

    if not all_raw:
        return {"summary": "無搜尋結果", "sources": []}

    llm = _build_llm()
    payload = {"user_input": user_input, "subqueries": subqueries, "raw_results": all_raw}

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        AIMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        response = await llm_ainvoke(llm, messages)
        parsed = json.loads(response.content.strip())
        return {"summary": parsed.get("summary", "無摘要"), "sources": parsed.get("sources", [])}
    except Exception as exc:
        logger.error(f"research 摘要 LLM 解析失敗: {exc}")
        return {"summary": "摘要無法產生", "sources": []}


async def research_node(state: dict) -> dict:
    """Research Agent 主節點。"""
    user_input = state.get("user_input", "")
    errors = list(state.get("errors", []))

    if not user_input.strip():
        logger.info("research_empty_input")
        return {
            "completed_agents": ["research"],
            "agent_outputs": {
                "research": {"summary": "無有效查詢內容", "raw": [], "subqueries_used": []}
            },
        }

    subqueries = await _generate_subqueries(user_input)
    search_tasks = [_search_single_subquery(q, DEFAULT_NUM_RESULTS) for q in subqueries]
    search_results = await asyncio.gather(*search_tasks)

    for sr in search_results:
        if "error" in sr:
            errors.append(f"查詢「{sr['query']}」失敗: {sr['error']}")

    summary_result = await _summarize_results(user_input, subqueries, search_results)

    return {
        "completed_agents": ["research"],
        "agent_outputs": {
            "research": {
                "summary": summary_result.get("summary", "無搜尋結果"),
                "sources": summary_result.get("sources", []),
                "subqueries_used": subqueries,
                "raw": search_results,
            }
        },
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════
# Part 3 / agents/document.py
# ═══════════════════════════════════════════════════════════════

"""
document.py
強化模組 5/7：Document Agent

職責：
  - 接收 research/finance 的輸出或原始筆記/逐字稿
  - 產生結構化文件內容（JSON 格式，方便轉 DOCX）
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# from llm_retry import llm_ainvoke
# from state import HermesState

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
只回傳 JSON，不要 markdown code block，不要其他說明文字。"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2000, temperature=0.3)


async def document_node(state: dict) -> dict:
    """Document Agent 主節點。"""
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


# ═══════════════════════════════════════════════════════════════
# Part 4 / agents/finance.py
# ═══════════════════════════════════════════════════════════════

"""
finance.py
強化模組 6/7：Finance Agent

職責：
  - 財務分析、估值、SBLC 融資結構、股票進出場分析
  - 計算邏輯與 LLM 判讀分離：先確定性計算，再交 LLM 解讀
"""

import json
import logging
from typing import Any, Optional
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# from llm_retry import llm_ainvoke
# from state import HermesState

logger = logging.getLogger("hermes.finance")

FINANCE_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Finance Agent，負責財務分析與解讀。
你會收到已經計算好的數字結果（不要自己重新計算或猜測數字），
你的任務是針對這些數字提供精簡、專業的繁體中文解讀與風險提示。
請回傳 JSON：
{"interpretation": "解讀文字", "risk_notes": ["風險提示1", "風險提示2"]}
只回傳 JSON，不要其他文字。"""

EXTRACTION_SYSTEM_PROMPT = """你是參數抽取器，負責從使用者輸入中找出股票/部位的進場價、壓力價、
支撐價、持股數量等數字。

規則：
- 只有當輸入裡明確提到「進場價/成本價/買進價」以及至少一個「壓力/阻力/目標價」或
  「支撐/停損價」時，才視為找到有效參數。
- 找不到足夠資訊時，回傳 {"found": false}。
- 找到時，回傳：
  {"found": true, "entry_price": 數字, "resistance": 數字或null, "support": 數字或null, "position_size": 數字或null}
- 只回傳 JSON，不要其他文字，不要 markdown code block。"""


def _build_extraction_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=300, temperature=0)


async def extract_finance_params(user_input: str) -> Optional[dict[str, Any]]:
    """從使用者原始輸入抽取進場價/壓力/支撐等數字。"""
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
        logger.info("抽取結果缺少必要欄位，放棄計算")
        return None

    params: dict[str, Any] = {"entry_price": entry_price}
    if resistance is None or support is None:
        logger.info("僅有 resistance 或 support 其中一項，跳過確定性計算")
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
    不交給 LLM，避免數字錯誤。
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


async def finance_node(state: dict) -> dict:
    """Finance Agent 主節點。"""
    user_input = state.get("user_input", "")

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
        AIMessage(content=json.dumps(
            {"user_input": user_input, "calculated_numbers": calc_result},
            ensure_ascii=False,
        )),
    ]

    response = await llm_ainvoke(llm, messages)

    try:
        interpretation = json.loads(response.content.strip())
    except (json.JSONDecodeError, AttributeError):
        interpretation = {"interpretation": getattr(response, "content", ""), "risk_notes": []}

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


# ═══════════════════════════════════════════════════════════════
# Part 5 / agents/synthesizer.py
# ═══════════════════════════════════════════════════════════════

"""
synthesizer.py
強化模組 7/7：Synthesizer Agent

職責：
  - 彙整 research / document / finance 各 agent 的輸出
  - 產生給使用者的最終回覆（純文字，繁體中文，商業顧問風格）
  - 若有技術性錯誤（circuit breaker 等），委婉說明但不主導語氣
"""

import json
import logging
from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# from llm_retry import llm_ainvoke
# from state import HermesState

logger = logging.getLogger("hermes.synthesizer")

SYNTHESIZER_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Synthesizer，負責把多個 agent 的輸出
彙整成一份給使用者的最終繁體中文回覆。

要求：
- 語氣專業、簡潔，符合商業顧問溝通風格。
- 如果有財務數字，務必精確引用，不要自行更動。
- 如果有部分資訊取得失敗（見 errors），用一句話委婉說明，不要過度強調技術細節。
- 直接給結論與重點，不要重複描述「我彙整了以下資訊」這類贅詞。"""


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1500, temperature=0.3)


async def synthesizer_node(state: dict) -> dict:
    """Synthesizer Agent 主節點。"""
    user_input = state.get("user_input", "")
    agent_outputs = state.get("agent_outputs", {})
    errors = state.get("errors", [])

    llm = _build_llm()
    payload = {"user_input": user_input, "agent_outputs": agent_outputs, "errors": errors}

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
