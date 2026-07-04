"""
research.py
強化模組 3/7：Research Agent

職責：
- 用 DuckDuckGo 搜尋引擎平行查詢多個子問題，增加召回率。
- 先讓 LLM 將使用者的問題拆解成多條子查詢（sub-query），再平行對 DDGS 下搜尋。
- 搜尋結果經過 LLM 摘要後回傳。
- 內建 TTL 快取，同樣的查詢在快取有效期內不重複呼叫搜尋引擎。
- 受 circuit breaker 保護，搜尋失敗時不會整個 node 掛掉。
"""

import json
import asyncio
import logging
import time
from cachetools import TTLCache
from duckduckgo_search import DDGS

from langchain_core.messages import AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from llm_retry import llm_ainvoke
from circuit_breaker import mcp_circuit_registry

from state import HermesState

logger = logging.getLogger("hermes.research")

# ── 快取設定 ──────────────────────────────────────────────────────────────────
SEARCH_CACHE_TTL = 300  # 5 分鐘
_search_cache: TTLCache = TTLCache(maxsize=128, ttl=SEARCH_CACHE_TTL)

# ── 常數 ──────────────────────────────────────────────────────────────────────
MAX_SUBQUERIES = 5
DEFAULT_NUM_RESULTS = 5

SUBQUERY_SYSTEM_PROMPT = """你是一個搜尋意圖拆解助手。你的工作是將使用者的問題拆成 1~5 條獨立的搜尋查詢。
每一條查詢應該專注於問題的一個面向，以便最大化搜尋引擎的召回率。

回傳格式（JSON 陣列）：["查詢1", "查詢2", ...]
只回傳 JSON 陣列，不要其他文字或 markdown code block。
若使用者輸入為空白或不合理，回傳空陣列 []。
"""

RESEARCH_SYSTEM_PROMPT = """你是 Hermes Agent 系統的 Research Agent。
你收到的資訊包含使用者原問題、子查詢清單、以及原始搜尋結果。
請總結這些資訊，產出一份客觀、基於事實的繁體中文摘要。

請回傳 JSON：
{"summary": "200-400 字摘要", "sources": ["來源標題1", "來源標題2"]}
只回傳 JSON，不要 markdown code block。
"""


# ── LLM 建構 ──────────────────────────────────────────────────────────────────
def _build_subquery_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=500, temperature=0)


def _build_llm() -> ChatAnthropic:
    return ChatAnthropic(model="claude-sonnet-4-6", max_tokens=800, temperature=0.3)


# ── 子查詢拆解 ─────────────────────────────────────────────────────────────────
async def _generate_subqueries(user_input: str) -> list[str]:
    """將使用者輸入拆解成多條平行查詢。失敗時 fallback 回原始輸入。"""
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


# ── 同步搜尋（獨立執行緒） ─────────────────────────────────────────────────────
def _sync_ddg_text(query: str, num_results: int) -> list[dict]:
    """
    同步執行 DDGS.text()，並利用 TTL 快取避免重複呼叫。
    此函式應在 asyncio.to_thread 中執行，不會阻塞 event loop。
    """
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
    """
    對單條子查詢執行搜尋，受 circuit breaker 保護。
    搜尋失敗時不會拋例外，而是回傳錯誤標記。
    """
    breaker_name = f"ddg_search:{subquery[:20]}"
    breaker = mcp_circuit_registry.get(breaker_name)

    if breaker.state.name == "OPEN":
        logger.warning("search_circuit_open", extra={"breaker": breaker_name})
        return {"query": subquery, "error": f"CircuitBreaker OPEN", "results": []}

    try:
        results = await asyncio.to_thread(_sync_ddg_text, subquery, num_results)
        await breaker.call(lambda: asyncio.sleep(0))  # 成功，記錄一次成功
        return {"query": subquery, "results": results}
    except Exception as exc:
        logger.error("search_failed", extra={"query": subquery[:40], "error": str(exc)})
        # 讓 circuit breaker 記錄這次失敗
        try:
            await breaker.call(lambda: (_ for _ in ()).throw(Exception(str(exc))))
        except Exception:
            pass
        return {"query": subquery, "error": str(exc), "results": []}


# ── LLM 摘要 ──────────────────────────────────────────────────────────────────
async def _summarize_results(
    user_input: str, subqueries: list[str], results: list[dict]
) -> dict:
    """將搜尋結果餵給 LLM 產生最後摘要。"""
    all_raw = []
    for r in results:
        for item in r.get("results", []):
            all_raw.append({"title": item.get("title", ""), "body": item.get("body", "")})

    if not all_raw:
        return {"summary": "無搜尋結果", "sources": []}

    llm = _build_llm()
    payload = {
        "user_input": user_input,
        "subqueries": subqueries,
        "raw_results": all_raw,
    }

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        AIMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]

    try:
        response = await llm_ainvoke(llm, messages)
        parsed = json.loads(response.content.strip())
        return {
            "summary": parsed.get("summary", "無摘要"),
            "sources": parsed.get("sources", []),
        }
    except Exception as exc:
        logger.error(f"research 摘要 LLM 解析失敗: {exc}")
        return {"summary": "摘要無法產生", "sources": []}


# ── Research Node ─────────────────────────────────────────────────────────────
async def research_node(state: HermesState) -> dict:
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

    # Step 1: 拆分子查詢
    subqueries = await _generate_subqueries(user_input)

    # Step 2: 平行搜尋每條子查詢
    search_tasks = [
        _search_single_subquery(q, DEFAULT_NUM_RESULTS) for q in subqueries
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Step 3: 收集錯誤
    for sr in search_results:
        if "error" in sr:
            errors.append(f"查詢「{sr['query']}」失敗: {sr['error']}")

    # Step 4: LLM 摘要
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
