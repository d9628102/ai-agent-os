#!/usr/bin/env python3
"""
rag_common.py

Shared helpers for the RAG tooling (rag_ingest.py, rag_query.py).
Standard library only — no pip installs needed on the GX10.

Note: scripts/rag_smoke_test.py deliberately keeps its own copy of these
helpers. It is the validated Step 4 acceptance check, so it stays
self-contained rather than being refactored onto this module.
"""
import json
import os
import sys
import urllib.error
import urllib.request

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8001/v1/embeddings")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "1024"))

# 共用片段：每一個要求模型輸出結構化 JSON 的 system prompt 都應該接上這段。
# 由來：score_dimension() 的評分呼叫實測出一個 100% 可重現的失敗——引用內容
# 裡只要有全角括號＋百分號（例如「（淨利率0.24%）」），模型就會在收尾 JSON
# 陣列時漏掉結尾的 `]`，10次輸入完全相同的測試全部同樣的方式壞掉，不是隨機
# 機率問題，重試救不回來。財務文件裡這種寫法（年化衰退62%、毛利率85%）到處
# 都是，理論上任何一個要求結構化 JSON 輸出的呼叫都可能踩到，不是評分這裡
# 獨有的風險，所以放在這裡讓三個 system prompt 共用同一段提醒，不要各自用
# 不同措辭各修一次。
JSON_OUTPUT_REMINDER = (
    "輸出 JSON 前，逐項確認陣列（[]）與物件（{}）的括號是否配對完整——"
    "特別注意如果引用或說明文字裡包含全角括號（（）中文括號）、百分號（%）"
    "等符號，這些不是 JSON 的結構符號，不要因為看到它們就提前結束陣列或"
    "物件，結尾務必要有對應的 ] 跟 }。"
)


def die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def http_json(method: str, url: str, payload=None, timeout: int = 300):
    """Returns (status_code, parsed_body). Exits on connection failure."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        return e.code, {"_raw_error_body": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        die(f"無法連到 {url}: {e.reason}")


def embed_batch(texts, batch_size: int = 16):
    """Embed a list of texts, returning a list of vectors in the same order."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        status, resp = http_json(
            "POST", EMBED_URL, {"model": EMBED_MODEL, "input": batch}
        )
        if status != 200:
            die(f"呼叫 embedding 端點失敗 (HTTP {status}): {resp}")
        try:
            # The API may return items out of order; sort by index to be safe.
            items = sorted(resp["data"], key=lambda d: d.get("index", 0))
            batch_vectors = [item["embedding"] for item in items]
        except (KeyError, TypeError):
            die(f"embedding 回應格式不如預期,原始回應: {resp}")
        if len(batch_vectors) != len(batch):
            die(
                f"embedding 回傳 {len(batch_vectors)} 筆向量,但送出了 {len(batch)} 段文字。"
            )
        for vec in batch_vectors:
            if len(vec) != VECTOR_SIZE:
                die(
                    f"embedding 向量維度是 {len(vec)},預期 {VECTOR_SIZE}。"
                    f"請確認 EMBED_MODEL/VECTOR_SIZE 與實際部署的模型一致。"
                )
        vectors.extend(batch_vectors)
        log(f"已 embedding {min(start + len(batch), len(texts))}/{len(texts)} 段文字。")
    return vectors


def embed_one(text: str):
    return embed_batch([text])[0]


def ensure_collection(name: str, create: bool = True) -> None:
    """Idempotent: reuses a matching collection, refuses a mismatched one."""
    status, resp = http_json("GET", f"{QDRANT_URL}/collections/{name}")
    if status == 200:
        existing = (
            resp.get("result", {})
            .get("config", {})
            .get("params", {})
            .get("vectors", {})
            .get("size")
        )
        if existing != VECTOR_SIZE:
            die(
                f"Collection '{name}' 已存在,但向量維度是 {existing}(預期 {VECTOR_SIZE})。"
                f"腳本不會自動覆蓋既有資料,請手動確認後再處理。"
            )
        log(f"Collection '{name}' 已存在且維度相符 ({existing})。")
        return
    if status != 404:
        die(f"查詢 collection '{name}' 失敗 (HTTP {status}): {resp}")
    if not create:
        die(f"Collection '{name}' 不存在。請先用 rag_ingest.py 寫入資料。")

    log(f"建立 collection '{name}' (size={VECTOR_SIZE}, distance=Cosine)...")
    status, resp = http_json(
        "PUT",
        f"{QDRANT_URL}/collections/{name}",
        {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
    )
    if status not in (200, 201):
        die(f"建立 collection 失敗 (HTTP {status}): {resp}")
    log("Collection 建立成功。")


def search(collection: str, vector, top_k: int = 5, query_filter=None):
    payload = {"vector": vector, "limit": top_k, "with_payload": True}
    if query_filter:
        payload["filter"] = query_filter
    status, resp = http_json(
        "POST", f"{QDRANT_URL}/collections/{collection}/points/search", payload
    )
    if status != 200:
        die(f"Qdrant 語意搜尋失敗 (HTTP {status}): {resp}")
    return resp.get("result", [])


def source_filter(source: str) -> dict:
    """組出限定單一 source 的 Qdrant filter，跟 search()/scroll 的 filter 參數搭配用。"""
    return {"must": [{"key": "source", "match": {"value": source}}]}


def distinct_sources(sources) -> list:
    """去重、濾掉空值（None/空字串）、排序——純邏輯，不牽涉 Qdrant 呼叫。
    決定 verify_sources_exist() 實際要去查哪些 source，避免同一個 source
    在問題清單裡出現多次就重複查詢多次。"""
    return sorted(set(s for s in sources if s))


def missing_sources(source_exists: dict) -> list:
    """source_exists: {source: bool}（bool 是那個 source 是否至少有 1 個
    chunk）。回傳排序後、不存在的 source 清單——純邏輯，不牽涉 Qdrant 呼叫。
    verify_sources_exist() 查完每個 source 之後，用這個函式判斷哪些算
    「找不到」。"""
    return sorted(s for s, found in source_exists.items() if not found)


def verify_sources_exist(collection: str, sources) -> None:
    """跨文件比對的前置檢查：問題清單裡引用的每個 source，在真正開始跑批次
    （花錢跑 LLM）之前，先確認至少有 1 個 chunk 存在。source 打錯字時
    Qdrant filter 只會回傳空結果，不會報錯——如果不先檢查，會讓
    answer_one() 在批次跑到一半時才因為 hits 為空而 die()，白白浪費前面
    幾題的呼叫。任何一個 source 查無結果，就在跑第一題之前直接死掉，一次
    列出所有有問題的 source，不要一個一個抓。"""
    source_exists = {}
    for source in distinct_sources(sources):
        status, resp = http_json(
            "POST",
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            {"filter": source_filter(source), "limit": 1, "with_payload": False},
        )
        if status != 200:
            die(f"查詢 source '{source}' 是否存在時失敗 (HTTP {status}): {resp}")
        source_exists[source] = bool(resp.get("result", {}).get("points"))
    missing = missing_sources(source_exists)
    if missing:
        die(
            f"Collection '{collection}' 裡找不到以下 source 的任何 chunk，"
            f"請確認檔名是否打錯字或還沒 ingest：{missing}"
        )
