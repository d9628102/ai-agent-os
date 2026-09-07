#!/usr/bin/env python3
"""
rag_smoke_test.py

Companion script for scripts/gx10-rag-setup.sh (Step 4.3). Uses only the
Python 3 standard library (no pip installs required) to:

  1. Create a Qdrant collection (idempotent — checks existing config first).
  2. Embed and upsert a handful of test documents with metadata.
  3. Embed a query sentence and run a semantic search against Qdrant.
  4. Print results for a human to judge relevance.

Any failure prints a clear message to stderr and exits non-zero — nothing
is silently skipped.

Env vars (all optional, defaults shown):
  EMBED_URL=http://localhost:8001/v1/embeddings
  QDRANT_URL=http://localhost:6333
  COLLECTION_NAME=psf_test_kb
  VECTOR_SIZE=1024
  EMBED_MODEL=BAAI/bge-m3
  TEST_DOCS_FILE=<unset>   # JSON file: [{"text": "...", "metadata": {...}}, ...]
                           # If unset, built-in placeholder docs are used —
                           # replace with real PSF EIM excerpts for a real test.
  QUERY_TEXT=<unset>       # defaults to a placeholder query matching the
                           # built-in placeholder docs.
"""
import json
import os
import sys
import urllib.request
import urllib.error

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:8001/v1/embeddings")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "psf_test_kb")
VECTOR_SIZE = int(os.environ.get("VECTOR_SIZE", "1024"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
TEST_DOCS_FILE = os.environ.get("TEST_DOCS_FILE", "")
QUERY_TEXT = os.environ.get("QUERY_TEXT", "")

# NOTE: these are placeholder documents so the script is runnable out of the
# box. They are NOT real PSF EIM content — set TEST_DOCS_FILE to a JSON file
# of real excerpts for a meaningful test.
DEFAULT_DOCS = [
    {
        "text": "Qdrant 是一個開源的向量資料庫，專門用來儲存 embedding 向量並支援高效的相似度搜尋，"
                "常被用在 RAG（檢索增強生成）架構中作為知識庫的檢索層。",
        "metadata": {"department": "test", "topic": "vector_db"},
    },
    {
        "text": "BGE-M3 是 BAAI 發布的多語言 embedding 模型，支援超過 100 種語言，"
                "輸出 1024 維的稠密向量，同時支援稀疏檢索與 ColBERT 多向量檢索。",
        "metadata": {"department": "test", "topic": "embedding_model"},
    },
    {
        "text": "RAG（Retrieval-Augmented Generation）的核心概念是先用向量搜尋找出與問題相關的文件片段，"
                "再把這些片段連同問題一起交給大型語言模型生成回答，藉此降低幻覺並補充模型未見過的知識。",
        "metadata": {"department": "test", "topic": "rag_concept"},
    },
]

DEFAULT_QUERY = "什麼是向量資料庫？"


def die(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def http_json(method: str, url: str, payload=None, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, {"_raw_error_body": body}
    except urllib.error.URLError as e:
        die(f"無法連到 {url}: {e.reason}")


def embed(text: str):
    status, resp = http_json(
        "POST",
        EMBED_URL,
        {"model": EMBED_MODEL, "input": text},
    )
    if status != 200:
        die(f"呼叫 embedding 端點失敗 (HTTP {status}): {resp}")
    try:
        vector = resp["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError):
        die(f"embedding 回應格式不如預期,原始回應: {resp}")
    if len(vector) != VECTOR_SIZE:
        die(
            f"embedding 向量維度是 {len(vector)},預期 {VECTOR_SIZE}。"
            f"請確認 EMBED_MODEL/VECTOR_SIZE 設定與實際部署的模型一致。"
        )
    return vector


def ensure_collection():
    status, resp = http_json("GET", f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
    if status == 200:
        existing_size = (
            resp.get("result", {})
            .get("config", {})
            .get("params", {})
            .get("vectors", {})
            .get("size")
        )
        if existing_size != VECTOR_SIZE:
            die(
                f"Collection '{COLLECTION_NAME}' 已存在,但向量維度是 {existing_size}"
                f"(預期 {VECTOR_SIZE})。請手動確認/刪除後再重跑,腳本不會自動覆蓋既有資料。"
            )
        log(f"Collection '{COLLECTION_NAME}' 已存在且維度相符 ({existing_size}),略過建立。")
        return
    if status != 404:
        die(f"查詢 collection '{COLLECTION_NAME}' 失敗 (HTTP {status}): {resp}")

    log(f"建立 collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, distance=Cosine)...")
    status, resp = http_json(
        "PUT",
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}",
        {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
    )
    if status not in (200, 201):
        die(f"建立 collection 失敗 (HTTP {status}): {resp}")
    log("Collection 建立成功。")


def load_docs():
    if TEST_DOCS_FILE:
        if not os.path.isfile(TEST_DOCS_FILE):
            die(f"TEST_DOCS_FILE 指定的檔案不存在: {TEST_DOCS_FILE}")
        with open(TEST_DOCS_FILE, "r", encoding="utf-8") as f:
            docs = json.load(f)
        log(f"從 {TEST_DOCS_FILE} 載入 {len(docs)} 段測試文字。")
        return docs
    log("未指定 TEST_DOCS_FILE,使用內建的 3 段【範例】文字(非真實 PSF EIM 內容,"
        "如需真實測試請用 TEST_DOCS_FILE 指向 JSON 檔案覆蓋)。")
    return DEFAULT_DOCS


def upsert_docs(docs):
    points = []
    for i, doc in enumerate(docs):
        text = doc.get("text")
        if not text:
            die(f"第 {i} 筆測試文件缺少 'text' 欄位: {doc}")
        metadata = doc.get("metadata", {})
        vector = embed(text)
        points.append(
            {
                "id": i + 1,
                "vector": vector,
                "payload": {"text": text, **metadata},
            }
        )
        log(f"已 embedding 並準備寫入第 {i + 1}/{len(docs)} 筆文件。")

    status, resp = http_json(
        "PUT",
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points?wait=true",
        {"points": points},
    )
    if status not in (200, 201):
        die(f"寫入 Qdrant 失敗 (HTTP {status}): {resp}")
    log(f"成功寫入 {len(points)} 筆文件到 '{COLLECTION_NAME}'。")


def search(query_text: str, top_k: int = 3):
    log(f"查詢句子: {query_text!r}")
    vector = embed(query_text)
    status, resp = http_json(
        "POST",
        f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
        {"vector": vector, "limit": top_k, "with_payload": True},
    )
    if status != 200:
        die(f"Qdrant 語意搜尋失敗 (HTTP {status}): {resp}")
    results = resp.get("result", [])
    if not results:
        die("語意搜尋沒有回傳任何結果,collection 可能是空的。")

    print("\n=== 語意搜尋結果 (由高到低相關) ===")
    for rank, hit in enumerate(results, start=1):
        score = hit.get("score")
        payload = hit.get("payload", {})
        text = payload.get("text", "")
        meta = {k: v for k, v in payload.items() if k != "text"}
        print(f"[{rank}] score={score:.4f} metadata={meta}")
        print(f"    {text}")
    print("=====================================\n")
    return results


def main():
    ensure_collection()
    docs = load_docs()
    upsert_docs(docs)
    query_text = QUERY_TEXT or DEFAULT_QUERY
    results = search(query_text)
    top_score = results[0].get("score")
    log(f"完成。最高分結果 score={top_score:.4f}(請自行判讀是否語意相關;"
        f"分數愈接近 1.0 代表 cosine 相似度愈高)。")


if __name__ == "__main__":
    main()
