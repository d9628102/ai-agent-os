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
