#!/usr/bin/env python3
"""
rag_query.py

Run one or more questions against a Qdrant collection and print ranked
results with scores, so retrieval quality can be judged by eye.
Standard library only.

Usage:
  python3 rag_query.py --collection psf_eim_kb "PSF EIM 的五大產品矩陣是什麼？"
  python3 rag_query.py --collection psf_eim_kb --queries-file questions.txt
  python3 rag_query.py --collection psf_eim_kb --top-k 5 --full "..."
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_common import (  # noqa: E402
    die,
    embed_one,
    ensure_collection,
    log,
    search,
)

SNIPPET_CHARS = 180


def run_query(collection, question, top_k, full):
    vector = embed_one(question)
    hits = search(collection, vector, top_k=top_k)

    print(f"\n{'=' * 70}")
    print(f"Q: {question}")
    print("=" * 70)
    if not hits:
        print("  (沒有任何結果 — collection 可能是空的)")
        return None

    for rank, hit in enumerate(hits, start=1):
        payload = hit.get("payload", {})
        score = hit.get("score", 0.0)
        heading = payload.get("heading_path", "(無章節)")
        body = payload.get("body") or payload.get("text", "")
        source = payload.get("source", "?")
        idx = payload.get("chunk_index", "?")
        print(f"\n[{rank}] score={score:.4f}  ← {heading}  ({source} #{idx})")
        if full:
            for line in body.splitlines():
                print(f"    {line}")
        else:
            flat = " ".join(body.split())
            print(f"    {flat[:SNIPPET_CHARS]}{'…' if len(flat) > SNIPPET_CHARS else ''}")
    return hits


def main():
    ap = argparse.ArgumentParser(description="Query a Qdrant collection and show ranked hits.")
    ap.add_argument("questions", nargs="*", help="一個或多個查詢問題")
    ap.add_argument("--collection", default="psf_eim_kb", help="Qdrant collection 名稱")
    ap.add_argument("--queries-file", default=None, help="每行一個問題的檔案")
    ap.add_argument("--top-k", type=int, default=5, help="每個問題回傳幾筆 (預設 5)")
    ap.add_argument("--full", action="store_true", help="顯示完整 chunk 內容而非摘要")
    args = ap.parse_args()

    questions = list(args.questions)
    if args.queries_file:
        if not os.path.isfile(args.queries_file):
            die(f"找不到 --queries-file: {args.queries_file}")
        with open(args.queries_file, "r", encoding="utf-8") as f:
            questions += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not questions:
        die("請至少提供一個問題(位置參數)或用 --queries-file 指定檔案。")

    ensure_collection(args.collection, create=False)

    summary = []
    for q in questions:
        hits = run_query(args.collection, q, args.top_k, args.full)
        if hits:
            top = hits[0]
            summary.append(
                (
                    q,
                    top.get("score", 0.0),
                    top.get("payload", {}).get("heading_path", "?"),
                    (hits[0].get("score", 0.0) - hits[1].get("score", 0.0))
                    if len(hits) > 1
                    else None,
                )
            )

    print(f"\n\n{'=' * 70}")
    print("彙總:每個問題的最佳命中")
    print("=" * 70)
    for q, score, heading, gap in summary:
        gap_txt = f",與第 2 名差距 {gap:+.4f}" if gap is not None else ""
        print(f"  {score:.4f}  {q}")
        print(f"          → {heading}{gap_txt}")
    print()
    log("分數是 cosine 相似度(愈接近 1.0 愈相關)。與第 2 名的差距愈大,"
        "代表排序愈有鑑別度;差距很小代表模型難以區分,值得檢視 chunk 切分。")


if __name__ == "__main__":
    main()
