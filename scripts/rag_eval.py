#!/usr/bin/env python3
"""
rag_eval.py

Retrieval-quality evaluation for a multi-document corpus. Standard library
only. Answers the question Task 4 exists for: does retrieval quality hold
up once the corpus stops being one tidy document?

Three things are measured per question:

  1. Top-1 correctness — is the best hit from the document/project the
     question is about?
  2. Top-3 purity — are ALL of the top 3 from the expected source? A
     question about A 專案 pulling B 專案's similarly-worded chunk into
     position 2 is the cross-document confusion this is built to catch.
  3. Pollution across top-k — how many hits come from an unrelated
     document, even when Top-1 is right. This is the leading indicator:
     it degrades before Top-1 does, so it warns while there is still time.

Results are written as JSON so a later run can be diffed against them
(--baseline). A corpus expansion that quietly pushes an old question's
correct answer down a rank shows up there rather than going unnoticed.

Test-set format (JSON):
  {
    "name": "psf-eim regression",
    "questions": [
      {"q": "...", "expect_source": "psf-eim.md", "kind": "regression"},
      {"q": "...", "expect_project": "A專案", "kind": "confusion"}
    ]
  }
Either expect_source or expect_project (or both) may be given; whichever
are present are checked.

Usage:
  python3 rag_eval.py --test-set scripts/data/eval/psf-eim-regression.json
  python3 rag_eval.py --test-set ... --save-baseline docs/eval-baseline.json
  python3 rag_eval.py --test-set ... --baseline docs/eval-baseline.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_common import (  # noqa: E402
    die,
    embed_one,
    ensure_collection,
    log,
    search,
)

# A top-1 score this much below baseline is called out as a regression.
SCORE_DROP_THRESHOLD = 0.05


def load_test_set(path):
    if not os.path.isfile(path):
        die(f"找不到測試集: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions")
    if not questions:
        die(f"測試集 {path} 沒有 questions 欄位或內容為空。")
    for i, item in enumerate(questions):
        if not item.get("q"):
            die(f"測試集第 {i} 題缺少 'q' 欄位: {item}")
        if not item.get("expect_source") and not item.get("expect_project"):
            die(f"測試集第 {i} 題必須至少指定 expect_source 或 expect_project，"
                f"否則無從判斷檢索是否命中正確來源: {item.get('q')}")
    return data


def hit_matches(payload, item):
    """Does this hit come from the source/project the question expects?"""
    if item.get("expect_source") and payload.get("source") != item["expect_source"]:
        return False
    if item.get("expect_project") and payload.get("project") != item["expect_project"]:
        return False
    return True


def evaluate_one(item, collection, top_k):
    vector = embed_one(item["q"])
    hits = search(collection, vector, top_k=top_k)
    if not hits:
        die(f"collection '{collection}' 對「{item['q']}」沒有回傳任何結果。")

    matched = [hit_matches(h.get("payload", {}), item) for h in hits]
    foreign = [
        {
            "rank": i + 1,
            "score": h.get("score", 0.0),
            "source": h.get("payload", {}).get("source", "?"),
            "project": h.get("payload", {}).get("project", ""),
            "heading": h.get("payload", {}).get("heading_path", "?"),
        }
        for i, h in enumerate(hits) if not matched[i]
    ]

    top = hits[0]
    return {
        "q": item["q"],
        "kind": item.get("kind", ""),
        "expect_source": item.get("expect_source", ""),
        "expect_project": item.get("expect_project", ""),
        "top1_ok": matched[0],
        "top3_pure": all(matched[:3]),
        "top1_score": top.get("score", 0.0),
        "top1_source": top.get("payload", {}).get("source", "?"),
        "top1_project": top.get("payload", {}).get("project", ""),
        "top1_heading": top.get("payload", {}).get("heading_path", "?"),
        "margin": (hits[0].get("score", 0.0) - hits[1].get("score", 0.0))
                  if len(hits) > 1 else None,
        "pollution_count": len(foreign),
        "pollution": foreign,
        "hits": [
            {
                "rank": i + 1,
                "score": h.get("score", 0.0),
                "source": h.get("payload", {}).get("source", "?"),
                "project": h.get("payload", {}).get("project", ""),
                "heading": h.get("payload", {}).get("heading_path", "?"),
                "matched": matched[i],
            }
            for i, h in enumerate(hits)
        ],
    }


def print_result(r, top_k, verbose):
    mark = "✅" if r["top1_ok"] else "❌"
    pure = "✅" if r["top3_pure"] else "⚠️"
    print(f"\n{mark} {r['q']}")
    expect = r["expect_project"] or r["expect_source"]
    print(f"   預期來源: {expect}")
    print(f"   Top1: {r['top1_score']:.4f} ← {r['top1_source']}"
          f"{' / ' + r['top1_project'] if r['top1_project'] else ''}"
          f"  ({r['top1_heading']})")
    margin = f"{r['margin']:+.4f}" if r["margin"] is not None else "N/A"
    print(f"   與第 2 名差距: {margin}   Top3 純度: {pure}"
          f"   跨文件污染: {r['pollution_count']}/{top_k}")
    if r["pollution"]:
        for p in r["pollution"]:
            print(f"      污染 #{p['rank']} {p['score']:.4f} ← {p['source']}"
                  f"{' / ' + p['project'] if p['project'] else ''}  ({p['heading']})")
    if verbose:
        for h in r["hits"]:
            flag = "·" if h["matched"] else "✗"
            print(f"      {flag} [{h['rank']}] {h['score']:.4f} {h['source']} — {h['heading']}")


def compare_baseline(results, baseline_path):
    if not os.path.isfile(baseline_path):
        die(f"找不到 baseline: {baseline_path}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    base_by_q = {r["q"]: r for r in baseline.get("results", [])}

    print(f"\n\n{'=' * 72}")
    print(f"與 baseline 比較（{baseline_path}）")
    print(f"  baseline 產生於 {baseline.get('generated_at', '?')}"
          f"，collection={baseline.get('collection', '?')}")
    print("=" * 72)

    regressions = []
    for r in results:
        b = base_by_q.get(r["q"])
        if not b:
            print(f"\n  [新增] {r['q']}（baseline 沒有這題，略過比較）")
            continue
        delta = r["top1_score"] - b["top1_score"]
        issues = []
        if b["top1_ok"] and not r["top1_ok"]:
            issues.append(f"Top1 從正確變成錯誤（現在命中 {r['top1_source']}）")
        if b["top1_source"] != r["top1_source"]:
            issues.append(f"Top1 來源改變: {b['top1_source']} → {r['top1_source']}")
        if delta < -SCORE_DROP_THRESHOLD:
            issues.append(f"分數下降 {delta:+.4f}（超過門檻 {SCORE_DROP_THRESHOLD}）")
        if b["top3_pure"] and not r["top3_pure"]:
            issues.append("Top3 純度從乾淨變成有混入其他文件")
        if r["pollution_count"] > b["pollution_count"]:
            issues.append(f"跨文件污染增加: {b['pollution_count']} → {r['pollution_count']}")

        if issues:
            regressions.append((r["q"], issues))
            print(f"\n  ⚠️  {r['q']}")
            for i in issues:
                print(f"        - {i}")
        else:
            print(f"\n  ✅ {r['q']}  ({delta:+.4f})")

    print()
    if regressions:
        print(f"  共 {len(regressions)} 題出現退化 —— 語料擴大確實影響了既有檢索品質。")
    else:
        print("  沒有任何一題退化：語料擴大後既有檢索品質維持。")
    return regressions


def main():
    ap = argparse.ArgumentParser(description="Retrieval quality evaluation across a multi-document corpus.")
    ap.add_argument("--test-set", required=True, help="測試集 JSON 檔")
    ap.add_argument("--collection", default="psf_eim_kb")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--baseline", default=None, help="與這份先前結果比較，找出退化")
    ap.add_argument("--save-baseline", default=None, help="把本次結果存成 baseline")
    ap.add_argument("--verbose", action="store_true", help="列出每題完整的 top-k")
    args = ap.parse_args()

    data = load_test_set(args.test_set)
    ensure_collection(args.collection, create=False)
    log(f"測試集「{data.get('name', args.test_set)}」共 {len(data['questions'])} 題，"
        f"collection={args.collection}, top_k={args.top_k}")

    results = []
    for item in data["questions"]:
        r = evaluate_one(item, args.collection, args.top_k)
        print_result(r, args.top_k, args.verbose)
        results.append(r)

    total = len(results)
    top1 = sum(1 for r in results if r["top1_ok"])
    pure = sum(1 for r in results if r["top3_pure"])
    polluted = sum(1 for r in results if r["pollution_count"] > 0)
    total_pollution = sum(r["pollution_count"] for r in results)

    print(f"\n\n{'=' * 72}")
    print("彙總")
    print("=" * 72)
    print(f"  Top1 命中正確來源:     {top1}/{total}")
    print(f"  Top3 完全無混入:       {pure}/{total}")
    print(f"  有跨文件污染的題數:     {polluted}/{total}"
          f"（總污染筆數 {total_pollution}/{total * args.top_k}）")

    by_kind = {}
    for r in results:
        k = r["kind"] or "(未分類)"
        by_kind.setdefault(k, []).append(r)
    if len(by_kind) > 1:
        print("\n  依題型：")
        for k, rs in sorted(by_kind.items()):
            ok = sum(1 for r in rs if r["top1_ok"])
            pk = sum(1 for r in rs if r["top3_pure"])
            print(f"    {k}: Top1 {ok}/{len(rs)}，Top3 純度 {pk}/{len(rs)}")

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_set": args.test_set,
        "collection": args.collection,
        "top_k": args.top_k,
        "results": results,
    }
    if args.save_baseline:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_baseline)), exist_ok=True)
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log(f"已存成 baseline: {args.save_baseline}")

    if args.baseline:
        regressions = compare_baseline(results, args.baseline)
        if regressions:
            sys.exit(1)   # 讓退化能被 CI 或腳本偵測到


if __name__ == "__main__":
    main()
