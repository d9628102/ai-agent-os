#!/usr/bin/env python3
"""
rag_answer.py

End-to-end RAG question answering: retrieve from Qdrant, then generate a
grounded answer with the main vLLM server. Standard library only.

Flow per question:
  1. Embed the question with BGE-M3 (vllm-embed, port 8001)
  2. Search psf_eim_kb for the top-k most similar chunks (Qdrant, 6333)
  3. Assemble a grounded prompt: system instruction + retrieved chunks +
     the question
  4. Generate the answer with Qwen3-30B-A3B (vllm-server, port 8000)
  5. Strip Qwen3's <think> reasoning from what's shown, and report
     per-stage plus end-to-end timing

Thinking mode defaults to ON (DEFAULT_THINKING below). Repeated A/B
testing showed that without it, questions the document can't actually
support get a manufactured answer instead of "文件中未提及" — 3/3 of runs,
against 0/3 with thinking on. It costs roughly 4x in latency. See
docs/rag-findings.md Known Issue #7 for that experiment.

Pass --think or --no-think to decide for a whole run explicitly. Leave
both off and detect_think_reason() decides per question instead: a
comparison word (不同/差異/區別/...) or two or more proper nouns (naming
confusion risk) forces thinking on regardless of DEFAULT_THINKING;
anything else falls back to DEFAULT_THINKING. An explicit --think/
--no-think always wins over the detector.

When thinking is on, Qwen3 emits its chain of thought inside
<think>...</think>. That is stripped from the answer (see split_think for
the four shapes it can arrive in) — pass --show-think to display it.

Usage:
  python3 rag_answer.py "PSF EIM 的五大產品矩陣是什麼？"
  python3 rag_answer.py --queries-file scripts/data/psf-eim-qa-questions.txt
  python3 rag_answer.py --think --show-think "..."
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_common import (  # noqa: E402
    die,
    embed_one,
    ensure_collection,
    http_json,
    log,
    search,
)

CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8000/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-30B-A3B")

# Whether Qwen3 reasons before answering when neither --think nor
# --no-think is given. This policy has changed more than once, so it lives
# in one named place and BOTH flags always exist — changing the default
# must never break a command someone already uses.
# Currently True: grounding on document-unsupported questions outweighs the
# ~4x latency (docs/rag-findings.md Known Issue #7).
DEFAULT_THINKING = True

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "你是 PSF 內部知識助理。請「只」根據下方提供的內部文件片段回答問題，"
    "禁止編造文件中未提及的資訊，也不要用你自己的既有知識補充。\n"
    "如果提供的文件片段中找不到答案，請明確回答「文件中未提及」，並簡短說明"
    "你在文件裡找到的相關內容為何不足以回答，不要臆測。\n"
    "特別注意：當問題涉及與外部事物的比較（例如與其他系統、產品或做法的差異），"
    "只能陳述文件中對「我方」的描述，不可以用你的既有知識去描述、定義或推論"
    "「對方」的特性。文件沒有說明比較對象是什麼，就明說文件未描述該對象。\n"
    "回答請使用繁體中文（台灣慣用字），不要出現簡體字，並盡量引用文件片段中的原始用語。",
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Known Issue #7 的規則式判斷（見 docs/rag-findings.md）。
COMPARISON_KEYWORDS = ("不同", "差異", "區別", "比較", "差別", "對比", "相比", "相較")

# 抓連續、大寫字母開頭的英文詞當作一個「專有名詞」；緊鄰的詞
# (例如 "PSF EIM"、"Customer Success") 視為同一個實體一起吃掉，避免
# 只因為問題裡出現一次產品名縮寫就誤判有命名混淆風險。
PROPER_NOUN_RE = re.compile(
    r"[A-Z][A-Za-z0-9]*(?:™|®)?(?:\s+[A-Z][A-Za-z0-9]*(?:™|®)?)*"
)


def detect_think_reason(question: str):
    """
    依 Known Issue #7 的兩條規則判斷這一題是否該自動開啟思考模式：
      1. 問題含比較性詞彙（例如「不同」「差異」「區別」）
      2. 偵測到兩個以上專有名詞（可能有命名混淆風險）
      3. 其他一般查詢維持 --no-think

    回傳 (should_think, reason)；reason 為 None 代表不需要自動開啟，
    否則是給 log 用的人類可讀說明。
    """
    hit_keywords = [kw for kw in COMPARISON_KEYWORDS if kw in question]
    if hit_keywords:
        return True, f"問題含比較性詞彙「{'、'.join(hit_keywords)}」"

    entities = sorted(set(PROPER_NOUN_RE.findall(question)))
    if len(entities) >= 2:
        return True, f"偵測到 {len(entities)} 個專有名詞(命名混淆風險): {'、'.join(entities)}"

    return False, None


def split_think(text: str):
    """
    Separate Qwen3's reasoning from the answer.

    Handles all four shapes seen in practice:
      1. A complete <think>...</think> block
      2. A dangling </think> with no opener (some chat templates open the
         think block themselves, so the model never emits <think>)
      3. An unterminated <think> (generation hit max_tokens mid-reasoning)
      4. No tags at all

    Returns (visible_answer, thinking).
    """
    thinking_parts = [
        m.group(0)[len("<think>"):-len("</think>")].strip()
        for m in THINK_BLOCK_RE.finditer(text)
    ]
    visible = THINK_BLOCK_RE.sub("", text)

    if "</think>" in visible:
        head, _, tail = visible.partition("</think>")
        thinking_parts.append(head.replace("<think>", "").strip())
        visible = tail

    if "<think>" in visible:
        head, _, tail = visible.partition("<think>")
        thinking_parts.append(tail.strip())
        visible = head

    thinking = "\n".join(p for p in thinking_parts if p).strip()
    return visible.strip(), thinking


def build_context(hits):
    """Render retrieved chunks as numbered, heading-labelled blocks."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        payload = hit.get("payload", {})
        heading = payload.get("heading_path", "(無章節)")
        body = payload.get("body") or payload.get("text", "")
        blocks.append(f"[文件片段 {i}] {heading}\n{body}")
    return "\n\n".join(blocks)


def generate(question: str, context: str, max_tokens: int, temperature: float,
             enable_thinking: bool = DEFAULT_THINKING):
    user_content = (
        f"以下是從內部文件中檢索到的片段：\n\n{context}\n\n"
        f"---\n\n請根據上方文件片段回答這個問題：{question}"
    )
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not enable_thinking:
        # Qwen3's chat template takes this switch; vLLM forwards
        # chat_template_kwargs through to it. Measured on GB10, the hidden
        # reasoning was over half of each response's tokens, so turning it
        # off is the single biggest latency lever for grounded QA — where
        # the answer is extraction from supplied context, not deduction.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    status, resp = http_json("POST", CHAT_URL, payload)
    if status != 200:
        die(f"呼叫 vllm-server 失敗 (HTTP {status}): {resp}")
    try:
        message = resp["choices"][0]["message"]
        raw = message.get("content") or ""
        # Some builds surface reasoning in a separate field rather than inline.
        reasoning_field = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        die(f"vllm-server 回應格式不如預期，原始回應: {resp}")
    usage = resp.get("usage", {})
    return raw, reasoning_field, usage


def answer_one(question, collection, top_k, max_tokens, temperature,
               show_think, show_context, enable_thinking=DEFAULT_THINKING):
    print(f"\n{'=' * 72}")
    print(f"Q: {question}")
    print("=" * 72)

    t0 = time.time()
    vector = embed_one(question)
    t_embed = time.time() - t0

    t1 = time.time()
    hits = search(collection, vector, top_k=top_k)
    t_search = time.time() - t1
    if not hits:
        die(f"collection '{collection}' 沒有回傳任何檢索結果，可能是空的。")

    context = build_context(hits)

    t2 = time.time()
    raw, reasoning_field, usage = generate(
        question, context, max_tokens, temperature, enable_thinking
    )
    t_gen = time.time() - t2
    total = time.time() - t0

    visible, thinking = split_think(raw)
    thinking = thinking or reasoning_field.strip()

    print("\n【檢索到的片段】")
    for i, hit in enumerate(hits, start=1):
        payload = hit.get("payload", {})
        print(f"  [{i}] {hit.get('score', 0):.4f}  {payload.get('heading_path', '?')}")
    if show_context:
        print("\n【送進模型的完整片段內容】")
        print(context)

    if show_think and thinking:
        print("\n【模型推理過程 <think>】")
        print(thinking)

    print("\n【回答】")
    if visible:
        print(visible)
    else:
        print("(空白：模型可能在 max_tokens 內只輸出了推理內容還沒開始作答，"
              f"可調高 --max-tokens，目前 {max_tokens})")

    completion_tokens = usage.get("completion_tokens")
    tps = f"{completion_tokens / t_gen:.1f}" if completion_tokens and t_gen > 0 else "N/A"
    print(
        f"\n【耗時】embedding {t_embed:.2f}s | 檢索 {t_search:.2f}s | "
        f"生成 {t_gen:.2f}s | 端到端 {total:.2f}s"
        f"  (輸出 {completion_tokens or '?'} tokens, 約 {tps} tokens/s)"
    )

    return {
        "question": question,
        "visible": visible,
        "thinking_chars": len(thinking),
        "top_score": hits[0].get("score", 0.0),
        "top_heading": hits[0].get("payload", {}).get("heading_path", "?"),
        "total": total,
        "gen": t_gen,
        # Everything below was already computed above; exposing it lets a
        # caller (e.g. generate_report.py) reuse this one orchestration path
        # instead of re-implementing retrieve+generate itself. Existing
        # callers are unaffected -- these are additions, not replacements.
        "hits": hits,
        "context": context,
        "completion_tokens": usage.get("completion_tokens"),
        "think_used": enable_thinking,
        "thinking": thinking,
    }


def main():
    ap = argparse.ArgumentParser(description="End-to-end RAG QA over a Qdrant collection.")
    ap.add_argument("questions", nargs="*", help="一個或多個問題")
    ap.add_argument("--collection", default="psf_eim_kb", help="Qdrant collection 名稱")
    ap.add_argument("--queries-file", default=None, help="每行一個問題的檔案")
    ap.add_argument("--top-k", type=int, default=8, help="檢索幾個 chunk 餵給模型 (預設 8)")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="生成上限。Qwen3 的推理內容也會吃掉額度，別設太小 (預設 2048)")
    ap.add_argument("--temperature", type=float, default=0.2,
                    help="低溫度較適合有依據的問答 (預設 0.2)")
    ap.add_argument("--show-think", action="store_true", help="顯示 <think> 推理內容")
    ap.add_argument("--show-context", action="store_true", help="顯示送進模型的完整片段")
    think_group = ap.add_mutually_exclusive_group()
    think_group.add_argument(
        "--think", dest="think", action="store_true", default=None,
        help=f"明確開啟 Qwen3 的思考模式,對這次執行的所有問題生效,"
             f"detect_think_reason() 不會覆寫(目前預設"
             f"{'開啟' if DEFAULT_THINKING else '關閉'})")
    think_group.add_argument(
        "--no-think", dest="think", action="store_false", default=None,
        help="明確關閉思考模式，換取約 4 倍速度，對這次執行的所有問題生效，"
             "detect_think_reason() 不會覆寫。代價：文件無法支撐的問題"
             "(例如「跟 X 有什麼不同」但文件從未描述 X)會被硬湊出答案而非"
             "回答「文件中未提及」，實測 3/3 發生 —— 見 docs/rag-findings.md。"
             "不確定該不該關的話，不要加這個旗標，讓 detect_think_reason() "
             "在偵測到比較句或命名混淆風險時自動開啟")
    args = ap.parse_args()

    questions = list(args.questions)
    if args.queries_file:
        if not os.path.isfile(args.queries_file):
            die(f"找不到 --queries-file: {args.queries_file}")
        with open(args.queries_file, "r", encoding="utf-8") as f:
            questions += [ln.strip() for ln in f
                          if ln.strip() and not ln.startswith("#")]
    if not questions:
        die("請至少提供一個問題(位置參數)或用 --queries-file 指定檔案。")

    enable_thinking = DEFAULT_THINKING if args.think is None else args.think
    source = "預設" if args.think is None else "指定"

    ensure_collection(args.collection, create=False)
    detector_note = "，逐題可能依規則自動切換為開啟，見 Known Issue #7" if args.think is None else ""
    log(f"檢索 top_k={args.top_k}，生成模型 {CHAT_MODEL} @ {CHAT_URL}"
        f"，思考模式 {'開啟' if enable_thinking else '關閉'}({source}{detector_note})")

    results = []
    for q in questions:
        if args.think is None:
            auto_think, think_reason = detect_think_reason(q)
        else:
            auto_think, think_reason = False, None
        effective_thinking = enable_thinking or auto_think
        if auto_think and not enable_thinking:
            log(f"「{q}」{think_reason} → 本題自動切換為 --think")
        results.append(
            answer_one(q, args.collection, args.top_k, args.max_tokens,
                       args.temperature, args.show_think, args.show_context,
                       enable_thinking=effective_thinking)
        )

    print(f"\n\n{'=' * 72}")
    print("彙總")
    print("=" * 72)
    for r in results:
        first_line = r["visible"].splitlines()[0] if r["visible"] else "(無回答)"
        print(f"\n  Q: {r['question']}")
        print(f"     最佳檢索: {r['top_score']:.4f} ← {r['top_heading']}")
        print(f"     回答首句: {first_line[:60]}{'…' if len(first_line) > 60 else ''}")
        print(f"     端到端 {r['total']:.2f}s(其中生成 {r['gen']:.2f}s)"
              f"，推理內容 {r['thinking_chars']} 字(已過濾)")
    avg = sum(r["total"] for r in results) / len(results)
    print(f"\n  平均端到端耗時: {avg:.2f}s")


if __name__ == "__main__":
    main()
