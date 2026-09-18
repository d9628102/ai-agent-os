#!/usr/bin/env python3
"""
generate_report.py

簡報工廠 MVP:給一份問題清單,批次跑過 rag_answer.py 的檢索+生成邏輯,把每題
答案、來源引用、QA 分數組合成一份結構化 Markdown 報告。

用途是內部快速產出一份說明文件(例如「PSF EIM 系統說明文件」),不是拿去
給外部客戶或投資人看的正式文件——被 QA 標記為需人工複核的段落,答案內容
照樣完整顯示,只是加警示,不隱藏、不跳過不生成(跟 n8n 那條對外的問答流程
刻意隱藏未審核內容不一樣,這裡是內部草稿,人工複核時要看得到原文才能判斷)。

QA 把關邏輯(確定性檢查 + LLM judge 評分)是照 n8n workflow 裡「QA
Deterministic Checks」/「QA LLM Judge」/「QA Parse & Threshold」這幾個
節點已經驗證過、修過 bug 的版本 port 過來的獨立 Python 實作——這個獨立
是刻意的:這支腳本要能在 n8n/Docker 沒啟動時也能手動跑。代價是同一套
QA 邏輯現在存在兩個地方(n8n JS 版、這裡的 Python 版),之後任一邊調整
規則要記得兩邊都要改;等出現第三個消費者需要重用這套邏輯時,才值得
拆成共用模組,這次不做。

Usage:
  python3 scripts/generate_report.py \\
      --questions scripts/data/psf-eim-qa-questions.txt \\
      --output reports/psf-eim-report.md \\
      --title "PSF EIM 系統說明文件"
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_answer import (  # noqa: E402
    CHAT_MODEL,
    CHAT_URL,
    DEFAULT_THINKING,
    answer_one,
    detect_think_reason,
)
from rag_common import die, ensure_collection, http_json, log  # noqa: E402
from numeric_consistency import (  # noqa: E402
    check_group_consistency,
    extract_metric_value,
    locate_source_heading,
)
from red_flag_detection import (  # noqa: E402
    dedupe_red_flags,
    detect_red_flag,
)

# 跟 n8n「QA Deterministic Checks」/「QA LLM Judge」節點目前(修過 bug 後)
# 的版本對齊,不要照舊版(headings-only judge、逐字比對截斷)。
FAITHFULNESS_THRESHOLD = 7

# 來源引用只列真的有支撐力的片段——負向對照題常見的情況是 top-k 裡沒有一個
# 真正相關的片段,只是矬中選優,列出來反而讓人誤以為答案有依據。
MIN_CITATION_SCORE = 0.6

JUDGE_SYSTEM_PROMPT = (
    "你是嚴格的 RAG 問答品質評審員。你會看到：使用者問題、檢索到的文件片段全文、"
    "以及系統產生的回答。請針對兩個維度各給 1-10 的整數評分：\n"
    "1. faithfulness(忠實度)：逐項核對回答裡的每個具體主張(名稱、數字、職責描述、"
    "專有名詞等)是否能在提供的文件片段全文裡找到依據。只要是片段全文裡明確寫到的內容，"
    "即使回答是逐字引用或改寫，都算忠實，不算捏造；只有片段全文找不到根據、且回答仍"
    "明確斷言的內容才算捏造，該大幅扣分。若片段全文不足以回答問題，回答有沒有誠實說"
    "「文件中未提及」而不是硬掰。\n"
    "2. relevance(相關性)：回答是否切題回應了使用者的問題，而不是答非所問或離題。\n"
    "請只輸出一個 JSON 物件，格式為 "
    '{"faithfulness": <整數>, "relevance": <整數>, "reasoning": "<20字以內的簡短理由>"}'
    "，不要有任何其他文字、不要用 markdown code fence。"
)


def check_deterministic(answer: str, completion_tokens, max_tokens: int):
    """對應 n8n「QA Deterministic Checks」節點:截斷判斷用是否撞到
    max_tokens 上限,不是猜結尾標點;亂碼判斷同一套規則。"""
    hit_token_cap = isinstance(completion_tokens, int) and completion_tokens >= max_tokens - 8
    not_truncated = bool(answer.strip()) and not hit_token_cap
    has_garbled = "�" in answer or re.search(r"(.)\1{9,}", answer) is not None
    no_garbled = not has_garbled
    return {
        "not_truncated": not_truncated,
        "no_garbled": no_garbled,
        "deterministic_pass": not_truncated and no_garbled,
    }


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def call_judge(question: str, context: str, answer: str, timeout: int = 180):
    """對應 n8n「QA LLM Judge」節點:給完整檢索片段全文,不是只給標題。"""
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"問題：{question}\n\n檢索到的文件片段全文：\n{context}\n\n"
                    f"系統回答：\n{answer}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, resp = http_json("POST", CHAT_URL, payload, timeout=timeout)
    if status != 200:
        return None, None, None, f"呼叫 judge 失敗 (HTTP {status}): {resp}"
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None, None, None, f"judge 回應格式不如預期: {resp}"

    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        faithfulness = int(parsed["faithfulness"])
        relevance = int(parsed["relevance"])
        reasoning = str(parsed.get("reasoning", "")).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None, None, f"judge 輸出無法解析成 JSON，原始輸出前 300 字：{raw[:300]}"
    return faithfulness, relevance, reasoning, None


def qa_check(question: str, answer: str, context: str, completion_tokens, max_tokens: int):
    """對應 n8n「QA Parse & Threshold」節點的合併判斷邏輯。"""
    det = check_deterministic(answer, completion_tokens, max_tokens)
    faithfulness, relevance, reasoning, judge_err = call_judge(question, context, answer)

    judge_parse_error = judge_err is not None
    needs_review = (
        judge_parse_error
        or not det["deterministic_pass"]
        or faithfulness is None
        or faithfulness < FAITHFULNESS_THRESHOLD
    )
    return {
        **det,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "judge_reasoning": reasoning or judge_err,
        "judge_parse_error": judge_parse_error,
        "needs_review": needs_review,
    }


def heading_text(index: int, question: str) -> str:
    return f"{index}. {question}"


def slugify_heading(text: str) -> str:
    """Approximates GitHub-flavored markdown's own heading-anchor slugger
    closely enough for the TOC links to actually jump to the right
    section in GitHub, VS Code's previewer, and most other renderers:
    lowercase, strip punctuation (keep unicode letters/digits so CJK
    survives), turn runs of whitespace into single hyphens."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text.strip())


def filter_citations(hits, min_score: float = MIN_CITATION_SCORE):
    """只留下真的有支撐力的檢索結果。負向對照題常見的情況是 top-k 裡沒有一個
    真正相關的片段,只是矬中選優——全部濾掉比硬列低分來源湊數更誠實。"""
    return [h for h in hits if h.get("score", 0.0) >= min_score]


def render_section(index: int, result: dict, qa: dict, group_metric: str = None,
                    red_flag_title: str = None) -> str:
    lines = []
    lines.append(f"## {heading_text(index, result['question'])}")
    lines.append("")

    if group_metric:
        lines.append(f"🔗 屬於一致性檢查群組：{group_metric}")
        lines.append("")

    if red_flag_title:
        lines.append(f"🚩 此題內容被判定為紅旗：{red_flag_title}")
        lines.append("")

    if qa["needs_review"]:
        reason_bits = []
        if qa["judge_parse_error"]:
            reason_bits.append(f"judge 輸出無法解析（{qa['judge_reasoning']}）")
        elif not qa["deterministic_pass"]:
            checks = []
            if not qa["not_truncated"]:
                checks.append("生成疑似被截斷")
            if not qa["no_garbled"]:
                checks.append("偵測到亂碼/生成迴圈")
            reason_bits.append("、".join(checks))
        elif qa["faithfulness"] is not None and qa["faithfulness"] < FAITHFULNESS_THRESHOLD:
            reason_bits.append(f"忠實度 {qa['faithfulness']}/10（{qa['judge_reasoning']}）")
        reason = "；".join(b for b in reason_bits if b) or "未知原因"
        lines.append(f"> ⚠️ **此段落待人工複核** — {reason}")
        lines.append("")

    answer = result["visible"] or "（模型未輸出可見答案，可能只產生了推理內容）"
    lines.append(answer)
    lines.append("")

    lines.append("**參考來源**（依檢索相關度排序）：")
    cited_hits = filter_citations(result["hits"])
    if cited_hits:
        for hit in cited_hits:
            payload = hit.get("payload", {})
            heading = payload.get("heading_path", "?")
            score = hit.get("score", 0.0)
            lines.append(f"- {heading}（相關度 {score:.4f}）")
    else:
        lines.append(f"未檢索到相關度足夠的文件片段（門檻 {MIN_CITATION_SCORE}）。")
    lines.append("")

    faith_display = f"{qa['faithfulness']}/10" if qa["faithfulness"] is not None else "N/A"
    faith_flag = " ⚠️" if qa["needs_review"] else ""
    rel_display = f"{qa['relevance']}/10" if qa["relevance"] is not None else "N/A"
    think_display = "開啟" if result["think_used"] else "關閉"
    lines.append(
        f"**QA 檢查**：忠實度 {faith_display}{faith_flag} ｜ 相關性 {rel_display} ｜ "
        f"思考模式：{think_display} ｜ 耗時 {result['total']:.1f}s"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}


def render_red_flags_section(red_flags) -> str:
    """紅旗清單，放在目錄之後、數字一致性檢查之前——比一致性檢查更高階，
    更接近執行摘要的定位。只有 --detect-red-flags 開啟時才會有內容可放。"""
    if not red_flags:
        return ""
    lines = [
        "## 紅旗清單",
        "",
        "> ⚠️ **驗證邊界**：以下紅旗由模型從檢索片段中判斷「主張/數字是否互相"
        "矛盾」自動抽取，通過驗證只代表檢索與格式化正確，不是對模型語意理解"
        "能力的嚴格證明——實際判斷仍需人工核對原文。",
        "",
    ]
    for rf in red_flags:
        icon = SEVERITY_ICON.get(rf["severity"], "⚪")
        lines.append(f"### {icon} {rf['title']}")
        lines.append("")
        lines.append(f"**現象**：{rf['phenomenon']}")
        lines.append(f"**為何是紅旗**：{rf['why_it_matters']}")
        lines.append(f"**必要動作**：{rf['required_action']}")
        questions_str = "、".join(f"「{q}」" for q in rf["matched_questions"])
        headings = sorted(set(rf["headings"])) if rf["headings"] else []
        headings_str = "、".join(headings) if headings else "（無法定位到具體片段）"
        lines.append(f"**出現於**：{questions_str}（相關片段：{headings_str}）")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def render_consistency_section(consistency_results) -> str:
    """數字一致性檢查節，放在目錄之後、逐題內容之前——先看結論，再看細
    節。只有問題清單裡有 `## group:` 標記時才會有內容可放。"""
    if not consistency_results:
        return ""
    lines = ["## 數字一致性檢查", ""]
    for r in consistency_results:
        found = [e for e in r["entries"] if e["extraction"]["found"]]
        if r["status"] == "conflict":
            lines.append(f"### ⚠️ {r['metric']} — 數字不一致")
            lines.append("")
            for e in found:
                ext = e["extraction"]
                review_note = "（⚠️ 此題答案本身待人工複核，數字可信度打折扣）" if e["needs_review"] else ""
                heading = e["heading"] or "（無法定位到具體檢索片段）"
                lines.append(f"- 「{e['question']}」→ **{ext['raw_value']}**"
                              f"（來源：{heading}）{review_note}")
            lines.append("")
        elif r["status"] == "consistent":
            value = found[0]["extraction"]["raw_value"] if found else "?"
            lines.append(f"- ✅ **{r['metric']}**：{len(found)} 種問法皆得到一致數字（{value}）")
        else:
            lines.append(f"- ℹ️ **{r['metric']}**：資料不足以比對（成功抽取到數字的題目少於 2 題）")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def generate_report(questions, collection, top_k, max_tokens, temperature,
                     think_override, title, chat_url=CHAT_URL, chat_model=CHAT_MODEL,
                     detect_red_flags=False):
    ensure_collection(collection, create=False)

    sections = []
    toc = []
    needs_review_count = 0
    groups = {}  # group id -> {"metric": str, "entries": [...]}
    red_flag_entries = []  # [{"question":..., "detection":..., "heading":...}, ...]

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        if think_override is None:
            auto_think, reason = detect_think_reason(question)
            if auto_think:
                log(f"「{question}」{reason} → 本題自動切換為 --think")
            enable_thinking = DEFAULT_THINKING or auto_think
        else:
            enable_thinking = think_override

        log(f"({i}/{len(questions)}) 產生答案：{question}")
        result = answer_one(
            question, collection, top_k, max_tokens, temperature,
            show_think=False, show_context=False, enable_thinking=enable_thinking,
        )

        log(f"({i}/{len(questions)}) 跑 QA 檢查...")
        qa = qa_check(question, result["visible"], result["context"],
                      result["completion_tokens"], max_tokens)
        if qa["needs_review"]:
            needs_review_count += 1
            log(f"({i}/{len(questions)}) ⚠️ 標記為待人工複核")

        group_id, metric = item.get("group"), item.get("metric")
        if group_id:
            log(f"({i}/{len(questions)}) 抽取指標「{metric}」的數字...")
            extraction = extract_metric_value(
                metric, question, result["visible"], result["context"],
                chat_url, chat_model,
            )
            heading = locate_source_heading(extraction["source_snippet"], result["hits"])
            groups.setdefault(group_id, {"metric": metric, "entries": []})
            groups[group_id]["entries"].append({
                "question": question, "extraction": extraction,
                "heading": heading, "needs_review": qa["needs_review"],
            })

        red_flag_title = None
        if detect_red_flags:
            log(f"({i}/{len(questions)}) 判斷是否構成紅旗...")
            top_heading = None
            if result["hits"]:
                top_heading = result["hits"][0].get("payload", {}).get("heading_path")
            detection = detect_red_flag(
                question, result["visible"], result["context"], chat_url, chat_model,
            )
            red_flag_entries.append({
                "question": question, "detection": detection, "heading": top_heading,
            })
            if detection["is_red_flag"]:
                red_flag_title = detection["title"]
                log(f"({i}/{len(questions)}) 🚩 判定為紅旗：{red_flag_title}")

        anchor = slugify_heading(heading_text(i, question))
        toc.append(f"{i}. [{question}](#{anchor})")
        sections.append(render_section(i, result, qa, group_metric=metric,
                                        red_flag_title=red_flag_title))

    consistency_results = [
        check_group_consistency(g["metric"], g["entries"]) for g in groups.values()
    ]
    red_flags = dedupe_red_flags(red_flag_entries)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    status_line = (
        f"⚠️ {len(questions)} 段中有 {needs_review_count} 段待人工複核"
        if needs_review_count else "✅ 全數段落 QA 檢查通過"
    )

    header = [
        f"# {title}",
        "",
        f"**產生時間**：{now}",
        f"**報告狀態**：{status_line}",
        "",
        "## 目錄",
        *toc,
        "",
        "---",
        "",
    ]
    return ("\n".join(header) + "\n" + render_red_flags_section(red_flags)
            + render_consistency_section(consistency_results)
            + "\n".join(sections))


GROUP_HEADER_RE = re.compile(
    r"^##\s*group:\s*(?P<gid>[^|]+?)\s*\|\s*metric:\s*(?P<metric>.+?)\s*$",
    re.IGNORECASE,
)


def load_questions(path):
    """回傳 [{"question": str, "group": str|None, "metric": str|None}, ...]。

    `## group: <id> | metric: <名稱>` 是可選的分組標記，標記之後的問題都
    屬於該組，直到下一個標記或檔案結束；標記之前（或整份檔案都沒有標記）
    的問題 group/metric 都是 None，不參與數字一致性檢查——舊的問題清單
    檔案完全不用改，行為不變。
    """
    if not os.path.isfile(path):
        die(f"找不到問題清單檔案：{path}")
    questions = []
    current_group, current_metric = None, None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            m = GROUP_HEADER_RE.match(line)
            if m:
                current_group = m.group("gid").strip()
                current_metric = m.group("metric").strip()
                continue
            if line.startswith("#"):
                continue
            questions.append({
                "question": line, "group": current_group, "metric": current_metric,
            })
    return questions


def main():
    ap = argparse.ArgumentParser(description="批次跑 RAG 問答 + QA 把關，組成一份 Markdown 報告。")
    ap.add_argument("--questions", required=True, help="問題清單檔案，每行一題")
    ap.add_argument("--output", required=True, help="輸出的 Markdown 檔案路徑")
    ap.add_argument("--title", default="PSF EIM 問答報告", help="報告標題")
    ap.add_argument("--collection", default="psf_eim_kb", help="Qdrant collection 名稱")
    ap.add_argument("--top-k", type=int, default=8, help="檢索幾個 chunk 餵給模型 (預設 8)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="生成上限 (預設 2048)")
    ap.add_argument("--temperature", type=float, default=0.2, help="生成溫度 (預設 0.2)")
    think_group = ap.add_mutually_exclusive_group()
    think_group.add_argument("--think", dest="think", action="store_true", default=None,
                              help="整份報告強制開啟思考模式")
    think_group.add_argument("--no-think", dest="think", action="store_false", default=None,
                              help="整份報告強制關閉思考模式")
    ap.add_argument("--detect-red-flags", action="store_true", default=False,
                     help="對每一題的答案額外跑一次紅旗判斷（主張/數字矛盾），"
                          "彙整成報告開頭的「紅旗清單」章節；不開啟時行為不變")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        die(f"{args.questions} 裡沒有找到任何問題。")

    t0 = time.time()
    report = generate_report(
        questions, args.collection, args.top_k, args.max_tokens, args.temperature,
        args.think, args.title, detect_red_flags=args.detect_red_flags,
    )
    elapsed = time.time() - t0

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    log(f"報告已寫入 {args.output}（{len(questions)} 題，耗時 {elapsed:.1f}s）")


if __name__ == "__main__":
    main()
