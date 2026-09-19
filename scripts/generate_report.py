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
from rag_common import (  # noqa: E402
    die,
    ensure_collection,
    http_json,
    log,
    source_filter,
    verify_sources_exist,
)
from numeric_consistency import (  # noqa: E402
    check_group_consistency,
    extract_metric_value,
    locate_source_heading,
)
from red_flag_detection import (  # noqa: E402
    dedupe_red_flags,
    detect_red_flag,
)
from scoring import (  # noqa: E402
    compute_weighted_total,
    lookup_grade,
    score_dimension,
)
from veto_check import (  # noqa: E402
    VETO_RULES,
    detect_veto,
    summarize_veto_results,
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
                    red_flag_title: str = None, veto_title: str = None,
                    dimension_name: str = None) -> str:
    lines = []
    lines.append(f"## {heading_text(index, result['question'])}")
    lines.append("")

    if group_metric:
        lines.append(f"🔗 屬於一致性檢查群組：{group_metric}")
        lines.append("")

    if dimension_name:
        lines.append(f"🎯 屬於評分維度：{dimension_name}")
        lines.append("")

    if red_flag_title:
        lines.append(f"🚩 此題內容被判定為紅旗：{red_flag_title}")
        lines.append("")

    if veto_title:
        lines.append(f"🚫 此題內容被判定觸發不合作紅線：{veto_title}")
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
                source_note = f"［來源文件：{e['source']}］" if e.get("source") else ""
                lines.append(f"- 「{e['question']}」{source_note}→ **{ext['raw_value']}**"
                              f"（來源：{heading}）{review_note}")
            lines.append("")
        elif r["status"] == "consistent":
            value = found[0]["extraction"]["raw_value"] if found else "?"
            sources = sorted({e["source"] for e in found if e.get("source")})
            source_note = f"，涵蓋文件：{'、'.join(sources)}" if sources else ""
            lines.append(f"- ✅ **{r['metric']}**：{len(found)} 筆記錄皆得到一致數字"
                          f"（{value}）{source_note}")
        else:
            lines.append(f"- ℹ️ **{r['metric']}**：資料不足以比對（成功抽取到數字的題目少於 2 題）")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def load_scoring_template(path: str, template_name: str) -> dict:
    """讀 scripts/data/scoring_templates.json，回傳指定模板名稱的設定。
    模板名稱打錯字時直接死掉列出可用的模板，不要跑到評分階段才發現——
    跟 verify_sources_exist() 的 typo 前置檢查同一個原則。"""
    if not os.path.isfile(path):
        die(f"找不到評分模板設定檔：{path}")
    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    if template_name not in templates:
        die(f"評分模板 '{template_name}' 不存在。可用模板：{sorted(templates)}")
    return templates[template_name]


def compute_scoring_summary(dimension_results, template):
    """dimension_results 涵蓋 template 全部維度、且全部成功評分時，回傳
    {"total", "max_total", "grade"}；只要有維度缺評分或評分失敗，回傳
    None（跟 render_scoring_section() 判斷「能不能對照等級表」同一個
    邏輯）。抽成獨立函式是因為報告最上層的veto標示旁邊也要顯示PSF總評
    （整合驗證發現的缺陷：報頭只有veto結論、要看完整份報告才知道總分，
    兩個結論沒有真正並列），不能讓兩處各自重複一份「能不能算總分」的
    判斷邏輯，以後改了一邊忘記改另一邊。純邏輯，不牽涉 LLM。"""
    if not dimension_results:
        return None
    weights = template["dimensions"]
    missing_dims = set(weights) - set(dimension_results)
    failed_dims = {dim for dim, result in dimension_results.items() if result["score"] is None}
    if missing_dims or failed_dims:
        return None
    scored = {dim: result["score"] for dim, result in dimension_results.items()}
    total, max_total = compute_weighted_total(scored, template)
    return {"total": total, "max_total": max_total, "grade": lookup_grade(total, template)}


def build_scoring_banner_line(dimension_results, template) -> str:
    """報告最上層的PSF總評標示，跟不合作紅線標示並列顯示——即使報頭原本
    只有veto結論，讀者也能同時看到總分等級，不用往下翻到評分總表章節。
    沒有 dimension_results（沒開 --scoring-template，或問題清單沒有
    dimension 標記）或算不出總分（有維度缺評分/評分失敗）時回傳 None，
    這一行完全不顯示，跟 build_veto_banner_line() 同一個「沒用到就不
    出現」慣例。"""
    if not dimension_results or not template:
        return None
    summary = compute_scoring_summary(dimension_results, template)
    if summary is None:
        return None
    grade = summary["grade"]
    action_note = f" → {grade['action']}" if grade.get("action") else ""
    return (f"**PSF 總評**：{summary['total']}/{summary['max_total']} — "
            f"{grade['grade']}（{grade['label']}）{action_note}")


def render_scoring_section(dimension_results, template) -> str:
    """評分總表章節，放在紅旗清單、數字一致性檢查之後——評分是最下游、
    綜合前兩者產出的判斷，理當放在最後。只有 --scoring-template 開啟
    時才會有內容可放。dimension_results: {維度名稱: score_dimension()
    的回傳值}，只含問題清單裡實際出現過的維度。"""
    if not dimension_results:
        return ""
    weights = template["dimensions"]
    lines = [
        "## 評分總表",
        "",
        "> ⚠️ **驗證邊界**：維度評分經 LLM 輔助產生，多數維度的評分浮動落在"
        "±1 分內；契合、貢獻兩個維度因評分慣例本身較主觀、缺乏明確依據可"
        "精確重現，浮動範圍可能更大，建議優先人工複核。方向性判斷（維度"
        "優劣）具參考價值，精確刻度仍建議搭配人工複核。",
        "",
        "| 維度 | 分數 | 權重 | 加權小計 |", "|---|---|---|---|",
    ]

    scored = {}
    failed_dims = []
    for dim, result in dimension_results.items():
        if result["score"] is None:
            failed_dims.append(dim)
            lines.append(f"| {dim} | ⚠️ 評分失敗，需人工判斷 | ×{weights[dim]} | — |")
        else:
            scored[dim] = result["score"]
            subtotal = result["score"] * weights[dim]
            lines.append(f"| {dim} | {result['score']}/{template['max_score_per_dimension']} "
                         f"| ×{weights[dim]} | {subtotal} |")

    missing_dims = sorted(set(weights) - set(dimension_results))
    for dim in missing_dims:
        lines.append(f"| {dim} | ⚠️ 未評分（問題清單未涵蓋此維度） | ×{weights[dim]} | — |")

    lines.append("")
    if missing_dims or failed_dims:
        reasons = []
        if missing_dims:
            reasons.append(f"問題清單未涵蓋：{'、'.join(missing_dims)}")
        if failed_dims:
            reasons.append(f"評分失敗：{'、'.join(failed_dims)}")
        lines.append(f"> ⚠️ **無法計算總分**——{'；'.join(reasons)}。"
                     f"以下僅列出已成功評分的維度，不對照等級表。")
        lines.append("")
    else:
        summary = compute_scoring_summary(dimension_results, template)
        grade = summary["grade"]
        action_note = f" → {grade['action']}" if grade.get("action") else ""
        lines.append(f"**PSF 總評：{summary['total']}/{summary['max_total']} — "
                     f"{grade['grade']}（{grade['label']}）{action_note}**")
        lines.append("")

    for dim, result in dimension_results.items():
        if result["score"] is None:
            continue
        lines.append(f"### {dim}（{result['score']}/{template['max_score_per_dimension']}）")
        lines.append("")
        lines.append(f"**評分依據**：{result['rationale']}")
        lines.append("**引用**：")
        for item in result["evidence"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def render_veto_section(veto_results) -> str:
    """不合作紅線檢查章節，放在評分總表之後——這是比評分更底線的判斷，
    不管九格總分多高，觸發任一條veto都建議不合作，理當放在報告最後面
    的判斷章節。veto_results: summarize_veto_results() 的回傳值，只有
    問題清單裡有 `## veto:` 標記時才會有內容可放。"""
    if not veto_results:
        return ""
    lines = [
        "## 不合作紅線檢查",
        "",
        "> ⚠️ **驗證邊界**：以下判斷依《PSF六壬合夥生態系統》「五不合作」"
        "條款，由模型輔助判斷。矛盾型（資源不實）跟缺失型（人不明、"
        "權責不清、利益不明、風險不揭露）用不同的判斷邏輯——矛盾型要找"
        "『主張跟查核結果對不上』，缺失型要找『文件明確承認揭露不完整』，"
        "單純沒提到不算缺失。通過驗證只代表檢索與格式化正確，不是對模型"
        "語意理解能力的嚴格證明，實際判斷仍需人工核對原文。",
        "",
    ]
    for rule_name, result in veto_results.items():
        if not result["triggered"]:
            lines.append(f"- ✅ **{rule_name}**：未觸發")
            continue
        lines.append(f"### 🚫 {rule_name}")
        lines.append("")
        for entry in result["entries"]:
            lines.append(f"**現象**：{entry['phenomenon']}")
            lines.append(f"**依據**：{entry['basis']}")
            lines.append(f"**建議**：{entry['suggested_action']}")
            heading = entry["heading"] or "（無法定位到具體片段）"
            lines.append(f"**出現於**：「{entry['question']}」（相關片段：{heading}）")
            lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_veto_banner_line(veto_results) -> str:
    """報告最上層的veto否決標示，跟報告狀態行並列——即使九格總分是A級
    核心夥伴，觸發veto時這一行一樣顯示「建議不合作」，兩個結論並存，
    不互相掩蓋。沒有任何veto被標記測試過時回傳 None，這一行完全不顯示
    （跟章節「沒用到就不出現」同一個慣例，避免舊報告多一行視覺噪音）。
    純邏輯，抽成獨立函式方便測試這段組字邏輯。"""
    if not veto_results:
        return None
    triggered_rules = [rule for rule, result in veto_results.items() if result["triggered"]]
    if triggered_rules:
        return f"**不合作紅線**：🚫 觸發「{'、'.join(triggered_rules)}」——建議不合作"
    return "**不合作紅線**：✅ 未觸發"


def generate_report(questions, collection, top_k, max_tokens, temperature,
                     think_override, title, chat_url=CHAT_URL, chat_model=CHAT_MODEL,
                     detect_red_flags=False, scoring_template=None):
    ensure_collection(collection, create=False)
    referenced_sources = [item.get("source") for item in questions if item.get("source")]
    if referenced_sources:
        verify_sources_exist(collection, referenced_sources)
    if scoring_template:
        referenced_dims = {item.get("dimension") for item in questions if item.get("dimension")}
        unknown_dims = sorted(referenced_dims - set(scoring_template["dimensions"]))
        if unknown_dims:
            die(f"問題清單裡的維度名稱跟模板對不上：{unknown_dims}。"
                f"模板裡的維度：{sorted(scoring_template['dimensions'])}")
    referenced_vetoes = {item.get("veto") for item in questions if item.get("veto")}
    unknown_vetoes = sorted(referenced_vetoes - set(VETO_RULES))
    if unknown_vetoes:
        die(f"問題清單裡的veto規則名稱跟文件定義的五條對不上：{unknown_vetoes}。"
            f"合法規則：{sorted(VETO_RULES)}")
    # 為什麼跨文件比對需要真正的 source 篩選、不能只靠語意相似度自然分開：
    # 實測過同一句問題不加篩選檢索兩份不同文件時，最高分是文件A的片段
    # （0.7247），但第二名是文件B的片段（0.6504）——分數差距不大，代表
    # embedding 的語意相似度並不會自動尊重文件邊界，兩份談論同一個指標的
    # 文件內容分數本來就會很接近。沒有 query_filter 真正限制檢索範圍，
    # 兩份文件的內容就是會混在一起。

    sections = []
    toc = []
    needs_review_count = 0
    groups = {}  # group id -> {"metric": str, "entries": [...]}
    red_flag_entries = []  # [{"question":..., "detection":..., "heading":...}, ...]
    dimension_entries = {}  # 維度名稱 -> [{"question","answer","detection","group_id"}, ...]
    veto_entries = []  # [{"question":..., "rule":..., "detection":..., "heading":...}, ...]

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        if think_override is None:
            auto_think, reason = detect_think_reason(question)
            if auto_think:
                log(f"「{question}」{reason} → 本題自動切換為 --think")
            enable_thinking = DEFAULT_THINKING or auto_think
        else:
            enable_thinking = think_override

        source = item.get("source")
        log(f"({i}/{len(questions)}) 產生答案：{question}"
            + (f"（限定 source={source}）" if source else ""))
        result = answer_one(
            question, collection, top_k, max_tokens, temperature,
            show_think=False, show_context=False, enable_thinking=enable_thinking,
            query_filter=source_filter(source) if source else None,
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
                "source": source,
            })

        red_flag_title = None
        detection = None  # 這一題的紅旗判斷結果——維度評分的 bundle 也會用到
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

        dimension = item.get("dimension")
        if dimension:
            dimension_entries.setdefault(dimension, []).append({
                "question": question, "answer": result["visible"],
                "detection": detection, "group_id": group_id,
            })

        veto_title = None
        veto_rule = item.get("veto")
        if veto_rule:
            log(f"({i}/{len(questions)}) 檢查是否觸發不合作紅線「{veto_rule}」...")
            top_heading = None
            if result["hits"]:
                top_heading = result["hits"][0].get("payload", {}).get("heading_path")
            veto_detection = detect_veto(
                veto_rule, question, result["visible"], result["context"], chat_url, chat_model,
            )
            veto_entries.append({
                "question": question, "rule": veto_rule,
                "detection": veto_detection, "heading": top_heading,
            })
            if veto_detection["is_veto_triggered"]:
                veto_title = veto_rule
                log(f"({i}/{len(questions)}) 🚫 觸發不合作紅線：{veto_rule}")

        anchor = slugify_heading(heading_text(i, question))
        toc.append(f"{i}. [{question}](#{anchor})")
        sections.append(render_section(i, result, qa, group_metric=metric,
                                        red_flag_title=red_flag_title, veto_title=veto_title,
                                        dimension_name=dimension))

    consistency_results = [
        check_group_consistency(g["metric"], g["entries"]) for g in groups.values()
    ]
    red_flags = dedupe_red_flags(red_flag_entries)

    dimension_results = {}
    if scoring_template:
        for dimension, entries in dimension_entries.items():
            log(f"評分維度「{dimension}」...")
            dimension_results[dimension] = score_dimension(
                dimension, entries, consistency_results, chat_url, chat_model,
            )

    veto_results = summarize_veto_results(veto_entries) if veto_entries else {}

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
    ]
    scoring_banner = (
        build_scoring_banner_line(dimension_results, scoring_template) if scoring_template else None
    )
    if scoring_banner:
        header.append(scoring_banner)
    veto_banner = build_veto_banner_line(veto_results)
    if veto_banner:
        header.append(veto_banner)
    header += [
        "",
        "## 目錄",
        *toc,
        "",
        "---",
        "",
    ]
    scoring_section = (
        render_scoring_section(dimension_results, scoring_template) if scoring_template else ""
    )
    return ("\n".join(header) + "\n" + render_red_flags_section(red_flags)
            + render_consistency_section(consistency_results)
            + scoring_section
            + render_veto_section(veto_results)
            + "\n".join(sections))


GROUP_HEADER_RE = re.compile(
    r"^##\s*group:\s*(?P<gid>[^|]+?)\s*\|\s*metric:\s*(?P<metric>.+?)\s*$",
    re.IGNORECASE,
)

# `## group: none` 明確清除目前生效的 group/metric——三種標記都是「宣告後
# 持續生效到下一個同類標記」，原本沒有清除語法，這是整合驗證（五種能力
# 同時啟用）第一次真正跑出跨主題問題清單時抓到的資料污染缺陷：同一份問題
# 清單裡先問完一家公司再換另一家公司（或換一份不相關的文件）時，前一段
# 的標記會不小心延續到新主題的題目上，見 load_questions() docstring 最後
# 那段「已知的資料污染案例」。
GROUP_CLEAR_RE = re.compile(r"^##\s*group:\s*none\s*$", re.IGNORECASE)

# 故意不接受 `| weight: N`——權重固定寫在模板設定檔裡（見
# scripts/data/scoring_templates.json），不做每案客製。問題清單裡如果
# 也能寫權重，會出現「檔案裡的權重」跟「模板裡的權重」兩個來源，容易對
# 不上又沒人發現；標記只給維度名稱，權重永遠只從模板查。
DIMENSION_HEADER_RE = re.compile(
    r"^##\s*dimension:\s*(?P<dim>.+?)\s*$", re.IGNORECASE,
)

# 跟 DIMENSION_HEADER_RE 同一個道理：veto規則名稱是《PSF六壬合夥生態
# 系統》第十四節定義的固定五條（見 veto_check.VETO_RULES），不是每案
# 客製的東西，這裡只接受規則名稱，合法性檢查在 generate_report() 裡跑
# 報告前就做（跟 unknown_dims 檢查同一個模式）。
VETO_HEADER_RE = re.compile(
    r"^##\s*veto:\s*(?P<rule>.+?)\s*$", re.IGNORECASE,
)

# dimension/veto 的清除不需要獨立的正規表達式——`## dimension: none`／
# `## veto: none` 本身就會被上面兩個 HEADER_RE 正常捕捉成 dim="none"／
# rule="none"，load_questions() 只要在賦值前檢查捕捉到的文字是不是（不分
# 大小寫）"none" 就能決定要清除還是要設值。這跟 group 不一樣——group 的
# 語法多了 `| metric:` 這段，"## group: none" 缺了這段本來就配不到
# GROUP_HEADER_RE，所以 group 才需要 GROUP_CLEAR_RE 這個獨立規則。
# 三種規則名稱（人不明/資源不實/...）跟維度名稱（信用/資源/...，來自
# scripts/data/scoring_templates.json）都是固定中文詞彙，跟英文的 "none"
# 不會衝突，用字面比對是安全的。
_CLEAR_VALUE = "none"

SOURCE_TAG_RE = re.compile(r"^\[source:\s*(?P<src>[^\]]+)\]\s*(?P<rest>.*)$")


def parse_source_tag(line: str):
    """剝掉一行問題前面選擇性的 `[source: <檔名>]` 前綴。回傳
    (source, remaining_question)——沒有前綴時 source 是 None、
    remaining_question 是原字串不變。純邏輯，跟 load_questions() 的檔案
    I/O 分開，方便測試。"""
    m = SOURCE_TAG_RE.match(line)
    if not m:
        return None, line
    return m.group("src").strip(), m.group("rest").strip()


def load_questions(path):
    """回傳 [{"question": str, "group": str|None, "metric": str|None,
    "source": str|None, "dimension": str|None, "veto": str|None}, ...]。

    `## group: <id> | metric: <名稱>` 是可選的分組標記，標記之後的問題都
    屬於該組，直到下一個標記或檔案結束；標記之前（或整份檔案都沒有標記）
    的問題 group/metric 都是 None，不參與數字一致性檢查。

    `## dimension: <名稱>` 是另一個獨立、可選的分組標記，標記之後的問題
    都屬於該評分維度，直到下一個 dimension 標記或檔案結束——跟 `## group:`
    互不影響、可以同時生效（同一題可以既屬於一個數字一致性檢查群組，又
    屬於一個評分維度）。維度只寫名稱，不寫權重——見上面 DIMENSION_HEADER_RE
    的說明。

    `## veto: <規則名稱>` 是第三個獨立、可選的分組標記，標記之後的問題
    都屬於該veto規則的檢查對象，直到下一個 veto 標記或檔案結束——跟
    `## group:`/`## dimension:` 一樣互不影響、可以同時生效。veto不需要
    每題都測，完全由人工標記決定這題要測哪一條，不做自動判斷。

    每一行問題前面可以選擇性加 `[source: <檔名>]` 前綴，把這一題的檢索範圍
    限定在該來源文件——跨文件比對時，同一個 group 底下不同行指向不同
    source，就是「同一個問題分別在不同文件範圍內各檢索一次」。沒有這個
    前綴時 source 是 None，檢索範圍不受限，跟舊行為一致。

    這三個標記都是可選、獨立的——舊的問題清單檔案完全不用改。

    **標記清除語法**：`## group: none`／`## dimension: none`／
    `## veto: none` 明確清除目前生效的對應標記，之後的題目回到「不屬於
    任何群組/維度/veto規則」的狀態，直到遇到下一個真正的標記。三種標記
    互相獨立，各自要清除各自宣告。

    設計問題清單時的兩個真實教訓：

    1.（跨文件比對驗證時踩過）每一行的問法要夠精確，指向該份文件裡最
       明確記載這個指標的段落，不要用籠統問法。用同一句籠統問題（例如
       「這份文件裡2026年營收預測是多少？」）分別對兩份文件各問一次，
       如果其中一份文件本身內部就已經對這個指標有多個版本（例如 DD 報告
       自己就在講「同一份文件三個數字」），模型會把「文件內部已知矛盾」
       整段摘要出來當答案，抽取邏輯反而可能從裡面挑出一個跟另一份文件
       恰好一樣的數字，讓真正該抓到的跨文件矛盾被蓋掉、誤判成一致。
       「同一份文件內部已知的矛盾」跟「這次要測的跨文件矛盾」是兩件不同
       的事，問法必須夠精確才能把兩者分開。

       這個教訓在整合驗證（五種能力同時啟用）時被重新踩到一次：問「DD
       報告查核後認定的2026年營收預測金額」，但 DD 報告從來沒有給出一個
       單一認定值——它記錄的正是「同一份文件出現三個版本」這件事本身，
       所以那題答案裡沒有可抽取的單一數字，跨文件比對群組顯示「資料不足
       以比對」。問法本身要問「文件裡列出的數字是多少」，不能問「認定的
       數字是多少」去假設一個文件本身沒給出的結論。

    2.（整合驗證第一次真正跑跨主題問題清單時踩到，之前每個功能分開驗證
       時從來沒遇到）三種標記都是「宣告後持續生效到下一個同類標記」，
       同一份問題清單裡如果先問完一家公司/一份主題再換到另一家公司/
       另一份不相關文件，前一段殘留的標記會不小心延續到新主題的題目上，
       造成真實的資料污染，不是理論風險。實際發生過的兩個案例：
       (a) veto誤判——換到 Branes.AI 的跨文件比對題時，沒清掉前一段
       日羿智能的 `## veto: 風險不揭露`，這題被誤判觸發，跟真正的股權
       未揭露案例並列在同一條規則底下，但這題的內容（DD報告沒寫出最終
       認定的營收數字）根本不屬於風險不揭露的定義範圍（法律/債務/糾紛/
       黑箱/隱性股東不揭露）。(b) 評分污染——同一個原因，日羿的
       `## dimension: 信用` 沒清掉，Branes 的營收數字矛盾被當成日羿信用
       維度的評分依據引用進去，實際進了日羿的總分計算。換主題、換文件
       時，一定要用清除語法把上一段的標記收掉，不要假設「反正這題有
       source標記限定檢索範圍，標記應該不會有影響」——source只限制檢索
       範圍，不影響 group/dimension/veto 標記的繼承邏輯，兩者是獨立的
       機制。
    """
    if not os.path.isfile(path):
        die(f"找不到問題清單檔案：{path}")
    questions = []
    current_group, current_metric = None, None
    current_dimension = None
    current_veto = None
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
            if GROUP_CLEAR_RE.match(line):
                current_group, current_metric = None, None
                continue
            dm = DIMENSION_HEADER_RE.match(line)
            if dm:
                dim_value = dm.group("dim").strip()
                current_dimension = None if dim_value.lower() == _CLEAR_VALUE else dim_value
                continue
            vm = VETO_HEADER_RE.match(line)
            if vm:
                veto_value = vm.group("rule").strip()
                current_veto = None if veto_value.lower() == _CLEAR_VALUE else veto_value
                continue
            if line.startswith("#"):
                continue
            source, line = parse_source_tag(line)
            questions.append({
                "question": line, "group": current_group, "metric": current_metric,
                "source": source, "dimension": current_dimension, "veto": current_veto,
            })
    return questions


def render_tag_table(questions) -> str:
    """把 load_questions() 解析出的標記逐題印成一份對照表——`--dry-run`
    用這個在真正花時間跑報告之前，讓人工先自己核對標記有沒有正確延續/
    清除，特別是跨主題問題清單（換公司、換文件）最容易在這裡出問題（見
    load_questions() docstring 裡記錄的資料污染案例：整合驗證第一次真正
    跑跨主題問題清單時，才發現前一段的標記會不小心延續到新主題的題目
    上，那次錯誤花了完整跑一次報告、包括全部 LLM 呼叫才發現）。純邏輯，
    不牽涉 LLM，不觸發任何檢索或生成。"""
    lines = [
        f"{'#':<4}{'source':<28}{'group':<10}{'metric':<18}{'dimension':<16}{'veto':<12}問題",
        "-" * 110,
    ]
    for i, item in enumerate(questions, start=1):
        lines.append(
            f"{i:<4}{item['source'] or '-':<28}{item['group'] or '-':<10}"
            f"{item['metric'] or '-':<18}{item['dimension'] or '-':<16}"
            f"{item['veto'] or '-':<12}{item['question']}"
        )
    return "\n".join(lines)


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
    ap.add_argument("--scoring-template", default=None,
                     help="開啟客製評分維度，值是模板名稱（例如「九格」，"
                          "定義在 --scoring-templates-file）；不指定時完全不跑"
                          "評分。建議搭配 --detect-red-flags 一起開，評分依據"
                          "才會引用到紅旗——沒開紅旗偵測時評分依據只能來自答案"
                          "內容跟數字一致性檢查結果")
    ap.add_argument("--scoring-templates-file",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "data", "scoring_templates.json"),
                     help="評分模板設定檔路徑（預設 scripts/data/scoring_templates.json）")
    ap.add_argument("--dry-run", action="store_true", default=False,
                     help="只解析問題清單、印出每題的 source/group/metric/"
                          "dimension/veto 標記對照表，不跑檢索或生成，也不"
                          "產生報告——設計跨主題問題清單時先用這個自檢標記"
                          "有沒有正確延續/清除")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        die(f"{args.questions} 裡沒有找到任何問題。")

    if args.dry_run:
        print(render_tag_table(questions))
        return

    scoring_template = None
    if args.scoring_template:
        scoring_template = load_scoring_template(args.scoring_templates_file, args.scoring_template)

    t0 = time.time()
    report = generate_report(
        questions, args.collection, args.top_k, args.max_tokens, args.temperature,
        args.think, args.title, detect_red_flags=args.detect_red_flags,
        scoring_template=scoring_template,
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
