#!/usr/bin/env python3
"""
generate_marketing_copy.py

行銷推廣文案生成——跟 generate_report.py 完全相反的用途：那邊是懷疑取向，
找矛盾/缺失/衝突，組成 Markdown 報告；這裡是正向包裝客戶公司優點，組成
給外部看的 HTML 文案。技術骨幹沿用（ingest→RAG檢索→LLM生成），但判斷
邏輯、輸出格式、品質標準是全新設計，不擴充 generate_report.py 的DD邏輯，
也不修改它。

架構（見 marketing_copy_spec.md，已核准）：這是orchestrator，兩塊判斷
邏輯拆到獨立檔案——marketing_faithfulness.py（正向忠實度檢查：賣點主張
有沒有在來源文件裡找到依據）、marketing_tone_score.py（說服力/語氣評分，
沒有客觀對錯，驗證方式待另外討論，這裡只先做出評分機制）。

明確排除網路爬蟲/外部查證：所有賣點的依據只能來自客戶自己提供、已經
ingest進collection的文件，這裡不會呼叫任何外部搜尋工具——跟 rag_answer.py/
generate_report.py 的既有行為一致，只是這裡明文重申。

風格參考庫（--style-collection）目前唯一現成的內容是佔位測試資料
（scripts/data/marketing_style_reference_PLACEHOLDER.md，檔名帶
PLACEHOLDER），只驗證「風格庫檢索+生成」這條技術路徑走得通，不是正式
業界範例——所以預設不開啟（--style-collection 不指定時完全不查詢），
避免每次生成都預設帶進非正式的佔位內容。正式素材之後提供後，重新
ingest到同一個collection名稱即可，不需要改程式碼。

Usage:
  python3 scripts/generate_marketing_copy.py \\
      --collection marketing_acme \\
      --brief "產品的核心優勢與五大能力矩陣" \\
      --layout 重點條列型 \\
      --output out/acme_copy.html
"""
import argparse
import html as html_lib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_answer import CHAT_MODEL, CHAT_URL, build_context  # noqa: E402
from rag_common import (  # noqa: E402
    JSON_OUTPUT_REMINDER,
    die,
    embed_one,
    ensure_collection,
    http_json,
    log,
    search,
)
from marketing_faithfulness import (  # noqa: E402
    ALERT_RED,
    ALERT_YELLOW,
    alert_levels,
    check_marketing_claims,
    load_first_json_object,
    normalize_claim_text,
)
from marketing_tone_score import score_marketing_tone  # noqa: E402

# 文案照樣寫出（人工要看得到內容才能修），但用非 0 結束碼讓任何包在外面的
# 自動化流程不會把它當成成功的成品。跟 die() 的 1 分開，1 代表根本沒產出。
EXIT_NEEDS_ATTENTION = 3

_LAYOUT_CONTENT_SCHEMAS = {
    "重點條列型": {
        "instruction": (
            "輸出 title（標題）、bullets（3-5條重點條列，每條一句話講清楚"
            "一個具體賣點）、cta（呼籲行動的短句，例如「立即預約產品導覽」）"
        ),
        "json_shape": '{"title": "<字串>", "bullets": ["<字串>", ...], "cta": "<字串>"}',
        "required_keys": ("title", "bullets", "cta"),
    },
    "故事敘述型": {
        "instruction": (
            "輸出 title（標題）、story_paragraphs（3-5段敘事，依緣起/挑戰/"
            "解方的脈絡展開）、cta（呼籲行動的短句）"
        ),
        "json_shape": '{"title": "<字串>", "story_paragraphs": ["<字串>", ...], "cta": "<字串>"}',
        "required_keys": ("title", "story_paragraphs", "cta"),
    },
    "數據導向型": {
        "instruction": (
            "輸出 title（標題）、stats（3-4個關鍵數字方塊，每個有 number "
            "跟 label 兩個欄位）、body_paragraphs（1-3段補充說明）、cta"
            "（呼籲行動的短句）"
        ),
        "json_shape": (
            '{"title": "<字串>", "stats": [{"number": "<字串>", '
            '"label": "<字串>"}, ...], "body_paragraphs": ["<字串>", ...], '
            '"cta": "<字串>"}'
        ),
        "required_keys": ("title", "stats", "body_paragraphs", "cta"),
    },
}


def build_marketing_copy_system_prompt(layout_name: str) -> str:
    schema = _LAYOUT_CONTENT_SCHEMAS[layout_name]
    return (
        "你是資深行銷文案寫手。你的任務是根據提供的「來源文件片段」，寫一份"
        "正向包裝的行銷文案，用來對外推廣客戶公司的產品/服務。\n"
        "重要規則：\n"
        "1. 每一條具體賣點主張（產品特點、能力、規模、數字等）都只能來自"
        "「來源文件片段」，不能引用「風格參考」（如果有提供）裡的任何具體"
        "事實或數字——風格參考只能用來學習語氣、結構、節奏，不能挪用裡面"
        "的內容\n"
        "2. 可以用有說服力、正向的語氣包裝這些事實，但不能無中生有捏造"
        "文件裡沒有的具體主張\n"
        "3. 不能編造引用來源——除非來源文件片段裡真的有那段調查/研究/觀察，"
        "否則不要寫「根據某某的觀察」「研究顯示」「專家指出」這類歸因句\n"
        f"4. {schema['instruction']}\n"
        f"{JSON_OUTPUT_REMINDER}\n"
        f"請只輸出一個 JSON 物件，格式為：{schema['json_shape']}\n"
        "不要有任何其他文字、不要用 markdown code fence。"
    )


def parse_marketing_copy_content(raw: str, layout_name: str) -> dict:
    """把模型輸出解析成文案結構化內容。缺少該樣板要求的必要欄位、或欄位
    型態不對（例如 bullets 不是非空字串清單）都視為抽取失敗——跟其他
    structured JSON 抽取primitive同一個「不半殘顯示」原則，寫壞掉的文案
    內容不該被拿去套版。"""
    try:
        parsed = load_first_json_object(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"content": None, "parse_error": f"文案生成結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}"}

    if not isinstance(parsed, dict):
        return {"content": None, "parse_error": f"生成結果不是 JSON 物件：{parsed}"}

    required = _LAYOUT_CONTENT_SCHEMAS[layout_name]["required_keys"]
    if any(k not in parsed for k in required):
        return {"content": None, "parse_error": f"缺少必要欄位（需要 {required}）：{parsed}"}

    title, cta = parsed.get("title"), parsed.get("cta")
    if not title or not cta:
        return {"content": None, "parse_error": f"title/cta 不能空白：{parsed}"}

    if layout_name == "重點條列型":
        bullets = parsed.get("bullets")
        if not isinstance(bullets, list) or not bullets or not all(
            isinstance(b, str) and b for b in bullets
        ):
            return {"content": None, "parse_error": f"bullets 不是非空字串清單：{parsed}"}
    elif layout_name == "故事敘述型":
        paragraphs = parsed.get("story_paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs or not all(
            isinstance(p, str) and p for p in paragraphs
        ):
            return {"content": None, "parse_error": f"story_paragraphs 不是非空字串清單：{parsed}"}
    elif layout_name == "數據導向型":
        stats = parsed.get("stats")
        body_paragraphs = parsed.get("body_paragraphs")
        valid_stats = isinstance(stats, list) and stats and all(
            isinstance(s, dict) and s.get("number") and s.get("label") for s in stats
        )
        valid_body = isinstance(body_paragraphs, list) and body_paragraphs and all(
            isinstance(p, str) and p for p in body_paragraphs
        )
        if not valid_stats or not valid_body:
            return {"content": None, "parse_error": f"stats/body_paragraphs 格式不對：{parsed}"}
    else:
        return {"content": None, "parse_error": f"未知樣板：{layout_name}"}

    return {"content": parsed, "parse_error": None}


def _call_marketing_copy_model(brief, layout_name, context, style_context,
                                chat_url, chat_model, max_tokens, temperature, timeout):
    system_prompt = build_marketing_copy_system_prompt(layout_name)
    user_parts = [
        f"這份文案的主題/重點：{brief}",
        f"---\n\n來源文件片段（賣點依據只能來自這裡）：\n{context}",
    ]
    if style_context:
        user_parts.append(
            "---\n\n風格參考（只能學習語氣/結構/節奏，不能挪用裡面的具體"
            f"主張或數字）：\n{style_context}"
        )
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return {"content": None, "parse_error": f"呼叫文案生成模型失敗 (HTTP {status}): {resp}"}
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return {"content": None, "parse_error": f"文案生成模型回應格式不如預期: {resp}"}
    return parse_marketing_copy_content(raw, layout_name)


def generate_marketing_copy_content(brief, layout_name, context, style_context,
                                     chat_url, chat_model, max_tokens=2048,
                                     temperature=0.3, timeout=180):
    """生成一份文案的結構化內容（JSON），layout_name決定必要欄位schema。
    重試一次：JSON解析失敗時用同一份輸入再呼叫一次——跟 scoring.py 的
    score_dimension() 同一個理由，偶發的JSON格式失誤重試成本很低，一次
    生成失敗代價（整份文案生不出來）偏高。"""
    result = _call_marketing_copy_model(
        brief, layout_name, context, style_context,
        chat_url, chat_model, max_tokens, temperature, timeout,
    )
    if result["parse_error"] is not None:
        result = _call_marketing_copy_model(
            brief, layout_name, context, style_context,
            chat_url, chat_model, max_tokens, temperature, timeout,
        )
    return result


# 兩級警示在視覺上刻意做到一眼能分：紅＝實線紅框＋「無依據」，黃＝虛線
# 黃框＋「待核實」。循環引用的誤報比真正捏造多，如果兩者長得一樣，看久了
# 會連紅色也一起習慣性略過。
_FLAG_STYLES = {
    ALERT_RED: ("background:#fee2e2;outline:2px solid #dc2626;", "⚠ 無依據："),
    ALERT_YELLOW: ("background:#fef9c3;outline:2px dashed #ca8a04;", "? 待核實："),
}


def flag_map(faithfulness) -> dict:
    """{要標記的文字: 警示等級}。無依據/待核實的事實主張依各自等級；整行
    漏檢的內容一律標紅（根本沒檢查到，不是循環引用）。純邏輯。"""
    if faithfulness is None or not faithfulness["claims"]:
        return {}
    flagged = {c["claim"]: c["alert"] for c in faithfulness["claims"] if c.get("alert")}
    for line in faithfulness["uncovered_lines"]:
        flagged[line] = ALERT_RED
    return flagged


def _flag_level(text: str, flagged: dict):
    """這段文字包含哪一級的警示主張（同時命中紅跟黃時取紅）。只做逐字子
    字串比對，所以是 best-effort：模型把主張改寫過一遍時會比對不到——這種
    情況靠頁首警示橫幅的完整清單兜底，不是靠這裡。"""
    norm = normalize_claim_text(text)
    hits = {level for claim, level in flagged.items()
            if normalize_claim_text(claim) and normalize_claim_text(claim) in norm}
    if ALERT_RED in hits:
        return ALERT_RED
    return ALERT_YELLOW if hits else None


def _cell(text: str, flagged: dict) -> str:
    escaped = html_lib.escape(text)
    level = _flag_level(text, flagged)
    if level:
        style, label = _FLAG_STYLES[level]
        return f'<mark style="{style}">{label}{escaped}</mark>'
    return escaped


def _stat_block(stat: dict, flagged: dict) -> str:
    """數據導向型的一個數字方塊。數字跟說明在 HTML 裡是兩個格子，但檢查器
    看到的純文字是「數字 說明」一整行（render_marketing_plaintext）——只逐格
    比對的話，「100+ 企業智慧協作案例」這種主張哪一格都不完整包含，整個方塊
    不會被標出來（實測漏掉過）。所以先用合起來的文字比對，命中就框整個方塊。"""
    level = _flag_level(f"{stat['number']} {stat['label']}", flagged)
    if level:
        style, label = _FLAG_STYLES[level]
        return (f'<div class="mc-stat" style="{style}">'
                f'<div class="mc-stat-num">{html_lib.escape(stat["number"])}</div>'
                f'<div class="mc-stat-label">{label}{html_lib.escape(stat["label"])}</div></div>')
    return (f'<div class="mc-stat"><div class="mc-stat-num">{_cell(stat["number"], flagged)}</div>'
            f'<div class="mc-stat-label">{_cell(stat["label"], flagged)}</div></div>')


def render_marketing_html(layout_name: str, template: dict, content: dict,
                          flagged=None) -> str:
    """把 generate_marketing_copy_content() 產出的結構化內容套進對應樣板的
    HTML template。純邏輯，不牽涉LLM。每個文字欄位都先 html.escape() 再
    嵌入，避免生成內容裡萬一出現 <, >, & 等字元弄壞版面——CSS本身包含
    大量 {}，但 str.format() 只掃描 html_template 字串裡的欄位，不會
    重新解析代入值(css)裡的花括號，兩者不會互相干擾。

    flagged：flag_map() 的結果，包含這些主張的元素會依等級加上紅框或黃框。
    <title> 標籤只放純文字，不加標記。"""
    flagged = flagged or {}
    css = template["css"]
    page_title = html_lib.escape(content["title"])
    title = _cell(content["title"], flagged)
    cta = _cell(content["cta"], flagged)
    if layout_name == "重點條列型":
        bullets_html = "".join(f"<li>{_cell(b, flagged)}</li>" for b in content["bullets"])
        body = template["html_template"].format(
            title=title, css=css, bullets_html=bullets_html, cta=cta,
        )
    elif layout_name == "故事敘述型":
        story_html = "".join(f"<p>{_cell(p, flagged)}</p>" for p in content["story_paragraphs"])
        body = template["html_template"].format(
            title=title, css=css, story_html=story_html, cta=cta,
        )
    elif layout_name == "數據導向型":
        stats_html = "".join(_stat_block(s, flagged) for s in content["stats"])
        body_html = "".join(f"<p>{_cell(p, flagged)}</p>" for p in content["body_paragraphs"])
        body = template["html_template"].format(
            title=title, css=css, stats_html=stats_html, body_html=body_html, cta=cta,
        )
    else:
        raise ValueError(f"未知樣板：{layout_name}")  # 不該發生，main()已在CLI階段用die()擋掉打錯字
    return re.sub(r"<title>.*?</title>", lambda _m: f"<title>{page_title}</title>", body, count=1)


_BANNER_COLORS = {
    ALERT_RED: ("#fef2f2", "#dc2626", "#7f1d1d", "solid"),
    ALERT_YELLOW: ("#fefce8", "#ca8a04", "#713f12", "dashed"),
}


def _banner_section(level: str, title: str, items) -> str:
    bg, border, fg, line = _BANNER_COLORS[level]
    lis = "".join(f'<li style="margin:6px 0">{i}</li>' for i in items)
    return (
        f'<div style="background:{bg};border:3px {line} {border};color:{fg};'
        'padding:12px 16px;margin:8px 0;font-family:sans-serif;font-size:15px;line-height:1.6">'
        f'<div style="font-weight:700;margin-bottom:6px">{title}</div>'
        f'<ul style="margin:0;padding-left:20px">{lis}</ul></div>'
    )


def build_warning_banner(faithfulness, skipped: bool) -> str:
    """文案不能被當成「查核過、沒問題」交出去時，回傳一段放在頁首的警示橫幅
    HTML；沒有問題時回傳空字串。純邏輯。

    分兩級：紅＝無依據（來源找不到支撐、守門規則判定捏造、整行漏檢、查核
    失敗、跳過檢查）；黃＝依據無法自動核實（模型拿文案自己的措辭當依據）。
    兩級都會出橫幅、都讓 main() 用非 0 結束碼離開，差別只在顏色、線條跟
    文字，讓人一眼分辨輕重。橫幅用行內樣式，不依賴樣板CSS。"""
    if skipped:
        return _wrap_banner(ALERT_RED, "⚠ 警告：這份文案未經忠實度檢查，不可直接對外使用", [
            _banner_section(ALERT_RED, "未經檢查",
                            ["生成時使用了 --skip-quality-check，賣點主張可能沒有依據。"]),
        ])
    if faithfulness["parse_error"] is not None:
        return _wrap_banner(ALERT_RED, "⚠ 警告：忠實度檢查失敗，無法確認賣點有依據，不可直接對外使用", [
            _banner_section(ALERT_RED, "檢查失敗", [html_lib.escape(faithfulness["parse_error"])]),
        ])
    levels = alert_levels(faithfulness)
    if not levels:
        return ""
    red_items = [
        f"<strong>{html_lib.escape(c['claim'])}</strong>"
        f"<br><small>{html_lib.escape(c['reason_unsupported'])}</small>"
        for c in faithfulness["claims"] if c.get("alert") == ALERT_RED
    ]
    red_items += [
        f"<strong>{html_lib.escape(line)}</strong>"
        "<br><small>檢查器整行漏掉這段內容，沒有判斷過有沒有依據</small>"
        for line in faithfulness["uncovered_lines"]
    ]
    if faithfulness["malformed_count"]:
        red_items.append(f"另有 {faithfulness['malformed_count']} 條查核結果格式損壞、無法顯示內容")
    yellow_items = [
        f"<strong>{html_lib.escape(c['claim'])}</strong>"
        f"<br><small>{html_lib.escape(c['reason_unsupported'])}</small>"
        for c in faithfulness["claims"] if c.get("alert") == ALERT_YELLOW
    ]
    sections = []
    if red_items:
        sections.append(_banner_section(
            ALERT_RED, f"紅色｜無依據（{len(red_items)}）——來源找不到支撐，或判定為捏造，必須修正", red_items))
    if yellow_items:
        sections.append(_banner_section(
            ALERT_YELLOW, f"黃色｜依據無法自動核實（{len(yellow_items)}）——模型引用的是文案自己的措辭，"
                          "需人工對照來源確認", yellow_items))
    if ALERT_RED in levels:
        headline = "⚠ 警告：這份文案含有沒有依據的內容，修正前不可對外使用"
        overall = ALERT_RED
    else:
        headline = "? 注意：部分主張的依據無法自動核實，人工對照來源確認前不可對外使用"
        overall = ALERT_YELLOW
    return _wrap_banner(overall, headline, sections)


def _wrap_banner(level: str, headline: str, sections) -> str:
    _, border, fg, _ = _BANNER_COLORS[level]
    return (
        f'<div style="border-bottom:4px solid {border};padding:12px 20px;margin:0;'
        f'font-family:sans-serif;color:{fg}">'
        f'<div style="font-size:18px;font-weight:700">{headline}</div>'
        + "".join(sections) + "</div>"
    )


TONE_GATED_MESSAGE = "因事實檢查未通過，語氣評分不具參考性"


def tone_gate_reason(faithfulness, skipped: bool):
    """語氣評分能不能顯示。只要忠實度有任何紅/黃項目、檢查失敗、或被跳過，
    就回傳不顯示的理由；可以顯示時回傳 None。純邏輯。

    由來：說服力試跑裡，數字全部捏造的那份文案拿到 19/20，比只用真實事實、
    刻意寫好的那份（16/20）還高——評分器照設計不管真假，會獎勵捏造的具體性。
    「語氣分數要搭配忠實度看」如果只寫在文件裡，遲早有人只看分數；所以做成
    工具本身的行為：事實檢查沒過，就根本不算、不顯示分數。"""
    if skipped:
        return f"{TONE_GATED_MESSAGE}（忠實度檢查被跳過）"
    if faithfulness is None or faithfulness["parse_error"] is not None:
        return f"{TONE_GATED_MESSAGE}（忠實度檢查失敗）"
    levels = alert_levels(faithfulness)
    if levels:
        names = "、".join(n for lv, n in ((ALERT_RED, "紅色"), (ALERT_YELLOW, "黃色")) if lv in levels)
        return f"{TONE_GATED_MESSAGE}（忠實度檢查有{names}警示）"
    return None


def format_tone_report(tone) -> list:
    """可以顯示語氣評分時的報告行（含總分）。純邏輯。"""
    if tone["parse_error"] is not None:
        return [f"[評分失敗] {tone['parse_error']}"]
    lines = [f"- {subitem}：{s['score']}/5 —— {s['rationale']}" for subitem, s in tone["scores"].items()]
    total = sum(s["score"] for s in tone["scores"].values())
    lines.append(f"合計：{total}/{5 * len(tone['scores'])}")
    return lines


def apply_warning_banner(html: str, banner: str) -> str:
    """把警示橫幅插在 <body> 之後的第一個位置。純邏輯。"""
    if not banner:
        return html
    if "<body>" in html:
        return html.replace("<body>", "<body>" + banner, 1)
    return banner + html


def render_marketing_plaintext(layout_name: str, content: dict) -> str:
    """把結構化內容轉成純文字版本，餵給忠實度/說服力檢查用——這兩個檢查
    判斷的是文案『內容』，HTML標籤只是排版噪音，不應該進到LLM的判斷輸入
    裡。純邏輯，不牽涉LLM。"""
    lines = [content["title"]]
    if layout_name == "重點條列型":
        lines.extend(content["bullets"])
    elif layout_name == "故事敘述型":
        lines.extend(content["story_paragraphs"])
    elif layout_name == "數據導向型":
        lines.extend(f"{s['number']} {s['label']}" for s in content["stats"])
        lines.extend(content["body_paragraphs"])
    else:
        raise ValueError(f"未知樣板：{layout_name}")
    lines.append(content["cta"])
    return "\n".join(lines)


def load_marketing_templates(path: str) -> dict:
    """讀 marketing_templates.json。模板名稱打錯字要在生成前就死掉，不是
    跑到套版階段才發現——跟 load_scoring_template() 的typo前置檢查同一個
    原則。"""
    if not os.path.isfile(path):
        die(f"找不到行銷樣板設定檔：{path}")
    with open(path, "r", encoding="utf-8") as f:
        templates = json.load(f)
    return templates


def main():
    ap = argparse.ArgumentParser(
        description="根據客戶文件生成行銷推廣文案(HTML)，附帶忠實度+說服力雙層品質把關。"
    )
    ap.add_argument("--collection", required=True,
                     help="客戶文件的Qdrant collection名稱"
                          "（用rag_ingest.py --doc-type structured事先ingest）")
    ap.add_argument("--brief", required=True,
                     help="這份文案要圍繞的主題/重點，同時當檢索查詢跟生成指令")
    ap.add_argument("--layout", required=True, help="樣板名稱（對應 --templates-file 裡的鍵）")
    ap.add_argument("--templates-file",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "data", "marketing_templates.json"),
                     help="行銷樣板設定檔路徑（預設 scripts/data/marketing_templates.json）")
    ap.add_argument("--top-k", type=int, default=8, help="從客戶文件collection檢索幾個chunk(預設8)")
    ap.add_argument("--style-collection", default=None,
                     help="風格參考庫的collection名稱，不指定則不使用風格參考——目前唯一"
                          "現成的內容是佔位測試資料（scripts/data/marketing_style_"
                          "reference_PLACEHOLDER.md），不是正式素材，預設不開啟")
    ap.add_argument("--style-top-k", type=int, default=3, help="從風格參考庫檢索幾個chunk(預設3)")
    ap.add_argument("--max-tokens", type=int, default=2048, help="生成上限 (預設 2048)")
    ap.add_argument("--temperature", type=float, default=0.3, help="生成溫度 (預設 0.3)")
    ap.add_argument("--output", required=True, help="輸出的HTML檔案路徑")
    ap.add_argument("--skip-quality-check", action="store_true", default=False,
                     help="跳過忠實度/說服力檢查（除錯用）。跳過時輸出的 HTML 頁首會帶"
                          "「未經忠實度檢查」警示橫幅，結束碼為 3")
    args = ap.parse_args()

    templates = load_marketing_templates(args.templates_file)
    if args.layout not in templates:
        die(f"樣板 '{args.layout}' 不存在。可用樣板：{sorted(templates)}")
    template = templates[args.layout]

    ensure_collection(args.collection, create=False)
    vector = embed_one(args.brief)
    hits = search(args.collection, vector, top_k=args.top_k)
    if not hits:
        die(f"collection '{args.collection}' 沒有回傳任何檢索結果，可能是空的。")
    context = build_context(hits)

    style_context = None
    if args.style_collection:
        # 佔位資料，正式素材待使用者提供：目前唯一現成的風格庫內容是
        # scripts/data/marketing_style_reference_PLACEHOLDER.md（Claude 自己寫
        # 的示範文案），只用來驗證這條檢索路徑走得通，不是正式業界範例。
        ensure_collection(args.style_collection, create=False)
        style_hits = search(args.style_collection, vector, top_k=args.style_top_k)
        if style_hits:
            style_context = build_context(style_hits)
        else:
            log(f"風格參考庫 '{args.style_collection}' 沒有檢索到內容，繼續生成但不附風格參考。")

    log("呼叫模型生成文案內容...")
    result = generate_marketing_copy_content(
        args.brief, args.layout, context, style_context,
        CHAT_URL, CHAT_MODEL, args.max_tokens, args.temperature,
    )
    if result["parse_error"] is not None:
        die(f"文案生成失敗（重試一次後仍失敗）：{result['parse_error']}")
    content = result["content"]
    plaintext = render_marketing_plaintext(args.layout, content)

    print("\n" + "=" * 78)
    print("文案全文（純文字）")
    print("=" * 78)
    print(plaintext)

    # 檢查一定要在寫出 HTML 之前跑完：結果要決定 HTML 裡有沒有警示橫幅跟
    # 紅框標記，不能先寫出一份看起來完全正常的文案、檢查結果只留在終端機。
    faithfulness = None
    if not args.skip_quality_check:
        log("跑忠實度檢查...")
        faithfulness = check_marketing_claims(plaintext, context, CHAT_URL, CHAT_MODEL)
        print_faithfulness_report(faithfulness)

    print("\n" + "=" * 78)
    print("說服力/語氣評分（機制可運作，評分本身的準確度尚未驗證，僅供參考）")
    print("=" * 78)
    gate = tone_gate_reason(faithfulness, skipped=args.skip_quality_check)
    if gate:
        # 事實檢查沒過就不呼叫評分器——不算，就不會有分數被單獨拿出來看
        print(gate)
    else:
        log("跑說服力/語氣評分...")
        for line in format_tone_report(score_marketing_tone(plaintext, CHAT_URL, CHAT_MODEL)):
            print(line)

    banner = build_warning_banner(faithfulness, skipped=args.skip_quality_check)
    html_out = apply_warning_banner(
        render_marketing_html(args.layout, template, content, flag_map(faithfulness)), banner,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_out)

    if banner:
        print(f"\n[WARNING] 文案已寫入 {args.output}，但頁首帶有警示橫幅——"
              "修正或人工確認前不可對外使用。", file=sys.stderr)
        sys.exit(EXIT_NEEDS_ATTENTION)
    log(f"文案已寫入 {args.output}（所有事實主張都找到依據）")


def print_faithfulness_report(faithfulness) -> None:
    print("\n" + "=" * 78)
    print("忠實度檢查（事實主張 vs. 來源文件；主觀修辭列出但不判斷依據）")
    print("=" * 78)
    if faithfulness["parse_error"] is not None:
        print(f"[抽取失敗] {faithfulness['parse_error']}")
        return
    if not faithfulness["claims"]:
        print("(沒有抽取到任何句子)")
    for c in faithfulness["claims"]:
        if c["kind"] == "rhetoric":
            print(f"- [修辭] {c['claim']}")
            continue
        mark = {None: "有依據", ALERT_RED: "紅｜無依據", ALERT_YELLOW: "黃｜待核實"}[c.get("alert")]
        detail = c["evidence"] if c["supported"] else c["reason_unsupported"]
        print(f"- [{mark}] {c['claim']}\n    {detail}")
    for line in faithfulness["uncovered_lines"]:
        print(f"- [紅｜未經檢查] {line}\n    檢查器整行漏掉，沒有判斷過有沒有依據")
    if faithfulness["malformed_count"]:
        print(f"- [紅｜格式損壞] 另有 {faithfulness['malformed_count']} 條查核結果無法顯示，需人工核對")


if __name__ == "__main__":
    main()
