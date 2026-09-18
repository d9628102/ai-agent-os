#!/usr/bin/env python3
"""
numeric_consistency.py

同一份被 ingest 文件內部的數字一致性檢查——簡報工廠往 DD 報告方向擴充的
第一步。範圍刻意縮小：只做「同一指標、不同問法」的多題答案之間的數值比
對，不做外部查證（web search/公開資料庫/競品比對）、不做完整 DD 報告模
板（九格評分、紅旗結構化輸出、情境式結論）。

跟 generate_report.py 的關係：這裡是純邏輯（抽取 prompt 組裝、JSON 解
析、數值比對），generate_report.py 負責在批次跑問答的迴圈裡呼叫這些函
式、組織成報告小節——純邏輯抽出來獨立測試，跟 filter_citations() 是同一
個理由。

分組不是由模型自動判斷「哪些問題算同一指標」——那本質是語意比對，跟
「外部查證」一樣是刻意排除的複雜度。分組由問題清單檔案裡人工標記（見
generate_report.py 的 `## group: <id> | metric: <名稱>` 語法），這裡只
在「同一組內」比對，不猜測哪些題目該分在一組。

抽取跟 QA judge 刻意用兩次獨立的 LLM 呼叫，不合併成一次——兩種結構化輸
出的 JSON schema 混在同一個 system prompt 裡容易讓模型混淆格式指令，
降低兩邊的可靠度。多一次呼叫的延遲換兩邊都更可靠，這次先接受這個代價。
"""
import json
import re

from rag_common import http_json

EXTRACT_SYSTEM_PROMPT = (
    "你是嚴謹的財務數字查核員。你會看到：一個指定的財務/數量指標名稱、使用"
    "者的問題、系統的回答、以及檢索到的文件片段全文。你的任務是：只針對"
    "『這個指定指標』，判斷回答裡有沒有給出具體數字。\n"
    "規則：\n"
    "1. 只在回答明確提到這個指標的具體數值時才算找到，不要用檢索片段裡其他"
    "不相關的數字硬湊；如果回答沒有明確給出這個指標的數字，found 填 false。\n"
    "2. 找到的話，把原始寫法（例如「$2M」「USD 1.5M」「1500萬」）跟正規化"
    "後的數值一起輸出——正規化數值一律換算成以 USD 為單位的數字；如果不是"
    "金額（例如百分比、件數），unit 欄位寫實際單位，value_normalized 填換"
    "算後的數字本身（例如 22.9% 填 22.9，單位寫 percent）；完全無法判斷單"
    "位換算基準時，value_normalized 填 null。\n"
    "3. 同時從回答或檢索片段全文裡找一段能直接證明這個數值的逐字引用（不"
    "超過 40 字，必須是原文照抄，不要自己改寫），填進 source_snippet；找"
    "不到就填 null。\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"found": true/false, "raw_value": "<字串或null>", '
    '"value_normalized": <數字或null>, "unit": "<字串或null>", '
    '"source_snippet": "<字串或null>"}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _not_found(parse_error, parsed=None):
    parsed = parsed or {}
    return {
        "found": False,
        "raw_value": parsed.get("raw_value"),
        "value_normalized": None,
        "unit": parsed.get("unit"),
        "source_snippet": parsed.get("source_snippet"),
        "parse_error": parse_error,
    }


def parse_extraction_response(raw: str) -> dict:
    """把模型輸出的原始文字（可能包在 <think> 或 markdown code fence 裡）
    解析、正規化成抽取結果的最終格式。從 extract_metric_value() 拆出來
    的純邏輯——不打真的模型也能測 found=false/value_normalized 缺漏/JSON
    壞掉這些邊界情況，跟 filter_citations() 被拆成獨立函式是同一個理由。
    """
    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        found = bool(parsed.get("found"))
        value_normalized = parsed.get("value_normalized")
        if value_normalized is not None:
            value_normalized = float(value_normalized)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _not_found(f"抽取結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    if not found or value_normalized is None:
        return _not_found(None, parsed)

    return {
        "found": True,
        "raw_value": parsed.get("raw_value"),
        "value_normalized": value_normalized,
        "unit": parsed.get("unit"),
        "source_snippet": parsed.get("source_snippet"),
        "parse_error": None,
    }


def extract_metric_value(metric, question, answer, context, chat_url, chat_model, timeout=180):
    """對指定的 metric，從這一題的回答（搭配檢索片段全文佐證）抽取具體數
    值。永遠回傳一個結構一致的 dict，不拋例外——找不到、呼叫失敗、JSON
    解析失敗都歸一成 found=False + parse_error，呼叫端不用個別 try/except。
    """
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"指定指標：{metric}\n\n問題：{question}\n\n系統回答：\n{answer}\n\n"
                    f"檢索到的文件片段全文：\n{context}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return _not_found(f"呼叫抽取模型失敗 (HTTP {status}): {resp}")
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return _not_found(f"抽取模型回應格式不如預期: {resp}")

    return parse_extraction_response(raw)


def locate_source_heading(source_snippet, hits):
    """用逐字子字串比對，把 source_snippet 定位回具體的檢索片段
    （heading_path）——刻意不用語意比對，找不到就老實說找不到，不是重
    要功能，只是輔助追溯用。"""
    if not source_snippet or not source_snippet.strip():
        return None
    snippet = source_snippet.strip()
    for hit in hits:
        payload = hit.get("payload", {})
        body = payload.get("body") or payload.get("text", "")
        if snippet in body:
            return payload.get("heading_path", "?")
    return None


def check_group_consistency(metric, entries):
    """entries: [{"question":..., "extraction": extract_metric_value() 的
    回傳值, "heading":..., "needs_review": bool}, ...]

    只比對 extraction["found"] 為真的項目：
    - 少於 2 個成功抽取 -> "insufficient"（資料不足以比對，不是衝突也不是一致）
    - >=2 個且數值全部相等（含極小浮點誤差） -> "consistent"
    - >=2 個但有任兩個不相等 -> "conflict"
    """
    found_entries = [e for e in entries if e["extraction"]["found"]]
    if len(found_entries) < 2:
        status = "insufficient"
    else:
        values = [e["extraction"]["value_normalized"] for e in found_entries]
        base = values[0]
        status = "consistent" if all(abs(v - base) < 1e-6 for v in values) else "conflict"
    return {"metric": metric, "status": status, "entries": entries}
