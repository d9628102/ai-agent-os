#!/usr/bin/env python3
"""
gate_check.py

深科技評分模板的 Gate 1／Gate 2 條件查核。依據《Branes.AI 商業模式暨
投資盡職調查報告》第玖章「投資前必要前置條件（Gate 查核清單）」：

- Gate 1（具名客戶的 LOI 正本）成立 → 市場驗證評分可由 C 上調至 B+
- Gate 2（Domain Flow 的實體驗證）成立 → 技術差異化評分可由 B+ 上調至 A−

條件、調升、不成立時的處理都逐字寫在 scripts/data/scoring_templates.json
的深科技模板 "gates" 裡，這個檔案只放判斷與彙總邏輯。Gate 3（智財移轉
鏈不完整即不可投資）不在這裡，沿用 veto_check.py 的 veto 機制。

兩個刻意的設計：

1. 三種結果，不是兩種。met／not_met 之外還有 undetermined——文件裡沒有
   查核結果時不能當成「不成立」。Branes.AI 原文的 Gate 本身就是「投資前
   要去取證的清單」，還沒取證是常態，把「沒看到」當「不成立」會讓報告
   替還沒發生的查核下結論。

2. 只顯示、不自動改等級。條件成立時報告顯示「可由 X 調至 Y」並附上
   目前的系統評分，實際要不要調升由人工決定；維度等級永遠以評分呼叫的
   結果為準（見 render_gate_line()）。

跟 veto 一樣每題單獨呼叫一次 LLM（保留「哪句回答、哪個檢索片段」的
可追溯性），開思考模式。
"""
import json
import re

from rag_common import JSON_OUTPUT_REMINDER, http_json

GATE_MET = "met"
GATE_NOT_MET = "not_met"
GATE_UNDETERMINED = "undetermined"
GATE_CONFLICT = "conflict"
_VALID_STATUSES = (GATE_MET, GATE_NOT_MET, GATE_UNDETERMINED)

_JSON_SCHEMA_BLOCK = (
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"status": "met" 或 "not_met" 或 "undetermined", '
    '"basis": "<字串或null>", "quote": "<字串或null>"}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)


def build_gate_system_prompt(gate: dict) -> str:
    return (
        "你是嚴謹的盡職調查分析師。你會看到一則問答的問題、系統回答，以及"
        "檢索到的文件片段全文。你的任務是判斷這一題的內容，能不能確認"
        f"投資前必要條件「{gate['id']}——{gate['title']}」的條件是否成立。\n"
        f"條件：{gate['condition']}。\n"
        f"原文判準（出自範例案子，公司名稱以本案為準）：「{gate['criterion_verbatim']}」\n"
        "判斷規則：\n"
        "1. status 填 met：回答或片段明確記載『條件已經被查證成立』的事實"
        "（例如已取得具名對方的正本、已有實測數據且達到門檻），並且你能"
        "引用那句話。\n"
        "2. status 填 not_met：回答或片段明確記載『條件目前不成立』的事實"
        "（例如對方仍為匿名、數字只來自模擬而非實測、實測倍數未達門檻），"
        "並且你能引用那句話。\n"
        "3. status 填 undetermined：文件沒有記載查核結果、只列出『需要去"
        "查核／需要取得』的要求，或內容不足以判斷。沒看到成立的證據不等於"
        "不成立——只有明確記載不成立的事實才能填 not_met。\n"
        "4. 模擬、預估、宣稱、計畫中的數字都不算實測；只有明確寫出已經在"
        "實體硬體上量測的結果才算。\n"
        "5. 只能根據提供的內容判斷，不能用你自己對這家公司或這個產業的既有"
        "印象補充。\n"
        "met 或 not_met 時：basis 用一兩句話說明為什麼，quote 逐字引用回答"
        "或片段裡支撐判斷的那句話（不能寫「根據整體內容」這種空泛的話）。"
        "undetermined 時 basis 說明缺了什麼，quote 填 null。\n"
        f"{JSON_OUTPUT_REMINDER}\n"
        f"{_JSON_SCHEMA_BLOCK}"
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _undetermined(parse_error, basis=None):
    return {"status": GATE_UNDETERMINED, "basis": basis, "quote": None, "parse_error": parse_error}


def parse_gate_response(raw: str) -> dict:
    """把模型輸出解析成 Gate 判斷結果。解析失敗、status 不合法、或 met／
    not_met 卻缺 basis/quote，一律歸入 undetermined 並記下 parse_error
    （fail-closed：不半殘顯示一個沒有引用依據的成立/不成立）。
    undetermined 本身是正常判斷，parse_error 為 None。"""
    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _undetermined(f"Gate 判斷結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")
    if not isinstance(parsed, dict):
        return _undetermined(f"Gate 判斷結果不是 JSON 物件：{parsed}")

    status = parsed.get("status")
    if status not in _VALID_STATUSES:
        return _undetermined(f"Gate 判斷結果的 status 不合法：{parsed}")
    basis = parsed.get("basis")
    quote = parsed.get("quote")
    if status == GATE_UNDETERMINED:
        return _undetermined(None, basis=basis or None)
    if not basis or not quote:
        return _undetermined(f"status={status} 但 basis/quote 不完整：{parsed}")
    return {"status": status, "basis": basis, "quote": quote, "parse_error": None}


def detect_gate(gate, question, answer, context, chat_url, chat_model, timeout=180):
    """對一題的回答＋檢索片段全文，判斷指定 Gate 的條件是否成立。永遠回傳
    結構一致的 dict（含 "gate" 欄位＝Gate id），不拋例外。gate 是模板
    "gates" 裡的一個物件——呼叫端（generate_report.py）負責在跑報告前
    驗證問題清單裡的 Gate 名稱合法。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": build_gate_system_prompt(gate)},
            {
                "role": "user",
                "content": (
                    f"問題：{question}\n\n系統回答：\n{answer}\n\n"
                    f"檢索到的文件片段全文：\n{context}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        result = _undetermined(f"呼叫 Gate 判斷模型失敗 (HTTP {status}): {resp}")
    else:
        try:
            raw = resp["choices"][0]["message"].get("content") or ""
            result = parse_gate_response(raw)
        except (KeyError, IndexError, TypeError):
            result = _undetermined(f"Gate 判斷模型回應格式不如預期: {resp}")
    result["gate"] = gate["id"]
    return result


def aggregate_gate_status(statuses) -> str:
    """同一個 Gate 被多題各自檢查時的彙總（純邏輯）：
    - 同時有 met 跟 not_met → conflict（結果互相矛盾，交給人工，不自動挑一個）
    - 否則有 met → met；有 not_met → not_met
    - 全部 undetermined（或沒有任何結果）→ undetermined
    """
    statuses = list(statuses)
    has_met = GATE_MET in statuses
    has_not_met = GATE_NOT_MET in statuses
    if has_met and has_not_met:
        return GATE_CONFLICT
    if has_met:
        return GATE_MET
    if has_not_met:
        return GATE_NOT_MET
    return GATE_UNDETERMINED


def summarize_gate_results(gate_entries, gates):
    """gate_entries: [{"question": str, "gate": str, "detection": dict,
    "heading": str|None}, ...]，detection 是 detect_gate() 的回傳值。
    gates: 模板的 "gates" 清單。

    依模板 gates 的順序，只保留實際有被標記檢查過的 Gate；回傳
    {gate_id: {"status": 彙總結果, "entries": [每題的判斷, ...]}}。
    entries 全部保留（不只成立的那些），undetermined 跟 conflict 的時候
    人工複核也需要看到每一題各自怎麼判。純邏輯，不牽涉 LLM。"""
    by_gate = {}
    for entry in gate_entries:
        by_gate.setdefault(entry["gate"], []).append(entry)

    results = {}
    for gate in gates:
        gid = gate["id"]
        if gid not in by_gate:
            continue
        entries = [{
            "question": e["question"],
            "heading": e["heading"],
            "status": e["detection"]["status"],
            "basis": e["detection"]["basis"],
            "quote": e["detection"]["quote"],
            "parse_error": e["detection"]["parse_error"],
        } for e in by_gate[gid]]
        results[gid] = {
            "status": aggregate_gate_status(e["status"] for e in entries),
            "entries": entries,
        }
    return results


def render_gate_line(gate, status, current_grade, template) -> str:
    """評分總表裡接在對應維度後面的那一行 Gate 顯示。純邏輯組字。

    current_grade 是這個維度目前的系統評分（字母；評分失敗或沒評分時是
    None）。只顯示、不自動調整——met 時如果目前等級已經等於或高於調升
    目標，就顯示「已達調升目標，無需調整」。"""
    # 延遲 import：scoring 只在需要比較字母高低時用到，避免兩個模組互相
    # 依賴成為載入順序問題。
    from scoring import grade_rank, is_at_or_above

    gid = gate["id"]
    head = f"{gid}（{gate['title']}）"
    if status == GATE_MET:
        if_met = gate["if_met"]
        line = f"✅ {head}條件成立 → {if_met['text']}。"
        if current_grade is None:
            return line + "目前系統評分：無（評分失敗或未評分），由人工決定。"
        grade_rank(current_grade, template)  # 不合法等級在這裡丟錯，不默默比較
        if is_at_or_above(current_grade, if_met["to"], template):
            return line + f"目前系統評分：{current_grade}，已達調升目標，無需調整。"
        return line + f"目前系統評分：{current_grade}（未自動調整，由人工決定）。"
    if status == GATE_NOT_MET:
        return f"❌ {head}條件不成立 → {gate['if_not_met']}"
    if status == GATE_CONFLICT:
        return f"⚠️ {head}各題判斷結果互相矛盾（有成立也有不成立），需人工判斷。"
    return f"⏸ {head}尚未查核／文件未提供查核結果。"
