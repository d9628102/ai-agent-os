#!/usr/bin/env python3
"""
scoring.py

客製評分維度——簡報工廠往完整 DD 報告模板方向擴充的第四步（財務數字
一致性檢查、紅旗結構化輸出、跨文件比對之後）。技術路線跟前三塊一樣：
獨立的 LLM 呼叫、structured JSON 抽取、純邏輯拆函式方便測試。

這次的維度定義/權重/等級對照表不寫在這個檔案裡，寫在
scripts/data/scoring_templates.json——之後真的要加第二套模板（深科技
版）時，只要加一個模板物件，不用改這裡的邏輯。權重固定寫在模板設定檔，
不做每案客製；問題清單裡的 `## dimension: <名稱>` 標記只寫維度名稱，
不重複寫權重，避免兩個來源打架。

評分不是 LLM 憑空打分：每個維度評分時，把該維度底下所有問題已經產出的
答案、紅旗判斷結果（含嚴重度）、數字一致性檢查結果（如果衝突）都彙總
成一份 bundle 餵給評分呼叫，並要求輸出裡的 evidence 具體引用這些既有
結構化物件，不能是空泛判斷——審核時要能直接對照回紅旗清單/數字一致性
檢查章節裡的原始項目。

跟數字一致性檢查、紅旗判斷一樣，評分用獨立的 LLM 呼叫，不跟其他判斷
共用一次呼叫；開思考模式——評分要同時消化多題答案+多個紅旗+多個數字
衝突，是比紅旗判斷更複雜的多來源綜合判斷，關掉思考的捷徑風險更高。
"""
import json
import re

from rag_common import JSON_OUTPUT_REMINDER, http_json

_VALID_SCORES = (1, 2, 3, 4, 5)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _failed_score(parse_error):
    return {"score": None, "rationale": None, "evidence": None, "parse_error": parse_error}


SCORING_SYSTEM_PROMPT = (
    "你是嚴謹的投資評估分析師。你會看到一個評估維度的名稱，以及這個維度"
    "底下所有相關問題的：系統回答、被判定為紅旗的項目（含嚴重度）、數字"
    "一致性檢查的衝突結果（如果有）。你的任務是針對這個維度給 1-5 分的"
    "整數評分，5 分最正面（例如風險維度 5 分代表風險極低），1 分最負面。\n"
    "重要規則——不要把『這題沒有被標記成紅旗』誤解成『答案內容沒有負面"
    "事實』：紅旗判斷器只標記『主張跟查核結果對不上』或『數字跟數字對不上』"
    "這種矛盾結構，一個單純的負面事實（例如頻繁更名、零技術資產、營收"
    "衰退）如果沒有矛盾結構，就不會被標記成紅旗，但這不代表這件事對評分"
    "沒有影響。評分時必須同時考慮：(a) 紅旗跟數字衝突（如果有），(b) 答案"
    "原文裡任何具體的負面或正面陳述，即使沒有被紅旗判斷器標記。『沒有"
    "紅旗』只能證明『沒有偵測到矛盾結構』，不能當作『內容正面、應給高分』"
    "的理由——一個沒有矛盾但充滿負面事實的答案（例如『3年內更名3次』），"
    "分數應該反映那些負面事實本身，不是因為沒有紅旗就給高分。\n"
    "評分規則：\n"
    "1. 只能根據提供的內容評分，不能用你自己對這類案子的既有印象或常識"
    "去補充沒有提到的資訊\n"
    "2. evidence 陣列裡每一條都必須具體引用——紅旗的標題、數字一致性"
    "檢查裡衝突的兩個數字、或答案裡的原句摘要——不能寫「根據整體評估」"
    "這種空泛的話。如果提供的內容裡有紅旗或數字衝突，evidence 必須至少"
    "引用一個；如果完全沒有紅旗或衝突，必須引用答案原文裡具體的正面或"
    "負面陳述，不能因為『沒有紅旗』就跳過對答案內容本身的檢視\n"
    "3. rationale 是簡短說明（150字以內），講清楚為什麼給這個分數\n"
    f"{JSON_OUTPUT_REMINDER}\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"score": <1-5的整數>, "rationale": "<字串>", "evidence": ["<字串>", ...]}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)


def parse_scoring_response(raw: str) -> dict:
    """把模型輸出解析成評分結果。score 不是 1-5 的整數、或 evidence 是
    空清單/缺漏，都視為抽取失敗（is_red_flag=true 卻缺欄位是同一個處理
    原則）——不半殘顯示一個沒有依據的分數。"""
    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        score = int(parsed.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return _failed_score(f"評分結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    rationale = parsed.get("rationale")
    evidence = parsed.get("evidence")

    if (score not in _VALID_SCORES or not rationale
            or not isinstance(evidence, list) or not evidence):
        return _failed_score(
            f"評分結果欄位不完整或 score 不是 1-5 的合法值：{parsed}"
        )

    return {"score": score, "rationale": rationale, "evidence": evidence, "parse_error": None}


def build_dimension_bundle(dimension_entries, consistency_results) -> str:
    """dimension_entries: 這個維度底下每題的
    [{"question": str, "answer": str, "detection": dict|None, "group_id": str|None}, ...]
    detection 是這一題自己的紅旗判斷結果（None 表示沒開紅旗偵測）；
    group_id 是這一題所屬的數字一致性檢查群組（None 表示沒有）。

    consistency_results: check_group_consistency() 的回傳值清單。

    純邏輯，組出一份文字，餵給 score_dimension() 當 user content。同一個
    數字矛盾可能同時出現在多個維度的評分依據裡——這是預期內的，跟紅旗
    清單/數字一致性檢查章節本來就會在報告裡重疊出現同一個道理，不是
    要去重的錯誤。
    """
    consistency_by_group = {r["metric"]: r for r in consistency_results}
    seen_conflicts = set()

    blocks = []
    for entry in dimension_entries:
        lines = [f"問題：{entry['question']}", f"回答：{entry['answer']}"]

        detection = entry.get("detection")
        if detection and detection.get("is_red_flag"):
            lines.append(
                f"這題被判定為紅旗：{detection['title']}"
                f"（嚴重度：{detection['severity']}）——{detection['phenomenon']}"
            )

        group_id = entry.get("group_id")
        if group_id:
            for metric, result in consistency_by_group.items():
                if result["status"] != "conflict":
                    continue
                key = id(result)
                if key in seen_conflicts:
                    continue
                matched_questions = {e["question"] for e in result["entries"]}
                if entry["question"] not in matched_questions:
                    continue
                seen_conflicts.add(key)
                values = "；".join(
                    f"「{e['question']}」→ {e['extraction']['raw_value']}"
                    for e in result["entries"] if e["extraction"]["found"]
                )
                lines.append(f"這題涉及數字一致性檢查衝突（指標「{metric}」）：{values}")

        blocks.append("\n".join(lines))

    return "\n\n---\n\n".join(blocks)


def _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout):
    """單次呼叫評分模型，回傳 parse_scoring_response() 的結果。抽成獨立
    函式方便 score_dimension() 的重試邏輯直接重複呼叫同一段。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": SCORING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"評估維度：{dimension_name}\n\n{bundle}",
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return _failed_score(f"呼叫評分模型失敗 (HTTP {status}): {resp}")
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return _failed_score(f"評分模型回應格式不如預期: {resp}")

    return parse_scoring_response(raw)


def score_dimension(dimension_name, dimension_entries, consistency_results,
                     chat_url, chat_model, timeout=180):
    """對一個評估維度打分。永遠回傳結構一致的 dict，不拋例外。

    重試一次：JSON 解析失敗（模型生成的 JSON 語法有誤，不是內容判斷本身
    的問題）時，用同一個 bundle 再呼叫一次——診斷過的真實案例顯示這類
    失敗是模型生成 JSON 時偶發的格式失誤（例如陣列漏了結尾括號），不是
    每次都會重現；一次評分失敗會讓整份報告的總分算不出來，代價偏高，
    重試的成本很低，值得做。兩次都失敗才真的標記「評分失敗，需人工
    判斷」。"""
    bundle = build_dimension_bundle(dimension_entries, consistency_results)
    result = _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout)
    if result["parse_error"] is not None:
        result = _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout)
    return result


def compute_weighted_total(dimension_scores: dict, template: dict):
    """dimension_scores: {維度名稱: score(1-5)}，只放已成功評分的維度。
    回傳 (加權總分, 最大可能分數)。最大可能分數 = 9個維度 × 每維度上限
    （不是權重總和 × 上限）——這是兩份真實範例交叉核對出來的既有算法，
    刻意照實做，不自行「修正」。純邏輯，不牽涉 LLM。"""
    weights = template["dimensions"]
    max_per_dim = template["max_score_per_dimension"]
    total = sum(score * weights[dim] for dim, score in dimension_scores.items())
    max_total = len(weights) * max_per_dim
    return total, max_total


def lookup_grade(total: int, template: dict) -> dict:
    """依 grade_bands（由高到低的 min 門檻）找出對應等級。純邏輯，
    grade_bands 必須是 min 值遞減排列（模板設定檔本身的責任），這裡只
    找第一個 total >= min 的區間。action 是這個等級對應的建議動作（例如
    「可進入核心合作」）——九格模板的 grade_bands 每一項都有這個欄位；
    用 .get() 讀取是為了不強制要求舊模板一定要有這個欄位。"""
    for band in template["grade_bands"]:
        if total >= band["min"]:
            return {"grade": band["grade"], "label": band["label"], "action": band.get("action")}
    return {"grade": None, "label": "無法判定等級", "action": None}
