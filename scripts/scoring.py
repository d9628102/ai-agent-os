#!/usr/bin/env python3
"""
scoring.py

客製評分維度——簡報工廠往完整 DD 報告模板方向擴充的第四步（財務數字
一致性檢查、紅旗結構化輸出、跨文件比對之後）。技術路線跟前三塊一樣：
獨立的 LLM 呼叫、structured JSON 抽取、純邏輯拆函式方便測試。

這次的維度定義/權重/等級對照表不寫在這個檔案裡，寫在
scripts/data/scoring_templates.json。第二套模板（深科技）加進來時發現
「只加一個模板物件」不夠：深科技用的是字母等級、沒有權重，跟九格的
數字加權是兩種評分制度，所以模板多了 scale_type 欄位（見下方
SCALE_NUMERIC/SCALE_LETTER），letter 模板走自己的 prompt/解析，不接
加權總分那條路。權重固定寫在模板設定檔，
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

from marketing_faithfulness import _norm_for_quote, load_first_json_object, unbacked_quote_pieces
from rag_common import JSON_OUTPUT_REMINDER, http_json

_VALID_SCORES = (1, 2, 3, 4, 5)

# 模板兩種評分制度：numeric（九格：1-5分＋權重＋分數區間對照等級）跟
# letter（深科技：直接給字母等級，不訂權重、不算綜合等級）。舊模板沒寫
# scale_type 一律視為 numeric，舊報告/舊測試行為完全不變。字母等級只做
# 「合不合法」跟「清單位置比高低」兩件事，刻意不提供任何字母轉分數的
# 函式——只有一份深科技範例，換算成數字等於憑空製造精確度。
SCALE_NUMERIC = "numeric"
SCALE_LETTER = "letter"

# 原文用的是 U+2212 MINUS SIGN（A−、B−），模型常輸出 ASCII 連字號或其他
# 橫線，解析時統一換成 U+2212 再比對 grade_scale，不因為字元長得像就判
# 成非法等級。只換字母後面的那一個符號，不動其他文字。
_MINUS_VARIANTS = ("-", "‐", "‑", "‒", "–", "—", "－")
_MINUS_SIGN = "−"


def _failed_score(parse_error):
    return {"score": None, "rationale": None, "evidence": None, "parse_error": parse_error}


_NO_RED_FLAG_IS_NOT_POSITIVE_RULE = (
    "重要規則——不要把『這題沒有被標記成紅旗』誤解成『答案內容沒有負面"
    "事實』：紅旗判斷器只標記『主張跟查核結果對不上』或『數字跟數字對不上』"
    "這種矛盾結構，一個單純的負面事實（例如頻繁更名、零技術資產、營收"
    "衰退）如果沒有矛盾結構，就不會被標記成紅旗，但這不代表這件事對評分"
    "沒有影響。評分時必須同時考慮：(a) 紅旗跟數字衝突（如果有），(b) 答案"
    "原文裡任何具體的負面或正面陳述，即使沒有被紅旗判斷器標記。『沒有"
    "紅旗』只能證明『沒有偵測到矛盾結構』，不能當作『內容正面、應給高分』"
    "的理由——一個沒有矛盾但充滿負面事實的答案（例如『3年內更名3次』），"
    "分數應該反映那些負面事實本身，不是因為沒有紅旗就給高分。\n"
)

_EVIDENCE_RULES = (
    "1. 只能根據提供的內容評分，不能用你自己對這類案子的既有印象或常識"
    "去補充沒有提到的資訊\n"
    "2. evidence 陣列裡每一條都必須具體引用——紅旗的標題、數字一致性"
    "檢查裡衝突的兩個數字、或答案裡的原句摘要——不能寫「根據整體評估」"
    "這種空泛的話。如果提供的內容裡有紅旗或數字衝突，evidence 必須至少"
    "引用一個；如果完全沒有紅旗或衝突，必須引用答案原文裡具體的正面或"
    "負面陳述，不能因為『沒有紅旗』就跳過對答案內容本身的檢視\n"
)

SCORING_SYSTEM_PROMPT = (
    "你是嚴謹的投資評估分析師。你會看到一個評估維度的名稱，以及這個維度"
    "底下所有相關問題的：系統回答、被判定為紅旗的項目（含嚴重度）、數字"
    "一致性檢查的衝突結果（如果有）。你的任務是針對這個維度給 1-5 分的"
    "整數評分，5 分最正面（例如風險維度 5 分代表風險極低），1 分最負面。\n"
    f"{_NO_RED_FLAG_IS_NOT_POSITIVE_RULE}"
    "評分規則：\n"
    f"{_EVIDENCE_RULES}"
    "3. rationale 是簡短說明（150字以內），講清楚為什麼給這個分數\n"
    f"{JSON_OUTPUT_REMINDER}\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"score": <1-5的整數>, "rationale": "<字串>", "evidence": ["<字串>", ...]}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)


def scale_type(template: dict) -> str:
    """沒寫 scale_type 的舊模板一律視為 numeric。"""
    return template.get("scale_type", SCALE_NUMERIC)


def is_letter_template(template: dict) -> bool:
    return scale_type(template) == SCALE_LETTER


_VERBATIM_EVIDENCE_RULE = (
    "4. evidence 的每一條都必須是從上面提供的內容（回答、紅旗、數字衝突）"
    "逐字複製的字句：可以只取其中一段，但不能改寫、不能摘要、不能翻譯成"
    "簡體字、不能把不同地方的文字拼成一句。系統會用程式逐字比對，任何一條"
    "在提供的內容裡找不到，這次評分就判定失敗\n"
)


def build_letter_scoring_prompt(template: dict, dimension_name: str) -> str:
    """字母等級版的評分 system prompt。跟數字版共用「沒有紅旗 ≠ 內容正面」
    跟 evidence 必須具體引用這兩段規則（同一個常數，不是複製一份），差別
    在輸出是字母、多一句維度的中性定義，以及 evidence 必須逐字引用（會被
    unbacked_evidence() 機械檢查）。

    刻意不放任何範例案子的結論或等級：第一版曾把 Branes.AI 在各維度的原文
    說明當「參考範例」放進 prompt，驗證發現模型會把範例裡的字句當成本案
    證據引用（估值合理性的 evidence 出現本案答案裡根本沒有的「Carta 對比
    屬選擇性引用」），拿它評其他案子會往 Branes.AI 的結論靠攏。原文等級與
    說明只留在模板的 validation_reference 當驗證對照，不進 prompt。"""
    scale = template["grade_scale"]
    below = template["below_scale_label"]
    definition = template["dimensions"][dimension_name]["definition"]
    return (
        "你是嚴謹的投資評估分析師。你會看到一個評估維度的名稱，以及這個維度"
        "底下所有相關問題的：系統回答、被判定為紅旗的項目（含嚴重度）、數字"
        "一致性檢查的衝突結果（如果有）。你的任務是針對這個維度給一個字母"
        f"等級。合法等級由高到低只有：{'、'.join(scale)}；如果比 {scale[-1]} "
        f"還差，填「{below}」。不能使用這份清單以外的等級（例如 D、F、A+），"
        "也不能輸出數字分數。\n"
        f"這個維度看的是：{definition}\n"
        f"{_NO_RED_FLAG_IS_NOT_POSITIVE_RULE}"
        "評分規則：\n"
        f"{_EVIDENCE_RULES}"
        "3. rationale 是簡短說明（150字以內），講清楚為什麼給這個等級\n"
        f"{_VERBATIM_EVIDENCE_RULE}"
        f"{JSON_OUTPUT_REMINDER}\n"
        "請只輸出一個 JSON 物件，格式為：\n"
        '{"grade": "<合法等級之一>", "rationale": "<字串>", "evidence": ["<字串>", ...]}\n'
        "不要有任何其他文字、不要用 markdown code fence。"
    )


def normalize_grade(raw_grade) -> str:
    """去掉前後空白，把字母後面的各種橫線統一成原文用的 U+2212。只處理
    「單一字母＋一個符號」這種形狀，其他字串原樣回傳（交給呼叫端判斷
    合不合法），不猜測模型的意思。"""
    if not isinstance(raw_grade, str):
        return raw_grade
    grade = raw_grade.strip()
    if len(grade) == 2 and grade[1] in _MINUS_VARIANTS:
        grade = grade[0] + _MINUS_SIGN
    return grade


def _failed_letter_grade(parse_error):
    return {"grade": None, "rationale": None, "evidence": None, "parse_error": parse_error}


def parse_letter_scoring_response(raw: str, template: dict) -> dict:
    """把模型輸出解析成字母等級結果。grade 不在 grade_scale 裡、也不是
    below_scale_label，或 evidence 是空清單/缺漏，都視為抽取失敗（跟數字
    版 score 不是 1-5 同一個 fail-closed 原則），不半殘顯示一個沒有依據
    或不合法的等級。

    JSON 用 marketing_faithfulness.load_first_json_object() 取出第一個完整
    物件：驗證時估值合理性三次都輸出 `{...}}`（完整物件後面多一個結尾
    括號），整段 json.loads 會失敗。物件本身不完整的照樣解析失敗。"""
    try:
        parsed = load_first_json_object(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _failed_letter_grade(f"評分結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")
    if not isinstance(parsed, dict):
        return _failed_letter_grade(f"評分結果不是 JSON 物件：{parsed}")

    grade = normalize_grade(parsed.get("grade"))
    legal = list(template["grade_scale"]) + [template["below_scale_label"]]
    rationale = parsed.get("rationale")
    evidence = parsed.get("evidence")

    if (grade not in legal or not rationale
            or not isinstance(evidence, list) or not evidence):
        return _failed_letter_grade(
            f"評分結果欄位不完整或 grade 不是合法等級（合法：{'、'.join(legal)}）：{parsed}"
        )

    return {"grade": grade, "rationale": rationale, "evidence": evidence, "parse_error": None}


def grade_rank(grade: str, template: dict) -> int:
    """字母等級在 grade_scale 裡的位置，0 最高；below_scale_label 排在全部
    合法字母之後。只拿來比較高低，不是分數——不能拿來加總或平均。不合法
    的等級丟 ValueError，不默默回傳一個位置。"""
    scale = template["grade_scale"]
    if grade in scale:
        return scale.index(grade)
    if grade == template["below_scale_label"]:
        return len(scale)
    raise ValueError(f"不合法的字母等級：{grade!r}（合法：{scale} 或 {template['below_scale_label']!r}）")


def is_at_or_above(grade: str, target: str, template: dict) -> bool:
    """grade 是否等於或高於 target（依 grade_scale 位置）。"""
    return grade_rank(grade, template) <= grade_rank(target, template)


def validate_scoring_template(template: dict) -> list:
    """回傳模板設定錯誤的清單（空清單代表沒問題）。純邏輯，main() 讀完
    模板後呼叫、有錯就 die()——設定錯誤在跑報告前就擋下，不要跑完一輪
    LLM 才在評分或渲染階段炸掉。

    letter 模板刻意不准出現 grade_bands、max_score_per_dimension、數字
    權重：這些是 numeric 模板的東西，出現在 letter 模板裡代表有人想把
    字母硬接回加權總分那條路。"""
    errors = []
    kind = scale_type(template)
    dims = template.get("dimensions")
    if not isinstance(dims, dict) or not dims:
        return [f"dimensions 必須是非空的物件（維度名稱 → 設定），目前是：{dims!r}"]

    if kind == SCALE_NUMERIC:
        for name, weight in dims.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                errors.append(f"numeric 模板的維度「{name}」權重必須是數字，目前是：{weight!r}")
        if not isinstance(template.get("max_score_per_dimension"), int):
            errors.append("numeric 模板缺少整數 max_score_per_dimension")
        if not isinstance(template.get("grade_bands"), list) or not template.get("grade_bands"):
            errors.append("numeric 模板缺少非空的 grade_bands")
        return errors

    if kind != SCALE_LETTER:
        return [f"不認得的 scale_type：{kind!r}（只支援 {SCALE_NUMERIC!r}、{SCALE_LETTER!r}）"]

    for forbidden in ("grade_bands", "max_score_per_dimension"):
        if forbidden in template:
            errors.append(f"letter 模板不能有 {forbidden}（字母等級不換算分數、不對照分數區間）")
    scale = template.get("grade_scale")
    if not isinstance(scale, list) or not scale or not all(isinstance(g, str) and g for g in scale):
        errors.append("letter 模板缺少非空的 grade_scale（由高到低的字母清單）")
        scale = []
    elif len(set(scale)) != len(scale):
        errors.append(f"grade_scale 有重複等級：{scale}")
    below = template.get("below_scale_label")
    if not isinstance(below, str) or not below:
        errors.append("letter 模板缺少 below_scale_label")
    elif below in scale:
        errors.append(f"below_scale_label 不能跟 grade_scale 裡的等級重複：{below!r}")
    if not template.get("display_name"):
        errors.append("letter 模板缺少 display_name（報告顯示「由人工判斷」時用的模板名稱）")
    if template.get("overall") != "manual":
        errors.append("letter 模板的 overall 必須是 \"manual\"（不計算綜合等級）")

    for name, cfg in dims.items():
        if not isinstance(cfg, dict):
            errors.append(f"letter 模板的維度「{name}」設定必須是物件，目前是：{cfg!r}"
                          "（數字代表權重，letter 模板不訂權重）")
            continue
        if not isinstance(cfg.get("definition"), str) or not cfg.get("definition").strip():
            errors.append(f"維度「{name}」缺少 definition（中性定義：這個維度看什麼）")
        for leaked in ("reference_criterion", "reference_grade"):
            if leaked in cfg:
                errors.append(f"維度「{name}」不能有 {leaked}：範例案子的結論不能放在會進 prompt 的"
                              "維度設定裡，只能放在 validation_reference")

    for name, ref in (template.get("validation_reference") or {}).items():
        if name not in dims:
            errors.append(f"validation_reference 的維度「{name}」不在模板維度裡")
        elif ref.get("grade") not in scale:
            errors.append(f"validation_reference「{name}」的等級 {ref.get('grade')!r} 不在 grade_scale 裡")

    seen_gate_ids = set()
    for gate in template.get("gates", []):
        gid = gate.get("id")
        if not gid or gid in seen_gate_ids:
            errors.append(f"Gate id 缺漏或重複：{gid!r}")
        seen_gate_ids.add(gid)
        if gate.get("dimension") not in dims:
            errors.append(f"{gid} 對應的維度 {gate.get('dimension')!r} 不在模板維度裡")
        for key in ("condition", "if_not_met"):
            if not gate.get(key):
                errors.append(f"{gid} 缺少 {key}")
        if_met = gate.get("if_met") or {}
        for key in ("from", "to"):
            if if_met.get(key) not in scale:
                errors.append(f"{gid} 的 if_met.{key} {if_met.get(key)!r} 不在 grade_scale 裡")
        if not if_met.get("text"):
            errors.append(f"{gid} 缺少 if_met.text")

    for rule_name, rule in (template.get("veto_rules") or {}).items():
        if rule.get("mode") not in ("矛盾", "缺失"):
            errors.append(f"veto 規則「{rule_name}」的 mode 必須是「矛盾」或「缺失」")
        for key in ("description", "source_label", "consequence"):
            if not rule.get(key):
                errors.append(f"veto 規則「{rule_name}」缺少 {key}")
    if template.get("veto_rules") and not template.get("veto_label"):
        errors.append("模板有 veto_rules 就必須有 veto_label（報告開頭與章節標題用）")
    return errors


def parse_scoring_response(raw: str) -> dict:
    """把模型輸出解析成評分結果。score 不是 1-5 的整數、或 evidence 是
    空清單/缺漏，都視為抽取失敗（is_red_flag=true 卻缺欄位是同一個處理
    原則）——不半殘顯示一個沒有依據的分數。

    JSON 用 load_first_json_object() 取出第一個完整物件（完整物件後面多餘
    的結尾括號不再讓整份評分失敗，見 parse_letter_scoring_response()）。
    解析出來不是物件（例如陣列）時 .get() 會丟 AttributeError，一併視為
    解析失敗——改用共用函式前，這種輸入會讓整支程式直接崩潰。"""
    try:
        parsed = load_first_json_object(raw)
        score = int(parsed.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
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


def unbacked_evidence(evidence, bundle: str) -> list:
    """字母模板的 evidence 機械檢查（概念同行銷文案的 Guard E）：每一條
    evidence 在標點處拆成片段，逐段比對本案 bundle；回傳
    [(evidence 原文, [找不到的片段, ...]), ...]，空清單代表全部找得到。
    拆片段沿用 marketing_faithfulness.unbacked_quote_pieces()：模型用省略
    或把清單項目串在一起引用時不會被誤判，但整個不存在的詞組（例如只出現
    在 prompt 範例裡的「選擇性引用」）照樣抓得到。純邏輯。"""
    normalized = _norm_for_quote(bundle)
    problems = []
    for item in evidence:
        missing = unbacked_quote_pieces(str(item), normalized)
        if missing:
            problems.append((str(item), missing))
    return problems


def _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout, template=None):
    """單次呼叫評分模型，回傳 parse_scoring_response()（numeric）或
    parse_letter_scoring_response()（letter）的結果。抽成獨立函式方便
    score_dimension() 的重試邏輯直接重複呼叫同一段。template 是 None 或
    numeric 模板時，送出的 prompt 跟加入 letter 模板之前逐字相同。"""
    letter = template is not None and is_letter_template(template)
    system_prompt = (build_letter_scoring_prompt(template, dimension_name)
                     if letter else SCORING_SYSTEM_PROMPT)
    fail = _failed_letter_grade if letter else _failed_score
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        return fail(f"呼叫評分模型失敗 (HTTP {status}): {resp}")
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return fail(f"評分模型回應格式不如預期: {resp}")

    if letter:
        result = parse_letter_scoring_response(raw, template)
        if result["parse_error"] is None:
            problems = unbacked_evidence(result["evidence"], bundle)
            if problems:
                detail = "；".join(f"「{item}」缺：{'、'.join(missing)}" for item, missing in problems)
                return _failed_letter_grade(
                    f"evidence 引用的字句在本案內容裡找不到（等級 {result['grade']} 不採用）：{detail}"
                )
        return result
    return parse_scoring_response(raw)


def score_dimension(dimension_name, dimension_entries, consistency_results,
                     chat_url, chat_model, timeout=180, template=None):
    """對一個評估維度打分。永遠回傳結構一致的 dict，不拋例外。

    重試一次：JSON 解析失敗（模型生成的 JSON 語法有誤，不是內容判斷本身
    的問題）時，用同一個 bundle 再呼叫一次——診斷過的真實案例顯示這類
    失敗是模型生成 JSON 時偶發的格式失誤（例如陣列漏了結尾括號），不是
    每次都會重現；一次評分失敗會讓整份報告的總分算不出來，代價偏高，
    重試的成本很低，值得做。兩次都失敗才真的標記「評分失敗，需人工
    判斷」。

    template 是 letter 模板時回傳 {"grade", "rationale", "evidence",
    "parse_error"}（沒有 "score"）；None 或 numeric 模板時照舊回傳
    {"score", ...}。"""
    bundle = build_dimension_bundle(dimension_entries, consistency_results)
    result = _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout, template)
    if result["parse_error"] is not None:
        result = _call_scoring_model(dimension_name, bundle, chat_url, chat_model, timeout, template)
    return result


def _reject_letter_template(template: dict, func_name: str):
    """字母模板不能走加權總分/分數區間這條路——直接丟錯，不默默回傳一個
    數字（規格核准的設計：不為了套用既有架構把字母硬轉成數字）。"""
    if is_letter_template(template):
        raise ValueError(
            f"{func_name}() 只適用 numeric 模板；letter 模板（字母等級）不計算"
            "加權總分、不對照分數區間，綜合等級由人工判斷"
        )


def compute_weighted_total(dimension_scores: dict, template: dict):
    """dimension_scores: {維度名稱: score(1-5)}，只放已成功評分的維度。
    回傳 (加權總分, 最大可能分數)。最大可能分數 = 9個維度 × 每維度上限
    （不是權重總和 × 上限）——這是兩份真實範例交叉核對出來的既有算法，
    刻意照實做，不自行「修正」。純邏輯，不牽涉 LLM。letter 模板丟
    ValueError。"""
    _reject_letter_template(template, "compute_weighted_total")
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
    用 .get() 讀取是為了不強制要求舊模板一定要有這個欄位。letter 模板
    丟 ValueError。"""
    _reject_letter_template(template, "lookup_grade")
    for band in template["grade_bands"]:
        if total >= band["min"]:
            return {"grade": band["grade"], "label": band["label"], "action": band.get("action")}
    return {"grade": None, "label": "無法判定等級", "action": None}
