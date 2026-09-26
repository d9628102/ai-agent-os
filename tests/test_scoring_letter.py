"""
測試重點（scoring.py 的字母等級部分，深科技模板）：
- normalize_grade()：各種橫線統一成原文用的 U+2212（A−、B−），其他輸入原樣
- parse_letter_scoring_response()：只接受 grade_scale 裡的字母與
  below_scale_label；D、F、A+、數字分數一律視為解析失敗（fail-closed）；
  evidence／rationale 缺漏失敗；配對 think 標籤與 code fence 剝殼
- grade_rank()／is_at_or_above()：只比較清單位置，不合法等級丟錯
- compute_weighted_total()／lookup_grade()：字母模板直接丟 ValueError，
  數字模板（九格）行為不變——規格核准的「不把字母硬轉成數字」
- build_letter_scoring_prompt()：只帶維度的中性定義與「evidence 必須逐字
  引用」規則；範例案子（Branes.AI）的原文說明與等級一律不進 prompt
  （第一版放了原文說明，驗證發現模型把範例字句當成本案證據）
- parse_letter_scoring_response()：完整物件後面多一個結尾括號（驗證時
  估值合理性三次都是 `{...}}`）要能解析；物件本身不完整仍然失敗
- unbacked_evidence()：evidence 的字句必須在本案 bundle 找得到（概念同
  行銷文案的 Guard E）
- score_dimension()：字母模板走字母 prompt；evidence 找不到時判定失敗並
  重試一次，兩次都找不到才失敗；數字模板不做這道檢查。用固定回應替換
  http_json，不呼叫真正的模型
- validate_scoring_template()：真實模板通過；每一種設定錯誤都要被擋下

模板一律讀 scripts/data/scoring_templates.json 的真實內容，壞設定用
copy.deepcopy 修改真實模板製造，不自己虛構一套字母清單。
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import scoring
from scoring import (
    build_letter_scoring_prompt,
    compute_weighted_total,
    grade_rank,
    is_at_or_above,
    lookup_grade,
    normalize_grade,
    parse_letter_scoring_response,
    unbacked_evidence,
    validate_scoring_template,
)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    _TEMPLATES = json.load(_f)
DEEP = _TEMPLATES["深科技"]
NINE = _TEMPLATES["九格"]
MINUS = "−"  # U+2212，原文用的減號
BELOW = DEEP["below_scale_label"]


def _deep():
    return copy.deepcopy(DEEP)


# normalize_grade()：各種橫線都統一成 U+2212
@pytest.mark.parametrize("raw", ["A-", "A–", "A－", "A‐", " A- ", "A" + MINUS])
def test_normalize_grade_minus_variants(raw):
    assert normalize_grade(raw) == "A" + MINUS


# normalize_grade()：不是「字母＋橫線」形狀的輸入原樣回傳（去空白除外）
@pytest.mark.parametrize("raw,expected", [
    ("B+", "B+"), ("A", "A"), (" C ", "C"), ("A--", "A--"), (BELOW, BELOW), (None, None), (3, 3),
])
def test_normalize_grade_other_inputs_untouched(raw, expected):
    assert normalize_grade(raw) == expected


# parse_letter_scoring_response()：grade_scale 裡的每一個字母、以及 below_scale_label 都合法
@pytest.mark.parametrize("grade", DEEP["grade_scale"] + [BELOW])
def test_parse_letter_accepts_every_legal_grade(grade):
    raw = json.dumps({"grade": grade, "rationale": "r", "evidence": ["e"]}, ensure_ascii=False)
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["parse_error"] is None
    assert result["grade"] == grade
    assert result["evidence"] == ["e"]


# parse_letter_scoring_response()：ASCII 連字號先正規化成 U+2212 再比對
def test_parse_letter_normalizes_ascii_hyphen():
    result = parse_letter_scoring_response('{"grade": "B-", "rationale": "r", "evidence": ["e"]}', DEEP)
    assert result["parse_error"] is None
    assert result["grade"] == "B" + MINUS


# parse_letter_scoring_response()：原文沒有定義的等級、數字分數、空字串一律失敗
@pytest.mark.parametrize("grade", ["D", "F", "A+", "C-", 3, "", None, "B++"])
def test_parse_letter_rejects_illegal_grade(grade):
    raw = json.dumps({"grade": grade, "rationale": "r", "evidence": ["e"]}, ensure_ascii=False)
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["grade"] is None
    assert result["parse_error"] is not None


# parse_letter_scoring_response()：rationale／evidence 缺漏或 evidence 為空都失敗
@pytest.mark.parametrize("payload", [
    {"grade": "B", "evidence": ["e"]},
    {"grade": "B", "rationale": "", "evidence": ["e"]},
    {"grade": "B", "rationale": "r"},
    {"grade": "B", "rationale": "r", "evidence": []},
    {"grade": "B", "rationale": "r", "evidence": "e"},
])
def test_parse_letter_rejects_incomplete_fields(payload):
    result = parse_letter_scoring_response(json.dumps(payload, ensure_ascii=False), DEEP)
    assert result["grade"] is None
    assert result["parse_error"] is not None


# parse_letter_scoring_response()：JSON 壞掉或不是物件都失敗，不拋例外
@pytest.mark.parametrize("raw", ["not json", '["B"]', '"B"', '{"grade": "B"'])
def test_parse_letter_rejects_non_object(raw):
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["grade"] is None
    assert result["parse_error"] is not None


# parse_letter_scoring_response()：配對的 think 標籤與 json code fence 要能剝殼
@pytest.mark.parametrize("raw", [
    '<think>先看紅旗</think>\n{"grade": "C+", "rationale": "r", "evidence": ["e"]}',
    '```json\n{"grade": "C+", "rationale": "r", "evidence": ["e"]}\n```',
])
def test_parse_letter_strips_think_and_fence(raw):
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["parse_error"] is None
    assert result["grade"] == "C+"


# grade_rank()：清單位置，A 最高，below_scale_label 排在全部字母之後
@pytest.mark.parametrize("grade,expected", [
    ("A", 0), ("A" + MINUS, 1), ("B+", 2), ("C", 6), (BELOW, 7),
])
def test_grade_rank_positions(grade, expected):
    assert grade_rank(grade, DEEP) == expected


# grade_rank()：不合法等級丟 ValueError，不默默回傳一個位置
@pytest.mark.parametrize("grade", ["D", "A-", "", None])
def test_grade_rank_illegal_raises(grade):
    with pytest.raises(ValueError):
        grade_rank(grade, DEEP)


# is_at_or_above()：等於目標、高於目標為 True，低於為 False
@pytest.mark.parametrize("grade,target,expected", [
    ("B+", "B+", True), ("A", "B+", True), ("C", "B+", False), (BELOW, "C", False), ("C", BELOW, True),
])
def test_is_at_or_above(grade, target, expected):
    assert is_at_or_above(grade, target, DEEP) is expected


# compute_weighted_total()／lookup_grade()：字母模板直接丟錯，不回傳任何數字
def test_weighted_total_rejects_letter_template():
    with pytest.raises(ValueError, match="letter 模板"):
        compute_weighted_total({"團隊": 3}, DEEP)


def test_lookup_grade_rejects_letter_template():
    with pytest.raises(ValueError, match="letter 模板"):
        lookup_grade(20, DEEP)


# compute_weighted_total()／lookup_grade()：九格模板行為不變（九個維度各 3 分 → 27/45 → C）
def test_numeric_template_unchanged():
    scores = {dim: 3 for dim in NINE["dimensions"]}
    total, max_total = compute_weighted_total(scores, NINE)
    assert (total, max_total) == (27, 45)
    assert lookup_grade(total, NINE)["grade"] == "C"


# build_letter_scoring_prompt()：每個維度都帶自己的中性定義、合法等級清單、below_scale_label
# 與逐字引用規則；七個維度的原文說明（validation_reference）一句都不能出現，
# 也不能出現「參考範例」「評為」這類把範例結論帶進來的字樣
@pytest.mark.parametrize("dimension", list(DEEP["dimensions"]))
def test_letter_prompt_uses_definition_only(dimension):
    prompt = build_letter_scoring_prompt(DEEP, dimension)
    assert DEEP["dimensions"][dimension]["definition"] in prompt
    for ref in DEEP["validation_reference"].values():
        assert ref["criterion"] not in prompt
    assert "參考範例" not in prompt
    assert "評為" not in prompt
    assert "逐字複製" in prompt
    for grade in DEEP["grade_scale"]:
        assert grade in prompt
    assert BELOW in prompt


# parse_letter_scoring_response()：完整物件後面多餘的結尾括號不影響解析（驗證時的真實輸出形狀）
@pytest.mark.parametrize("raw", [
    '{"grade": "C+", "rationale": "r", "evidence": ["e"]}}',
    '<think>推理</think>\n\n{"grade": "C+", "rationale": "r", "evidence": ["e"]}}',
])
def test_parse_letter_tolerates_trailing_brace(raw):
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["parse_error"] is None
    assert result["grade"] == "C+"


# parse_letter_scoring_response()：物件本身不完整（少了結尾括號）仍然失敗
@pytest.mark.parametrize("raw", [
    '{"grade": "C+", "rationale": "r", "evidence": ["e"]',
    '{"grade": "C+", "rationale": "r", "evidence": ["e"',
])
def test_parse_letter_incomplete_object_still_fails(raw):
    result = parse_letter_scoring_response(raw, DEEP)
    assert result["grade"] is None
    assert result["parse_error"] is not None


_BUNDLE = (
    "問題：USD 27M pre-money 對應的資產狀態為何？\n"
    "回答：晶片未流片、客戶未具名、營收為零。Deck 引用 Carta 2026 年 4 月的 deep-tech seed 數據。\n"
    "這題被判定為紅旗：比較基準選擇偏誤（嚴重度：high）——未納入同功耗區間的專用 NPU"
)


# unbacked_evidence()：逐字引用（含只取一段、用頓號串接）都找得到
@pytest.mark.parametrize("evidence", [
    ["晶片未流片、客戶未具名、營收為零"],
    ["「晶片未流片」", "營收為零"],
    ["這題被判定為紅旗：比較基準選擇偏誤（嚴重度：high）"],
    ["Deck 引用 Carta 2026 年 4 月的 deep-tech seed 數據"],
])
def test_unbacked_evidence_all_found(evidence):
    assert unbacked_evidence(evidence, _BUNDLE) == []


# unbacked_evidence()：bundle 或 evidence 帶 Markdown 粗體 `**` 時，逐字引用仍然找得到
# （驗證時技術差異化三次都因為這個被誤擋：bundle 是 `為 **預測值（…）**`）
def test_unbacked_evidence_ignores_markdown_bold():
    bundle = "回答：- **100× 能源效率**：為 **預測值（projected over 2 stages）**，文件未說明其具體驗證方式"
    assert unbacked_evidence(["為 預測值（projected over 2 stages）"], bundle) == []
    assert unbacked_evidence(["**100× 能源效率**：為 **預測值"], bundle) == []
    # 去掉 `*` 不會讓改寫過關
    assert unbacked_evidence(["為預估值"], bundle)


# unbacked_evidence()：只出現在範例案子結論裡的字句（驗證時真實發生的洩漏）要被抓出來
def test_unbacked_evidence_catches_leaked_reference_wording():
    problems = unbacked_evidence(["晶片未流片", "Carta 對比屬選擇性引用", "USD 27M pre 對「有專利未流片」偏高"], _BUNDLE)
    assert [item for item, _ in problems] == ["Carta 對比屬選擇性引用", "USD 27M pre 對「有專利未流片」偏高"]
    assert "Carta對比屬選擇性引用" in [m.replace(" ", "") for m in problems[0][1]]


# unbacked_evidence()：改寫、翻成簡體都算找不到
@pytest.mark.parametrize("evidence", [["芯片未流片"], ["營收等於零"], ["客户未具名"]])
def test_unbacked_evidence_rejects_paraphrase_and_simplified(evidence):
    assert unbacked_evidence(evidence, _BUNDLE)


# validate_scoring_template()：真實的深科技、九格模板都沒有錯誤
@pytest.mark.parametrize("name", ["深科技", "九格"])
def test_validate_real_templates_pass(name):
    assert validate_scoring_template(_TEMPLATES[name]) == []


def _break(mutator):
    t = _deep()
    mutator(t)
    return t


def _set_gate_to(t):
    t["gates"][0]["if_met"]["to"] = "D"


# validate_scoring_template()：每一種壞設定都要被擋下，錯誤訊息點名問題所在
@pytest.mark.parametrize("mutator,keyword", [
    (lambda t: t.__setitem__("grade_bands", []), "grade_bands"),
    (lambda t: t.__setitem__("max_score_per_dimension", 5), "max_score_per_dimension"),
    (lambda t: t["dimensions"].__setitem__("團隊", 1), "團隊"),
    (lambda t: t.__setitem__("grade_scale", ["A", "A", "B"]), "重複"),
    (lambda t: t.__setitem__("below_scale_label", "C"), "below_scale_label"),
    (lambda t: t.__setitem__("overall", "weighted"), "overall"),
    (lambda t: t.pop("display_name"), "display_name"),
    (lambda t: t["gates"][0].__setitem__("dimension", "不存在的維度"), "Gate 1"),
    (_set_gate_to, "if_met.to"),
    (lambda t: t["veto_rules"]["智財歸屬不完整"].pop("consequence"), "consequence"),
    (lambda t: t.pop("veto_label"), "veto_label"),
    (lambda t: t["dimensions"]["團隊"].pop("definition"), "definition"),
    (lambda t: t["dimensions"]["團隊"].__setitem__("definition", "  "), "definition"),
    (lambda t: t["dimensions"]["團隊"].__setitem__("reference_criterion", "範例結論"), "reference_criterion"),
    (lambda t: t["dimensions"]["團隊"].__setitem__("reference_grade", "A"), "reference_grade"),
    (lambda t: t["validation_reference"]["團隊"].__setitem__("grade", "D"), "validation_reference"),
    (lambda t: t["validation_reference"].__setitem__("不存在的維度", {"grade": "A", "criterion": "x"}), "validation_reference"),
    (lambda t: t.__setitem__("scale_type", "percent"), "scale_type"),
])
def test_validate_rejects_broken_letter_template(mutator, keyword):
    errors = validate_scoring_template(_break(mutator))
    assert errors, "壞設定應該回傳至少一條錯誤"
    assert any(keyword in e for e in errors), errors


# validate_scoring_template()：數字模板的權重不是數字、dimensions 為空都要報錯
def test_validate_rejects_broken_numeric_template():
    t = copy.deepcopy(NINE)
    t["dimensions"]["信用"] = "1"
    assert any("信用" in e for e in validate_scoring_template(t))
    t = copy.deepcopy(NINE)
    t["dimensions"] = {}
    assert validate_scoring_template(t)


def _fake_http(contents, calls):
    """依序回傳 contents 裡的模型輸出，並記下每次送出的 payload。"""
    def fake(method, url, payload, timeout=None):
        calls.append(payload)
        return 200, {"choices": [{"message": {"content": contents[len(calls) - 1]}}]}
    return fake


_ENTRIES = [{"question": "USD 27M pre-money 對應的資產狀態為何？",
             "answer": "晶片未流片、客戶未具名、營收為零。", "detection": None, "group_id": None}]


# score_dimension()：字母模板、evidence 逐字找得到 → 採用等級，只呼叫一次，送出的是字母 prompt
def test_score_dimension_letter_evidence_found(monkeypatch):
    calls = []
    monkeypatch.setattr(scoring, "http_json", _fake_http(
        ['{"grade": "C", "rationale": "r", "evidence": ["晶片未流片、客戶未具名"]}'], calls))
    result = scoring.score_dimension("估值合理性", _ENTRIES, [], "u", "m", template=DEEP)
    assert result["grade"] == "C"
    assert result["parse_error"] is None
    assert len(calls) == 1
    assert DEEP["dimensions"]["估值合理性"]["definition"] in calls[0]["messages"][0]["content"]


# score_dimension()：evidence 引用本案沒有的字句 → 這次不採用，重試一次；重試找得到就採用重試結果
def test_score_dimension_letter_leaked_evidence_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(scoring, "http_json", _fake_http([
        '{"grade": "C+", "rationale": "r", "evidence": ["Carta 對比屬選擇性引用"]}',
        '{"grade": "C", "rationale": "r", "evidence": ["營收為零"]}',
    ], calls))
    result = scoring.score_dimension("估值合理性", _ENTRIES, [], "u", "m", template=DEEP)
    assert len(calls) == 2
    assert result["grade"] == "C"


# score_dimension()：兩次 evidence 都找不到 → 失敗，錯誤訊息點名找不到的片段與不採用的等級
def test_score_dimension_letter_leaked_evidence_twice_fails(monkeypatch):
    calls = []
    leaked = '{"grade": "C+", "rationale": "r", "evidence": ["Carta 對比屬選擇性引用"]}'
    monkeypatch.setattr(scoring, "http_json", _fake_http([leaked, leaked], calls))
    result = scoring.score_dimension("估值合理性", _ENTRIES, [], "u", "m", template=DEEP)
    assert result["grade"] is None
    assert "選擇性引用" in result["parse_error"]
    assert "C+" in result["parse_error"]


# score_dimension()：數字模板（九格）不做 evidence 逐字檢查，行為跟原本一樣
def test_score_dimension_numeric_skips_evidence_check(monkeypatch):
    calls = []
    monkeypatch.setattr(scoring, "http_json", _fake_http(
        ['{"score": 2, "rationale": "r", "evidence": ["本案內容裡沒有的摘要"]}'], calls))
    result = scoring.score_dimension("風險", _ENTRIES, [], "u", "m", template=NINE)
    assert result["score"] == 2
    assert calls[0]["messages"][0]["content"] == scoring.SCORING_SYSTEM_PROMPT
