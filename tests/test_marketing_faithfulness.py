"""
測試重點：
- parse_faithfulness_response()：fact/rhetoric 分類、單條判斷不完整時
  fail-closed（保留下來當成無依據，不是丟掉）、連 claim 文字都沒有的項目
  計入 malformed_count、跟 fact 完全重複的 rhetoric 去重、整批解析失敗
- find_uncovered_lines()：檢查器整行漏掉的文案行要被找出來
- apply_deterministic_guards()：模型判完之後的五道機械檢查（A 修辭裡的
  最高級/規模/數字、B 數字、C 最高級用詞、D 避險用詞、E 捏造引文），
  每一道都各自驗證「該攔的攔到、不該攔的不攔」——這幾道是實測模型判太寬
  之後才加的，測試要能證明它們真的擋在模型判斷後面，不是裝飾
- load_first_json_object()：模型先輸出一份少了結尾 } 的裸 JSON、再輸出一份
  包在 code fence 裡的完整 JSON 時，要解析後面那份——實測說服力評分對一份
  滿是「%」的文案三次都這樣輸出
- needs_attention()：什麼情況下文案不能被當成「查核過、沒問題」
- check_marketing_claims()：被 max_tokens 截斷時要單獨回報，不偽裝成
  JSON 解析失敗（monkeypatch http_json，不打真的模型）
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import marketing_faithfulness
from marketing_faithfulness import (
    ALERT_RED,
    ALERT_YELLOW,
    INCOMPLETE_JUDGMENT_REASON,
    RECLASSIFIED_REASON,
    _cn_to_int,
    alert_levels,
    apply_deterministic_guards,
    check_marketing_claims,
    cited_segments,
    find_uncovered_lines,
    load_first_json_object,
    needs_attention,
    parse_faithfulness_response,
)


def _fact(claim, supported=True, evidence="文件片段1提到", reason=None, alert=None):
    return {"claim": claim, "kind": "fact", "supported": supported,
            "evidence": evidence if supported else None,
            "reason_unsupported": None if supported else (reason or "沒提到"),
            "alert": None if supported else (alert or ALERT_RED)}


def _rhetoric(claim):
    return {"claim": claim, "kind": "rhetoric", "supported": None,
            "evidence": None, "reason_unsupported": None, "alert": None}


# ---------------------------------------------------------------------------
# parse_faithfulness_response()
# ---------------------------------------------------------------------------

def test_parse_valid_fact_and_rhetoric():
    raw = json.dumps({"claims": [
        {"claim": "A", "kind": "fact", "supported": True, "evidence": "「A」", "reason_unsupported": None},
        {"claim": "B", "kind": "fact", "supported": False, "evidence": None, "reason_unsupported": "沒提到B"},
        {"claim": "打造新紀元", "kind": "rhetoric", "supported": None, "evidence": None, "reason_unsupported": None},
    ]})
    result = parse_faithfulness_response(raw)
    assert result["parse_error"] is None
    assert result["malformed_count"] == 0
    assert result["claims"] == [
        _fact("A", evidence="「A」"), _fact("B", supported=False, reason="沒提到B"), _rhetoric("打造新紀元"),
    ]


@pytest.mark.parametrize("raw,description", [
    ("not json at all", "純文字"),
    ('{"claims": [', "JSON語法不完整"),
    ('["claims"]', "頂層是清單不是物件——原本會拋 AttributeError"),
    ('{"claims": "not a list"}', "claims是字串"),
    ('{"claims": null}', "claims是null"),
    ('{"no_claims_key": []}', "根本沒有claims欄位"),
])
def test_parse_whole_batch_failure(raw, description):
    result = parse_faithfulness_response(raw)
    assert result["parse_error"] is not None, description
    assert result["claims"] is None, description


# 判斷不完整的 fact 不能消失：要保留下來、當成無依據——丟掉的話一條沒依據
# 的主張會直接從報告裡蒸發
@pytest.mark.parametrize("bad,description", [
    ({"claim": "A", "kind": "fact", "supported": True, "evidence": None}, "supported=true卻缺evidence"),
    ({"claim": "A", "kind": "fact", "supported": False, "reason_unsupported": None}, "supported=false卻缺理由"),
    ({"claim": "A", "kind": "fact", "supported": "yes", "evidence": "有"}, "supported不是bool"),
    ({"claim": "A", "kind": "fact", "evidence": "有"}, "缺supported"),
    ({"claim": "A", "kind": "unknown", "supported": True, "evidence": "有"}, "kind不是合法值"),
    ({"claim": "A", "supported": True, "evidence": "有"}, "缺kind"),
])
def test_parse_incomplete_judgment_fails_closed(bad, description):
    result = parse_faithfulness_response(json.dumps({"claims": [bad]}))
    assert result["parse_error"] is None, description
    assert result["claims"] == [
        {"claim": "A", "kind": "fact", "supported": False, "evidence": None,
         "reason_unsupported": INCOMPLETE_JUDGMENT_REASON, "alert": ALERT_RED}
    ], description


@pytest.mark.parametrize("bad", [
    "not a dict", {"kind": "fact", "supported": True, "evidence": "有"}, {"claim": "   "}, {"claim": 123},
])
def test_parse_item_without_claim_text_counted_as_malformed(bad):
    good = {"claim": "好", "kind": "fact", "supported": True, "evidence": "有"}
    result = parse_faithfulness_response(json.dumps({"claims": [bad, good]}))
    assert result["malformed_count"] == 1
    assert result["claims"] == [_fact("好", evidence="有")]


def test_parse_drops_rhetoric_that_duplicates_a_fact():
    raw = json.dumps({"claims": [
        {"claim": "深受百家企業信賴", "kind": "fact", "supported": False, "reason_unsupported": "沒提到"},
        {"claim": "深受百家企業信賴", "kind": "rhetoric"},
        {"claim": "迎向未來", "kind": "rhetoric"},
    ]})
    claims = parse_faithfulness_response(raw)["claims"]
    assert [c["kind"] for c in claims] == ["fact", "rhetoric"]
    assert claims[1]["claim"] == "迎向未來"


@pytest.mark.parametrize("raw", [
    '```json\n{"claims": []}\n```',
    '<think>先拆句</think>\n{"claims": []}',
])
def test_parse_strips_think_and_fence(raw):
    result = parse_faithfulness_response(raw)
    assert result["parse_error"] is None
    assert result["claims"] == []


# ---------------------------------------------------------------------------
# find_uncovered_lines()
# ---------------------------------------------------------------------------

def test_find_uncovered_lines_reports_line_with_no_claim():
    copy_text = "…的終極方案\n五大產品矩陣。\n\n立即預約"
    claims = [_fact("五大產品矩陣"), _rhetoric("立即預約")]
    assert find_uncovered_lines(copy_text, claims) == ["…的終極方案"]


def test_find_uncovered_lines_ignores_punctuation_and_quote_differences():
    copy_text = "以『五大產品矩陣』打造新紀元。"
    claims = [_fact("以「五大產品矩陣」打造新紀元")]
    assert find_uncovered_lines(copy_text, claims) == []


def test_find_uncovered_lines_partial_claim_counts_as_covered():
    copy_text = "深受超過百家企業信賴，一起迎向智慧未來。"
    assert find_uncovered_lines(copy_text, [_fact("深受超過百家企業信賴", supported=False)]) == []


def test_find_uncovered_lines_paraphrased_claim_is_reported():
    # 改寫過的 claim 比對不到 → 當成漏檢，故意偏向誤報
    assert find_uncovered_lines("提供五大產品矩陣", [_fact("有五個產品矩陣")]) == ["提供五大產品矩陣"]


# ---------------------------------------------------------------------------
# apply_deterministic_guards()
# ---------------------------------------------------------------------------

SOURCE = (
    "[文件片段 1] 標題\n五大產品矩陣依序為 Decision Matrix、Knowledge Matrix。"
    "AI 可以每天完成數萬個決策。認證：ISO 27001。建立全球企業智慧的新基礎建設。最重要的底層。"
)


@pytest.mark.parametrize("claim", [
    "…解決知識流失的終極方案", "業界領先的平台", "深受超過百家企業信賴", "七成企業都在用", "10年經驗",
    "上百種可能，一次滿足", "數以萬計的企業選擇",
])
def test_guard_a_risky_rhetoric_becomes_unsupported_fact(claim):
    [c] = apply_deterministic_guards([_rhetoric(claim)], SOURCE)
    assert c["kind"] == "fact" and c["supported"] is False
    assert c["reason_unsupported"] == RECLASSIFIED_REASON


@pytest.mark.parametrize("claim", [
    "打造企業智慧新紀元", "讓每一個決策都充滿溫度", "千萬別錯過", "萬無一失的守護", "一起迎向智慧未來",
    "讓每一家企業都能建構屬於自己的智慧基礎建設", "數位轉型的新篇章", "以數據驅動未來",
])
def test_guard_a_leaves_pure_rhetoric_alone(claim):
    assert apply_deterministic_guards([_rhetoric(claim)], SOURCE) == [_rhetoric(claim)]


def test_guard_b_number_missing_from_source_downgrades():
    [c] = apply_deterministic_guards([_fact("導入後效率提升 3 倍")], SOURCE)
    assert c["supported"] is False and "['3']" in c["reason_unsupported"]


def test_guard_b_ignores_fragment_label_numbers():
    # SOURCE 裡只有「[文件片段 1]」這個標籤帶數字 1——不能讓捏造的 1 因此過關
    [c] = apply_deterministic_guards([_fact("排名第 1 的平台", evidence="有")], SOURCE)
    assert c["supported"] is False


def test_guard_b_whole_number_match_not_substring():
    [c] = apply_deterministic_guards([_fact("符合 2700 項規範")], SOURCE)
    assert c["supported"] is False  # 2700 不能因為來源有 27001 就過關


@pytest.mark.parametrize("claim", ["5 大產品矩陣", "通過 ISO 27001 認證"])
def test_guard_b_number_present_in_source_passes(claim):
    # 「5」靠來源裡的中文數字「五大」對應過關
    assert apply_deterministic_guards([_fact(claim)], SOURCE) == [_fact(claim)]


@pytest.mark.parametrize("token,value", [
    ("五", 5), ("八", 8), ("十", 10), ("十五", 15), ("二十", 20), ("兩", 2), ("九十九", 99),
    ("一百", None), ("十十", None),
])
def test_cn_to_int(token, value):
    assert _cn_to_int(token) == value


def test_guard_c_superlative_missing_from_source_downgrades():
    [c] = apply_deterministic_guards([_fact("最完整的決策平台")], SOURCE)
    assert c["supported"] is False and "最完" in c["reason_unsupported"]


def test_guard_c_superlative_present_in_source_passes():
    claim = "建立全球企業智慧的新基礎建設"
    assert apply_deterministic_guards([_fact(claim)], SOURCE) == [_fact(claim)]


def test_guard_c_bare_zui_needs_following_char_match():
    # 來源有「最重要」，不能讓「最強」靠單一個「最」字過關
    [c] = apply_deterministic_guards([_fact("最強的平台")], SOURCE)
    assert c["supported"] is False


@pytest.mark.parametrize("evidence", ["文件2間接對應", "可推論出這點", "兩者形成間接對應"])
def test_guard_d_hedged_evidence_downgrades(evidence):
    [c] = apply_deterministic_guards([_fact("提供決策功能", evidence=evidence)], SOURCE)
    assert c["supported"] is False


def test_guard_e_fabricated_quote_downgrades():
    [c] = apply_deterministic_guards(
        [_fact("無需自建系統", evidence="文件片段7：無需自建複雜系統")], SOURCE)
    assert c["supported"] is False and "無需自建複雜系統" in c["reason_unsupported"]


@pytest.mark.parametrize("evidence", [
    "文件片段1：AI 可以每天完成數萬個決策",
    "文件片段1提到「五大產品矩陣依序為 Decision Matrix」",
    "文件1的 Decision Matrix 提供決策功能",
])
def test_guard_e_real_quote_or_narrative_passes(evidence):
    c = _fact("每天完成數萬個決策", evidence=evidence)
    assert apply_deterministic_guards([c], SOURCE) == [c]


def test_cited_segments_extracts_quotes_and_label_citations():
    # 引號內至少兩個字才算引文（避免把零星的引號當成引用）
    ev = "文件片段1：甲甲，對應文案；文件片段2提到「乙乙」；文件3的描述「丙」"
    assert cited_segments(ev) == ["乙乙", "甲甲"]


def test_guards_leave_unsupported_facts_untouched():
    c = _fact("三秒完成清算", supported=False, reason="沒提到")
    assert apply_deterministic_guards([c], SOURCE) == [c]


# ---------------------------------------------------------------------------
# needs_attention()
# ---------------------------------------------------------------------------

def _result(claims=(), malformed=0, uncovered=(), parse_error=None):
    return {"claims": list(claims), "malformed_count": malformed,
            "uncovered_lines": list(uncovered), "parse_error": parse_error}


@pytest.mark.parametrize("result,expected", [
    (_result([_fact("A"), _rhetoric("B")]), False),
    (_result([_fact("A", supported=False)]), True),
    (_result([_fact("A")], malformed=1), True),
    (_result([_fact("A")], uncovered=["標題"]), True),
    ({"claims": None, "malformed_count": 0, "uncovered_lines": [], "parse_error": "壞了"}, True),
])
def test_needs_attention(result, expected):
    assert needs_attention(result) is expected


# ---------------------------------------------------------------------------
# check_marketing_claims()：截斷與覆蓋率接線
# ---------------------------------------------------------------------------

def _fake_http(content, finish_reason="stop"):
    def fake(method, url, payload=None, timeout=300):
        return 200, {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    return fake


def test_check_reports_truncation_explicitly(monkeypatch):
    monkeypatch.setattr(marketing_faithfulness, "http_json", _fake_http("<think>還沒想完", "length"))
    result = check_marketing_claims("文案", SOURCE, "url", "model")
    assert result["claims"] is None
    assert "截斷" in result["parse_error"]


def test_check_runs_guards_and_coverage(monkeypatch):
    raw = json.dumps({"claims": [
        {"claim": "效率提升 3 倍", "kind": "fact", "supported": True, "evidence": "有"},
    ]})
    monkeypatch.setattr(marketing_faithfulness, "http_json", _fake_http(raw))
    result = check_marketing_claims("效率提升 3 倍\n終極方案", SOURCE, "url", "model")
    assert result["claims"][0]["supported"] is False  # 經過 guard B
    assert result["uncovered_lines"] == ["終極方案"]


# ---------------------------------------------------------------------------
# load_first_json_object()
# ---------------------------------------------------------------------------

def test_load_first_json_object_handles_bare_then_fenced_duplicate():
    obj = '{"scores": {"a": 1}}'
    raw = f"<think>想一下</think>\n{obj}\n\n```json\n{obj}\n```"
    assert load_first_json_object(raw) == {"scores": {"a": 1}}


def test_load_first_json_object_prefers_valid_fenced_block_over_broken_bare_copy():
    # 實測形狀：裸的那份少了最外層 }，fence 裡那份才完整
    broken = '{"scores": {"a": {"score": 4}}'
    good = '{"scores": {"a": {"score": 5}}}'
    raw = f"<think>評分</think>\n{broken}\n\n```json\n{good}\n```"
    assert load_first_json_object(raw) == {"scores": {"a": {"score": 5}}}


def test_load_first_json_object_bare_object_followed_by_prose():
    assert load_first_json_object('{"a": 1}\n\n以上是評分結果，供參考。') == {"a": 1}


def test_load_first_json_object_plain_and_fenced():
    assert load_first_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert load_first_json_object('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize("raw", ["完全不是 JSON", '{"a": 1', ""])
def test_load_first_json_object_still_raises_on_garbage(raw):
    with pytest.raises(ValueError):
        load_first_json_object(raw)


def test_parse_accepts_duplicated_json_output():
    obj = json.dumps({"claims": [{"claim": "A", "kind": "rhetoric"}]})
    result = parse_faithfulness_response(f"{obj}\n\n```json\n{obj}\n```")
    assert result["parse_error"] is None
    assert result["claims"] == [_rhetoric("A")]


# guard E 的兩種誤報：「...」省略、把清單裡不相鄰的項目串成一句
@pytest.mark.parametrize("evidence", [
    "文件片段1：「五大產品矩陣依序為...Knowledge Matrix」",
    "「Decision Matrix、ISO 27001」",
    "「標題五大產品矩陣依序為 Decision Matrix。AI 可以每天完成數萬個決策」",
    "「認證：ISO 27001。建立全球企業智慧的新基礎建設」",
])
def test_guard_e_ellipsis_and_stitched_items_pass(evidence):
    c = _fact("有依據的主張", evidence=evidence)
    assert apply_deterministic_guards([c], SOURCE) == [c]


def test_guard_e_fabricated_piece_inside_real_quote_still_caught():
    c = _fact("有依據的主張", evidence="「Decision Matrix、全年無休客服中心」")
    [g] = apply_deterministic_guards([c], SOURCE)
    assert g["supported"] is False and "全年無休客服中心" in g["reason_unsupported"]




# guard B：模糊量詞（數萬、數十萬、上千…）是需要來源的數字主張
def test_guard_b_vague_quantity_present_in_source_passes():
    # 實測案例：來源寫「AI 可以每天完成數萬個決策」，文案的「數萬次」有依據
    c = _fact("數萬次 每日AI決策處理量", evidence="「AI 可以每天完成數萬個決策」")
    assert apply_deterministic_guards([c], SOURCE) == [c]


@pytest.mark.parametrize("claim,token", [
    ("每日處理數十萬次決策", "數十萬"), ("已有上千家企業導入", "上千"), ("累積近百年經驗", "近百"),
])
def test_guard_b_vague_quantity_missing_from_source_downgrades(claim, token):
    [c] = apply_deterministic_guards([_fact(claim, evidence="有")], SOURCE)
    assert c["supported"] is False and c["alert"] == ALERT_RED and token in c["reason_unsupported"]


# guard E：兩級——捏造引文紅、循環引用（引的是文案自己的措辭）黃
def test_guard_e_circular_quote_is_yellow():
    claim = "整合五大產品矩陣，打造企業智慧底座"
    [c] = apply_deterministic_guards([_fact(claim, evidence="「整合五大產品矩陣」")], SOURCE)
    assert c["supported"] is False and c["alert"] == ALERT_YELLOW
    assert "循環引用" in c["reason_unsupported"]


def test_guard_e_quote_from_elsewhere_in_copy_is_yellow():
    copy_text = "整合五大產品矩陣\n永續累積與共享"
    [c] = apply_deterministic_guards(
        [_fact("整合五大產品矩陣", evidence="「永續累積與共享」")], SOURCE, copy_text)
    assert c["alert"] == ALERT_YELLOW


def test_guard_e_fabricated_plus_circular_is_red():
    claim = "整合五大產品矩陣"
    [c] = apply_deterministic_guards(
        [_fact(claim, evidence="「整合五大產品矩陣、無需自建複雜系統」")], SOURCE)
    assert c["alert"] == ALERT_RED and "無需自建複雜系統" in c["reason_unsupported"]


# alert_levels()
@pytest.mark.parametrize("result,expected", [
    ({"claims": [_fact("A"), _rhetoric("B")], "malformed_count": 0, "uncovered_lines": [], "parse_error": None}, set()),
    ({"claims": [_fact("A", supported=False, alert=ALERT_YELLOW)], "malformed_count": 0,
      "uncovered_lines": [], "parse_error": None}, {ALERT_YELLOW}),
    ({"claims": [_fact("A", supported=False), _fact("B", supported=False, alert=ALERT_YELLOW)],
      "malformed_count": 0, "uncovered_lines": [], "parse_error": None}, {ALERT_RED, ALERT_YELLOW}),
    ({"claims": [_fact("A")], "malformed_count": 0, "uncovered_lines": ["x"], "parse_error": None}, {ALERT_RED}),
    ({"claims": [_fact("A")], "malformed_count": 1, "uncovered_lines": [], "parse_error": None}, {ALERT_RED}),
    ({"claims": None, "malformed_count": 0, "uncovered_lines": [], "parse_error": "壞"}, {ALERT_RED}),
])
def test_alert_levels(result, expected):
    assert alert_levels(result) == expected
