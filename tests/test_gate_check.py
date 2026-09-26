"""
測試重點（gate_check.py，深科技模板的 Gate 1／Gate 2）：
- parse_gate_response()：三種結果 met／not_met／undetermined；met／not_met
  缺 basis 或 quote 一律歸入 undetermined 並記下 parse_error（fail-closed）；
  undetermined 本身是正常判斷；配對 think 標籤與 code fence 剝殼
- aggregate_gate_status()：同一個 Gate 多題判斷的彙總，有成立也有不成立
  時是 conflict，不自動挑一個
- summarize_gate_results()：依模板 gates 順序、只含實際檢查過的 Gate、
  entries 全部保留
- render_gate_line()：只顯示、不自動改等級；目前等級已達調升目標時說明
  無需調整；不成立時逐字顯示原文的 if_not_met
- build_gate_system_prompt()：帶入原文條件與判準、三種狀態

Gate 設定一律讀 scripts/data/scoring_templates.json 的深科技真實模板。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from gate_check import (
    aggregate_gate_status,
    build_gate_system_prompt,
    parse_gate_response,
    render_gate_line,
    summarize_gate_results,
)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    DEEP = json.load(_f)["深科技"]
GATES = DEEP["gates"]
GATE1 = next(g for g in GATES if g["id"] == "Gate 1")
GATE2 = next(g for g in GATES if g["id"] == "Gate 2")


# parse_gate_response()：met／not_met 且 basis、quote 都有才成功
@pytest.mark.parametrize("status", ["met", "not_met"])
def test_parse_gate_complete(status):
    raw = json.dumps({"status": status, "basis": "b", "quote": "q"})
    result = parse_gate_response(raw)
    assert result == {"status": status, "basis": "b", "quote": "q", "parse_error": None}


# parse_gate_response()：undetermined 是正常判斷，parse_error 為 None，quote 一律 None
def test_parse_gate_undetermined_is_normal():
    result = parse_gate_response('{"status": "undetermined", "basis": "文件沒寫查核結果", "quote": "x"}')
    assert result["status"] == "undetermined"
    assert result["parse_error"] is None
    assert result["basis"] == "文件沒寫查核結果"
    assert result["quote"] is None


# parse_gate_response()：met／not_met 卻缺引用或依據 → 歸入 undetermined 並記錄錯誤
@pytest.mark.parametrize("payload", [
    {"status": "met", "basis": "b"},
    {"status": "met", "basis": "b", "quote": ""},
    {"status": "not_met", "quote": "q"},
    {"status": "not_met", "basis": None, "quote": "q"},
])
def test_parse_gate_incomplete_falls_back_to_undetermined(payload):
    result = parse_gate_response(json.dumps(payload))
    assert result["status"] == "undetermined"
    assert result["parse_error"] is not None


# parse_gate_response()：status 不合法、JSON 壞掉、不是物件 → undetermined 並記錄錯誤
@pytest.mark.parametrize("raw", [
    '{"status": "yes", "basis": "b", "quote": "q"}', "not json", '["met"]', '{"basis": "b"}',
])
def test_parse_gate_invalid_output(raw):
    result = parse_gate_response(raw)
    assert result["status"] == "undetermined"
    assert result["parse_error"] is not None


# parse_gate_response()：配對的 think 標籤與 json code fence 要能剝殼
@pytest.mark.parametrize("raw", [
    '<think>先找有沒有實測數據</think>\n{"status": "not_met", "basis": "b", "quote": "q"}',
    '```json\n{"status": "not_met", "basis": "b", "quote": "q"}\n```',
])
def test_parse_gate_strips_think_and_fence(raw):
    result = parse_gate_response(raw)
    assert result["status"] == "not_met"
    assert result["parse_error"] is None


# aggregate_gate_status()：彙總規則
@pytest.mark.parametrize("statuses,expected", [
    (["met", "not_met"], "conflict"),
    (["met", "undetermined", "not_met"], "conflict"),
    (["met"], "met"),
    (["met", "undetermined"], "met"),
    (["not_met", "undetermined"], "not_met"),
    (["undetermined", "undetermined"], "undetermined"),
    ([], "undetermined"),
])
def test_aggregate_gate_status(statuses, expected):
    assert aggregate_gate_status(statuses) == expected


def _entry(gate_id, status, question="q"):
    return {
        "question": question, "gate": gate_id, "heading": "玖",
        "detection": {"status": status, "basis": "b", "quote": "q" if status != "undetermined" else None,
                      "parse_error": None},
    }


# summarize_gate_results()：依模板順序輸出，即使輸入順序相反
def test_summarize_gate_results_follows_template_order():
    results = summarize_gate_results([_entry("Gate 2", "not_met"), _entry("Gate 1", "met")], GATES)
    assert list(results) == ["Gate 1", "Gate 2"]
    assert results["Gate 1"]["status"] == "met"
    assert results["Gate 2"]["status"] == "not_met"


# summarize_gate_results()：只包含實際被檢查過的 Gate
def test_summarize_gate_results_only_checked_gates():
    results = summarize_gate_results([_entry("Gate 2", "undetermined")], GATES)
    assert list(results) == ["Gate 2"]


# summarize_gate_results()：entries 全部保留（不成立、無法判定也在），彙總為 conflict
def test_summarize_gate_results_keeps_all_entries():
    entries = [_entry("Gate 1", "met", "q1"), _entry("Gate 1", "not_met", "q2"), _entry("Gate 1", "undetermined", "q3")]
    results = summarize_gate_results(entries, GATES)
    assert [e["question"] for e in results["Gate 1"]["entries"]] == ["q1", "q2", "q3"]
    assert [e["status"] for e in results["Gate 1"]["entries"]] == ["met", "not_met", "undetermined"]
    assert results["Gate 1"]["status"] == "conflict"


# render_gate_line()：成立但目前等級低於調升目標 → 顯示調升、不自動改
def test_render_gate_line_met_below_target():
    line = render_gate_line(GATE1, "met", "C", DEEP)
    assert GATE1["if_met"]["text"] in line
    assert "目前系統評分：C（未自動調整，由人工決定）" in line


# render_gate_line()：成立且目前等級已等於或高於調升目標 → 無需調整
@pytest.mark.parametrize("current", ["B+", "A"])
def test_render_gate_line_met_already_at_target(current):
    line = render_gate_line(GATE1, "met", current, DEEP)
    assert f"目前系統評分：{current}，已達調升目標，無需調整" in line


# render_gate_line()：Gate 2 的目標是 A−，目前 B+ 仍低於目標
def test_render_gate_line_gate2_target():
    line = render_gate_line(GATE2, "met", "B+", DEEP)
    assert "未自動調整" in line


# render_gate_line()：成立但維度評分失敗或沒評分
def test_render_gate_line_met_without_grade():
    line = render_gate_line(GATE1, "met", None, DEEP)
    assert "評分失敗或未評分" in line


# render_gate_line()：目前等級不合法時丟錯，不默默比較
def test_render_gate_line_met_illegal_grade_raises():
    with pytest.raises(ValueError):
        render_gate_line(GATE1, "met", "D", DEEP)


# render_gate_line()：不成立逐字顯示原文的 if_not_met；矛盾與無法判定各有顯示
@pytest.mark.parametrize("status,expected", [
    ("not_met", GATE1["if_not_met"]),
    ("conflict", "需人工判斷"),
    ("undetermined", "尚未查核"),
])
def test_render_gate_line_other_statuses(status, expected):
    line = render_gate_line(GATE1, status, "C", DEEP)
    assert expected in line
    assert "調至" not in line


# build_gate_system_prompt()：帶入原文條件、原文判準、三種狀態
@pytest.mark.parametrize("gate", GATES)
def test_build_gate_system_prompt(gate):
    prompt = build_gate_system_prompt(gate)
    assert gate["condition"] in prompt
    assert gate["criterion_verbatim"] in prompt
    for status in ("met", "not_met", "undetermined"):
        assert status in prompt
