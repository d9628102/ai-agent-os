"""
測試重點：
- parse_veto_response()：驗證 JSON 解析、欄位完整性檢查、code fence/配對 think 標籤剝殼、錯誤處理
- summarize_veto_results()：驗證規則分組邏輯、觸發狀態聚合、固定順序輸出、僅包含實際測試過的規則
- build_veto_banner_line()：驗證空結果處理、觸發規則顯示、未觸發狀態顯示
- _build_veto_system_prompt()：驗證五條規則各自組出正確的模式區塊（矛盾型/
  缺失型 wiring，逐條核對不是只測一矛盾一缺失就假設其他都對）、規則名稱與
  描述文字、JSON_OUTPUT_REMINDER 共用防線；不存在的規則名稱要拋出 KeyError
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from veto_check import (
    VETO_RULES,
    _ABSENCE_MODE_BLOCK,
    _CONTRADICTION_MODE_BLOCK,
    _build_veto_system_prompt,
    parse_veto_response,
    summarize_veto_results,
)
from generate_report import build_veto_banner_line
from rag_common import JSON_OUTPUT_REMINDER

# parse_veto_response()：合法JSON(is_veto_triggered=True且欄位完整)時,回傳完整結果,parse_error=None
@pytest.mark.parametrize("raw,description", [
    ('{"is_veto_triggered": true, "phenomenon": "test", "basis": "rule1", "suggested_action": "reject"}', "合法觸發"),
    ('{"is_veto_triggered": false}', "合法不觸發"),
])
def test_parse_veto_response_valid(raw, description):
    result = parse_veto_response(raw)
    assert result["parse_error"] is None, f"{description}：parse_error 應為 None"
    assert "is_veto_triggered" in result
    if result["is_veto_triggered"]:
        assert "phenomenon" in result and result["phenomenon"]
        assert "basis" in result and result["basis"]
        assert "suggested_action" in result and result["suggested_action"]
    else:
        assert result["phenomenon"] is None
        assert result["basis"] is None
        assert result["suggested_action"] is None

# parse_veto_response()：JSON解析失敗時,parse_error要有值
@pytest.mark.parametrize("raw,description", [
    ("invalid json", "純文字"),
    ('"is_veto_triggered": true}', "缺開頭大括號"),
    ("{ \"is_veto_triggered\": \"true\" }", "無法轉成布林"),
])
def test_parse_veto_response_json_error(raw, description):
    result = parse_veto_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"

# parse_veto_response()：觸發但欄位不完整時,視為抽取失敗,parse_error要有值
@pytest.mark.parametrize("raw,description", [
    ('{"is_veto_triggered": true}', "缺欄位"),
    ('{"is_veto_triggered": true, "phenomenon": "test"}', "缺basis"),
    ('{"is_veto_triggered": true, "phenomenon": "test", "basis": "rule"}', "缺suggested_action"),
])
def test_parse_veto_response_incomplete_fields(raw, description):
    result = parse_veto_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"
    # 欠欄位視同抽取失敗，跟 red_flag_detection.py 的 is_red_flag=true 缺欄位
    # 同一個原則——不半殘顯示一個沒有依據的veto，is_veto_triggered 要回退成 False
    assert result["is_veto_triggered"] is False

# parse_veto_response()：模型輸出被配對的<think>...</think>標籤或三個反引號
# 加json包住時,要能正確剝殼後解析——一定要用真正配對的 think 標籤,不能用
# 孤立的結尾標籤(真實模型輸出只會是配對標籤或完全沒有標籤)
@pytest.mark.parametrize("raw,expected_triggered", [
    ("```json\n{ \"is_veto_triggered\": true, \"phenomenon\": \"test\", \"basis\": \"rule1\", \"suggested_action\": \"reject\" }\n```", True),
    ("<think>先確認片段裡有沒有明確反證，沒有的話不觸發</think>\n{ \"is_veto_triggered\": false }\n", False),
])
def test_parse_veto_response_fences(raw, expected_triggered):
    result = parse_veto_response(raw)
    assert result["parse_error"] is None, "應成功解析"
    assert result["is_veto_triggered"] == expected_triggered

# summarize_veto_results()：同一條規則有多筆entries，其中一筆觸發整條規則彙總為觸發
@pytest.mark.parametrize("veto_entries,expected_rules", [
    (
        [
            {"question": "Q1", "rule": "人不明", "heading": "第一節",
             "detection": {"is_veto_triggered": True, "phenomenon": "p1",
                           "basis": "b1", "suggested_action": "a1"}},
            {"question": "Q2", "rule": "人不明", "heading": "第二節",
             "detection": {"is_veto_triggered": False}},
        ],
        ["人不明"]
    ),
])
def test_summarize_veto_results_one_rule_triggered(veto_entries, expected_rules):
    results = summarize_veto_results(veto_entries)
    triggered_rules = [rule for rule, res in results.items() if res["triggered"]]
    assert set(triggered_rules) == set(expected_rules)
    # 只有觸發的那一筆(Q1)進 entries，未觸發的Q2不應該出現
    assert len(results["人不明"]["entries"]) == 1
    assert results["人不明"]["entries"][0]["question"] == "Q1"

# summarize_veto_results()：所有entries都未觸發時，規則彙總為不觸發
@pytest.mark.parametrize("veto_entries", [
    [{"rule": "資源不實", "detection": {"is_veto_triggered": False}}],
    [],
])
def test_summarize_veto_results_all_not_triggered(veto_entries):
    results = summarize_veto_results(veto_entries)
    for res in results.values():
        assert not res["triggered"]

# summarize_veto_results()：輸出順序固定跟VETO_RULES一致，而非輸入順序；
# 且只有實際被標記測試過的規則才出現（這裡輸入只有兩條，預期輸出也只有
# 這兩條，不是VETO_RULES全部五條）
@pytest.mark.parametrize("veto_entries,expected_order", [
    (
        [
            {"question": "Q1", "rule": "利益不明", "heading": None,
             "detection": {"is_veto_triggered": True, "phenomenon": "p",
                           "basis": "b", "suggested_action": "a"}},
            {"question": "Q2", "rule": "人不明", "heading": None,
             "detection": {"is_veto_triggered": True, "phenomenon": "p",
                           "basis": "b", "suggested_action": "a"}},
        ],
        ["人不明", "利益不明"]
    ),
])
def test_summarize_veto_results_order(veto_entries, expected_order):
    results = summarize_veto_results(veto_entries)
    assert list(results.keys()) == expected_order

# summarize_veto_results()：只有實際被標記測試過的規則才出現在結果裡
@pytest.mark.parametrize("veto_entries", [
    [{"rule": "不存在規則", "detection": {"is_veto_triggered": True}}],
])
def test_summarize_veto_results_only_existing_rules(veto_entries):
    results = summarize_veto_results(veto_entries)
    assert len(results) < len(VETO_RULES)  # 假設VETO_RULES是預先定義的常數

# build_veto_banner_line()：veto_results為空字典時回傳None
def test_build_veto_banner_line_empty():
    assert build_veto_banner_line({}) is None

# build_veto_banner_line()：有觸發規則時組出包含規則名稱的🚫文字
@pytest.mark.parametrize("veto_results,expected", [
    ({"人不明": {"triggered": True}}, "**不合作紅線**：🚫 觸發「人不明」——建議不合作"),
    ({"人不明": {"triggered": True}, "資源不實": {"triggered": True}}, "**不合作紅線**：🚫 觸發「人不明、資源不實」——建議不合作"),
])
def test_build_veto_banner_line_triggered(veto_results, expected):
    assert build_veto_banner_line(veto_results) == expected

# build_veto_banner_line()：全部測試過但都未觸發時組出✅未觸發文字
@pytest.mark.parametrize("veto_results", [
    {"人不明": {"triggered": False}},
    {"人不明": {"triggered": False}, "資源不實": {"triggered": False}},
])
def test_build_veto_banner_line_not_triggered(veto_results):
    assert build_veto_banner_line(veto_results) == "**不合作紅線**：✅ 未觸發"


# _build_veto_system_prompt()：逐條驗證五條規則各自組出正確的模式區塊——
# 不能只測一條矛盾型加一條缺失型就假設其他都對，這是這個函式歷史上最
# 容易在編輯時不小心弄反的部分。直接 import 真正的模組常數逐段比對，不
# 自己重寫一份同名替代品（跟 tests/test_code_review_helpers.py 記錄的
# shadow 定義假陽性是同一個要避免的陷阱）。
@pytest.mark.parametrize("rule_name, expected_mode_block", [
    ("資源不實", _CONTRADICTION_MODE_BLOCK),
    ("人不明", _ABSENCE_MODE_BLOCK),
    ("權責不清", _ABSENCE_MODE_BLOCK),
    ("利益不明", _ABSENCE_MODE_BLOCK),
    ("風險不揭露", _ABSENCE_MODE_BLOCK),
])
def test_build_veto_system_prompt_rule_coverage(rule_name, expected_mode_block):
    prompt = _build_veto_system_prompt(rule_name)
    assert rule_name in prompt, f"提示應包含規則名稱 {rule_name}"
    assert VETO_RULES[rule_name]["description"] in prompt, f"提示應包含規則描述 {VETO_RULES[rule_name]['description']}"
    assert expected_mode_block in prompt, f"提示應包含正確的模式塊 {expected_mode_block}"
    assert JSON_OUTPUT_REMINDER in prompt, "提示應包含 JSON_OUTPUT_REMINDER"


def test_build_veto_system_prompt_invalid_rule():
    with pytest.raises(KeyError):
        _build_veto_system_prompt("invalid_rule")