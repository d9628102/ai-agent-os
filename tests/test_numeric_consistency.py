"""
測試重點：
- parse_extraction_response()：拆成三組情境，分開斷言
  1. 真正的解析失敗（JSON 壞掉、value_normalized 無法轉 float）—— parse_error 一定要有值
  2. 正常判斷「沒找到」（合法 JSON 但缺 found、found 宣稱 true 卻沒有可用數值、
     模型直接回答 found:false）—— parse_error 應為 None（這不是錯誤），且模型
     有給的 raw_value/unit 等欄位要保留下來，不是全部清空
  3. 正常找到數字的成功路徑，含 markdown code fence 與 <think> 區塊要先被剝除
     才能解析
- locate_source_heading()：source_snippet 空值、子字串比對、多重匹配取第一個、
  無匹配情境
- check_group_consistency()：0 個/1 個 found=True 都算 insufficient、浮點誤差
  容許範圍內的一致值、實質不一致值、混合 found=False 與 found=True 時排除
  found=False 只看剩下的子集合
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from numeric_consistency import (
    check_group_consistency,
    locate_source_heading,
    parse_extraction_response,
)


# parse_extraction_response()：真正的解析失敗 —— parse_error 一定要有值
@pytest.mark.parametrize("raw,description", [
    ('{"found": true, "value_normalized": "123.45"', "JSON 格式不完整（缺結尾括號）"),
    ('{"found": true, "value_normalized": "abc"}', "value_normalized 是無法轉成 float 的字串"),
])
def test_parse_extraction_response_parse_errors(raw, description):
    result = parse_extraction_response(raw)
    assert result["found"] is False, f"{description}：found 應為 False，實際 {result['found']}"
    assert result["value_normalized"] is None, f"{description}：value_normalized 應為 None"
    assert result["parse_error"] is not None, (
        f"{description}：這是真正的解析失敗，parse_error 應該要有說明，實際是 None"
    )


# parse_extraction_response()：正常判斷「沒找到」—— 這不是錯誤，parse_error
# 應為 None；模型有給的欄位（raw_value/unit）要保留，不是全部清空
@pytest.mark.parametrize("raw,expected", [
    (
        '{"score": 0.6}',
        {"raw_value": None, "unit": None},
    ),
    (
        '{"found": true, "value_normalized": null, "raw_value": "看不出具體數字", "unit": "USD"}',
        {"raw_value": "看不出具體數字", "unit": "USD"},
    ),
    (
        '{"found": false, "raw_value": "123", "unit": "kg"}',
        {"raw_value": "123", "unit": "kg"},
    ),
])
def test_parse_extraction_response_not_found_is_not_an_error(raw, expected):
    result = parse_extraction_response(raw)
    assert result["found"] is False
    assert result["value_normalized"] is None
    assert result["parse_error"] is None, (
        f"這是正常判斷沒找到，不是解析失敗，parse_error 應為 None，實際是 {result['parse_error']!r}"
    )
    assert result["raw_value"] == expected["raw_value"], (
        f"raw_value 應該保留模型給的值 {expected['raw_value']!r}，實際 {result['raw_value']!r}"
    )
    assert result["unit"] == expected["unit"], (
        f"unit 應該保留模型給的值 {expected['unit']!r}，實際 {result['unit']!r}"
    )


# parse_extraction_response()：正常找到數字的成功路徑，含 code fence 與
# <think> 區塊需要先被剝除才能解析成 JSON
@pytest.mark.parametrize("raw,expected_value", [
    (
        '{"found": true, "value_normalized": "123.45", "raw_value": "123.45", "unit": "kg"}',
        123.45,
    ),
    (
        "```json\n{\"found\": true, \"value_normalized\": \"123.45\"}\n```",
        123.45,
    ),
    (
        "<think>先確認單位再換算</think>\n{\"found\": true, \"value_normalized\": \"123.45\"}",
        123.45,
    ),
])
def test_parse_extraction_response_success(raw, expected_value):
    result = parse_extraction_response(raw)
    assert result["found"] is True
    assert result["value_normalized"] == expected_value, (
        f"應解析出 {expected_value}，實際 {result['value_normalized']}"
    )
    assert result["parse_error"] is None


# locate_source_heading() 測試案例
@pytest.mark.parametrize("source_snippet, hits, expected", [
    # 空值情境
    (None, [], None),
    ("   ", [], None),
    # 子字串匹配
    ("body text", [{"payload": {"body": "body text", "heading_path": "3.2 章節"}}], "3.2 章節"),
    # 多個匹配取第一個（依 hits 清單順序，不是分數或長度）
    ("text", [
        {"payload": {"body": "text", "heading_path": "A"}},
        {"payload": {"body": "text", "heading_path": "B"}},
    ], "A"),
    # 無匹配
    ("missing", [{"payload": {"body": "other"}}], None),
])
def test_locate_source_heading_boundary_cases(source_snippet, hits, expected):
    result = locate_source_heading(source_snippet, hits)
    assert result == expected, (
        f"source_snippet {source_snippet!r} 在 hits 中應返回 {expected!r}，實際得到 {result!r}"
    )


# check_group_consistency() 測試案例
@pytest.mark.parametrize("entries, expected_status", [
    # 0 個 found=True
    ([{"extraction": {"found": False}}], "insufficient"),
    # 剛好 1 個 found=True（規格明訂至少要 2 個才夠比對）
    ([{"extraction": {"found": True, "value_normalized": 1.0}}], "insufficient"),
    # 多個一致值（含極小浮點誤差，仍視為一致）
    ([{"extraction": {"found": True, "value_normalized": 1.0}},
      {"extraction": {"found": True, "value_normalized": 1.0000001}}], "consistent"),
    # 多個不一致值
    ([{"extraction": {"found": True, "value_normalized": 1.0}},
      {"extraction": {"found": True, "value_normalized": 2.0}}], "conflict"),
    # 混合 found=True 和 found=False —— found=False 不計入，只看剩下的子集合
    ([{"extraction": {"found": False}},
      {"extraction": {"found": True, "value_normalized": 1.0}},
      {"extraction": {"found": True, "value_normalized": 1.0}}], "consistent"),
])
def test_check_group_consistency_boundary_cases(entries, expected_status):
    result = check_group_consistency("metric_name", entries)
    assert result["status"] == expected_status, (
        f"entries {entries} 應返回 {expected_status}，實際得到 {result['status']}"
    )
