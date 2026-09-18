"""
測試重點：
- parse_red_flag_response()：分三組情境
  1. is_red_flag=False 時，不管其他欄位是否完整，全部回傳 None 且 parse_error=None（正常判斷）
  2. is_red_flag=True 但欄位不全或 severity 不合法時，回傳 is_red_flag=False 且 parse_error 有值（格式錯誤）
  3. 完整、合法的紅旗 JSON 被 markdown code fence 或配對的 <think>...</think> 包住時，
     要能正確剝殼後解析出 is_red_flag=True 且欄位齊全；非合法 JSON 時走解析失敗路徑
- dedupe_red_flags()：空列表、過濾非紅旗、相同 title 合併、不同 title 不合併、依 severity 排序、
  維持原始順序、混合紅旗與非紅旗
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from red_flag_detection import (
    parse_red_flag_response,
    dedupe_red_flags,
)


# parse_red_flag_response()：is_red_flag=False 時，不管其他欄位是否完整，全部回傳 None 且 parse_error=None（正常判斷）
@pytest.mark.parametrize("raw,description", [
    ('{"is_red_flag": false}', "is_red_flag=False 且欄位完整"),
    ('{"is_red_flag": false, "title": "test"}', "is_red_flag=False 但其他欄位有值"),
    ('{"is_red_flag": false, "severity": "high"}', "is_red_flag=False 但 severity 有值"),
])
def test_parse_red_flag_response_not_red_flag_is_normal(raw, description):
    result = parse_red_flag_response(raw)
    assert result["is_red_flag"] is False, f"{description}：is_red_flag 應為 False"
    assert result["title"] is None, f"{description}：title 應為 None"
    assert result["phenomenon"] is None, f"{description}：phenomenon 應為 None"
    assert result["why_it_matters"] is None, f"{description}：why_it_matters 應為 None"
    assert result["required_action"] is None, f"{description}：required_action 應為 None"
    assert result["severity"] is None, f"{description}：severity 應為 None"
    assert result["parse_error"] is None, f"{description}：parse_error 應為 None"


# parse_red_flag_response()：is_red_flag=True 但欄位不全或 severity 不合法時，回傳 is_red_flag=False 且 parse_error 有值（格式錯誤）
@pytest.mark.parametrize("raw,description", [
    ('{"is_red_flag": true}', "缺少必要欄位"),
    ('{"is_red_flag": true, "title": "test"}', "缺少 phenomenon"),
    ('{"is_red_flag": true, "title": "test", "phenomenon": "x", "severity": "invalid"}', "severity 不合法"),
])
def test_parse_red_flag_response_invalid_format(raw, description):
    result = parse_red_flag_response(raw)
    assert result["is_red_flag"] is False, f"{description}：is_red_flag 應為 False"
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"


# parse_red_flag_response()：完整、合法的紅旗 JSON 被 code fence 或配對的 <think> 區塊包住時，
# 要能正確剝殼後解析出 is_red_flag=True 且欄位齊全；非合法 JSON 時走解析失敗路徑
_COMPLETE_RED_FLAG_JSON = (
    '{"is_red_flag": true, "title": "test", "phenomenon": "p", '
    '"why_it_matters": "w", "required_action": "a", "severity": "high"}'
)


@pytest.mark.parametrize("raw,expected", [
    (
        f"```json\n{_COMPLETE_RED_FLAG_JSON}\n```",
        {"is_red_flag": True},
    ),
    (
        f"<think>先分析答案主張，再核對片段全文</think>\n{_COMPLETE_RED_FLAG_JSON}",
        {"is_red_flag": True},
    ),
    (
        "invalid json",
        {"is_red_flag": False, "parse_error": "紅旗判斷結果無法解析成 JSON"},
    ),
])
def test_parse_red_flag_response_fences_and_errors(raw, expected):
    result = parse_red_flag_response(raw)
    assert result["is_red_flag"] == expected["is_red_flag"]
    if "parse_error" in expected:
        assert result["parse_error"] is not None
    else:
        assert result["title"] is not None
        assert result["phenomenon"] is not None
        assert result["why_it_matters"] is not None
        assert result["required_action"] is not None
        assert result["severity"] is not None


def _det(is_red_flag, title=None, severity=None):
    """組出跟 parse_red_flag_response() 保證回傳形狀一致的 detection dict
    （所有 key 都存在，未指定的欄位填 None）——dedupe_red_flags() 用
    det["phenomenon"] 這種直接索引，缺 key 會是 KeyError 不是合理輸入。"""
    if not is_red_flag:
        return {"is_red_flag": False, "title": None, "phenomenon": None,
                "why_it_matters": None, "required_action": None,
                "severity": None, "parse_error": None}
    return {"is_red_flag": True, "title": title, "phenomenon": None,
            "why_it_matters": None, "required_action": None,
            "severity": severity, "parse_error": None}


# dedupe_red_flags()：空列表、過濾非紅旗、相同 title 合併、不同 title 不合併
@pytest.mark.parametrize("entries,expected", [
    ([], []),
    ([{"question": "q0", "detection": _det(False), "heading": None}], []),
    (
        [
            {"question": "q1", "detection": _det(True, title="Test"), "heading": "h1"},
            {"question": "q2", "detection": _det(True, title="test"), "heading": "h2"},
        ],
        [
            {
                "title": "Test",
                "matched_questions": ["q1", "q2"],
                "headings": ["h1", "h2"],
            }
        ],
    ),
    (
        [
            {"question": "q1", "detection": _det(True, title="Test"), "heading": "h1"},
            {"question": "q2", "detection": _det(True, title="Different"), "heading": "h2"},
        ],
        [
            {
                "title": "Test",
                "matched_questions": ["q1"],
                "headings": ["h1"],
            },
            {
                "title": "Different",
                "matched_questions": ["q2"],
                "headings": ["h2"],
            },
        ],
    ),
])
def test_dedupe_red_flags_basic_scenarios(entries, expected):
    result = dedupe_red_flags(entries)
    assert len(result) == len(expected)
    for i in range(len(result)):
        assert result[i]["title"] == expected[i]["title"]
        assert result[i]["matched_questions"] == expected[i]["matched_questions"]
        assert result[i]["headings"] == expected[i]["headings"]


# dedupe_red_flags()：依 severity 排序、維持原始順序、混合紅旗與非紅旗
@pytest.mark.parametrize("entries,expected_severities", [
    (
        [
            {"question": "q1", "detection": _det(True, title="A", severity="low"), "heading": None},
            {"question": "q2", "detection": _det(True, title="B", severity="high"), "heading": None},
            {"question": "q3", "detection": _det(True, title="C", severity="medium"), "heading": None},
        ],
        ["high", "medium", "low"],
    ),
    (
        [
            {"question": "q1", "detection": _det(True, title="A", severity="medium"), "heading": None},
            {"question": "q2", "detection": _det(True, title="B", severity="medium"), "heading": None},
            {"question": "q3", "detection": _det(True, title="C", severity="medium"), "heading": None},
        ],
        ["medium", "medium", "medium"],
    ),
    (
        [
            {"question": "q1", "detection": _det(True, title="A", severity="high"), "heading": None},
            {"question": "q2", "detection": _det(False), "heading": None},
            {"question": "q3", "detection": _det(True, title="B", severity="high"), "heading": None},
        ],
        ["high", "high"],
    ),
])
def test_dedupe_red_flags_severity_order(entries, expected_severities):
    result = dedupe_red_flags(entries)
    assert [rf["severity"] for rf in result] == expected_severities
