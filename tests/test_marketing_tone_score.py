"""
測試重點：
- parse_tone_score_response()：驗證四個子項目（TONE_SUBITEMS）都必須存在
  且各自欄位合法才算成功——跟 marketing_faithfulness.py 不同，這裡任何
  一項缺漏就整批視為失敗，不做部分成功（見模組docstring的理由）
- score_marketing_tone()：被 max_tokens 截斷時要單獨回報，不偽裝成 JSON
  解析失敗（monkeypatch http_json，不打真的模型）
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import marketing_tone_score
from marketing_tone_score import TONE_SUBITEMS, parse_tone_score_response, score_marketing_tone


def _valid_scores_payload():
    return {
        subitem: {"score": 4, "rationale": f"{subitem}的理由", "evidence": f"{subitem}的引用段落"}
        for subitem in TONE_SUBITEMS
    }


# parse_tone_score_response()：四個子項目都合法時,全部正確保留
def test_parse_tone_score_response_valid():
    raw = json.dumps({"scores": _valid_scores_payload()})
    result = parse_tone_score_response(raw)
    assert result["parse_error"] is None
    assert set(result["scores"]) == set(TONE_SUBITEMS)
    for subitem in TONE_SUBITEMS:
        assert result["scores"][subitem]["score"] == 4


# parse_tone_score_response()：整段輸出不是合法JSON,或scores不是物件時,整批失敗
@pytest.mark.parametrize("raw,description", [
    ("not json", "純文字"),
    ('{"scores": "not a dict"}', "scores是字串"),
    ('{"scores": null}', "scores是null"),
    ('{"no_scores_key": {}}', "根本沒有scores欄位"),
])
def test_parse_tone_score_response_top_level_error(raw, description):
    result = parse_tone_score_response(raw)
    assert result["parse_error"] is not None, description
    assert result["scores"] is None, description


# parse_tone_score_response()：任何一個子項目缺漏,整批視為失敗(不是部分成功)
def test_parse_tone_score_response_missing_one_subitem_fails_whole_batch():
    payload = _valid_scores_payload()
    del payload["行動呼籲"]
    result = parse_tone_score_response(json.dumps({"scores": payload}))
    assert result["parse_error"] is not None
    assert result["scores"] is None


# parse_tone_score_response()：子項目存在但score不是1-5整數/缺rationale或evidence時,整批視為失敗
@pytest.mark.parametrize("bad_field,bad_value,description", [
    ("score", 0, "score=0"),
    ("score", 6, "score=6"),
    ("score", "5", "score是字串不是int"),
    ("rationale", "", "rationale空字串"),
    ("evidence", None, "evidence是null"),
])
def test_parse_tone_score_response_invalid_subitem_field(bad_field, bad_value, description):
    payload = _valid_scores_payload()
    payload["開頭吸引力"][bad_field] = bad_value
    result = parse_tone_score_response(json.dumps({"scores": payload}))
    assert result["parse_error"] is not None, description
    assert result["scores"] is None, description


# parse_tone_score_response()：能剝掉<think>推理區塊跟markdown code fence
def test_parse_tone_score_response_strips_think_and_fence():
    raw = "<think>先評估開頭吸引力</think>\n```json\n" + json.dumps({"scores": _valid_scores_payload()}) + "\n```"
    result = parse_tone_score_response(raw)
    assert result["parse_error"] is None
    assert set(result["scores"]) == set(TONE_SUBITEMS)


# score_marketing_tone()：截斷要單獨回報
def test_score_marketing_tone_reports_truncation(monkeypatch):
    def fake(method, url, payload=None, timeout=300):
        return 200, {"choices": [{"message": {"content": "<think>還沒想完"}, "finish_reason": "length"}]}
    monkeypatch.setattr(marketing_tone_score, "http_json", fake)
    result = score_marketing_tone("文案", "url", "model")
    assert result["scores"] is None
    assert "截斷" in result["parse_error"]

