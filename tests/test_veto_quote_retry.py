"""
測試重點（Gate 3 三態判斷：引文驗證失敗時只重試一次）：

背景：回歸案例 A1（Gate 3 真實正例）9 次只觸發 6 次，另外 3 次是模型把原文
「必須在投資前釐清」寫成「需在投資前釐清」，少一個字，逐字驗證擋下後降級成 ⏸。
設計：模型判為承認（admitted）但引文驗證失敗時，把失敗的引文告訴模型，請它重新從
檢索片段逐字複製，最多重試一次；重試的輸出走跟第一次完全相同的解析與逐字驗證，
標準沒有放寬；重試仍失敗就照舊降級成 silent。

- needs_quote_retry()：只有「admitted 但引文驗證失敗」才重試；模型自己判 silent／clear、
  解析失敗、admitted 且驗證通過都不重試
- build_quote_retry_message()：只轉述失敗的事實（失敗的引文、找不到的片段），不提供
  正確答案
- detect_veto()（有 verify_quote 的規則）：模型呼叫次數——沉默、未觸發、驗證通過、
  解析失敗、HTTP 失敗一律 1 次；需要重試時最多 2 次，絕不會有第 3 次
- 重試後的引文一樣要逐字驗證：改寫的引文、不在文件裡的引文，重試也放不過
- 重試失敗的各種情況（HTTP 失敗、輸出壞掉）保留第一次的降級結果，不會把 ⏸ 變成別的狀態
- 沒有 verify_quote 旗標的規則（所有五不合作）從不重試，回傳結構不變
- summarize_veto_results()：重試過仍失敗的沉默明細，原因補一句「已重試」

真實資料：引文用的是遮蔽後 Branes.AI 報告的真實句子；第一次的錯誤引文就是 2026-09-29
實測時模型真的寫過的那一句。
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import veto_check
from veto_check import (
    REASON_NO_QUOTE,
    REASON_QUOTE_NOT_IN_CONTEXT,
    build_quote_retry_message,
    detect_veto,
    needs_quote_retry,
    summarize_veto_results,
)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    DEEP = json.load(_f)["深科技"]
RULE = "智財歸屬不完整"
RULES = DEEP["veto_rules"]

# 真實文件（遮蔽後的 Branes.AI DD 報告 3.3 節）
CONTEXT = (
    "[文件片段 1] Branes.AI 商業模式暨投資盡職調查報告 > 參 技術架構與智慧財產權查核 > 3.3 智財歸屬風險\n"
    "核心專利之原始受讓人為 Stillwater Supercomputing, Inc.，而非 Branes.AI。這產生一個必須在投資前釐清的問題：\n"
    "\n"
    "  IP 轉移鏈是否完整？\n"
)
GOOD_QUOTE = "這產生一個必須在投資前釐清的問題：IP 轉移鏈是否完整？"
# 2026-09-29 實測，模型真的寫過的那一句：「必須」被寫成「需」
BAD_QUOTE_REAL = "需在投資前釐清的問題：IP 轉移鏈是否完整？"


def _admitted(quote):
    return json.dumps({"status": "admitted", "quote": quote, "phenomenon": "智財移轉鏈未確認",
                       "suggested_action": "取得移轉文件前不可投資"}, ensure_ascii=False)


SILENT = json.dumps({"status": "silent", "quote": None, "phenomenon": None, "suggested_action": None})
CLEAR = json.dumps({"status": "clear", "quote": None, "phenomenon": None, "suggested_action": None})


class FakeHttp:
    """依序回傳預先寫好的模型輸出，並記下每次呼叫的 payload。輸出為 None 時模擬 HTTP 500，
    為 dict 時原樣當成 200 回應（用來模擬格式不對）。"""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, method, url, payload, timeout=None):
        self.calls.append(copy.deepcopy(payload))
        out = self.outputs[len(self.calls) - 1] if len(self.calls) <= len(self.outputs) else self.outputs[-1]
        if out is None:
            return 500, {"error": "boom"}
        if isinstance(out, dict):
            return 200, out
        return 200, {"choices": [{"message": {"content": out}}]}


def _run(monkeypatch, *outputs, context=CONTEXT, rules=RULES, rule=RULE):
    fake = FakeHttp(*outputs)
    monkeypatch.setattr(veto_check, "http_json", fake)
    return detect_veto(rule, "問題", "回答", context, "u", "m", rules=rules), fake


# ---------------------------------------------------------------------------
# needs_quote_retry() / build_quote_retry_message()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("result,expected", [
    ({"status": "silent", "downgrade_reason": REASON_QUOTE_NOT_IN_CONTEXT}, True),
    ({"status": "silent", "downgrade_reason": REASON_NO_QUOTE}, True),
    ({"status": "silent", "downgrade_reason": None}, False),        # 模型自己判 silent
    ({"status": "clear", "downgrade_reason": None}, False),
    ({"status": "admitted", "downgrade_reason": None}, False),      # 驗證通過
    ({"status": "undetermined", "downgrade_reason": None, "parse_error": "x"}, False),
    ({"status": "silent"}, False),
])
def test_needs_quote_retry_only_for_admitted_with_failed_quote(result, expected):
    assert needs_quote_retry(result) is expected


def test_retry_message_states_the_failure_without_giving_the_answer():
    msg = build_quote_retry_message({"downgrade_reason": REASON_QUOTE_NOT_IN_CONTEXT, "quote": BAD_QUOTE_REAL,
                                     "missing_pieces": ["需在投資前釐清的問題"]})
    assert BAD_QUOTE_REAL in msg and "「需在投資前釐清的問題」" in msg
    assert "逐字複製" in msg and "silent" in msg and "不能改寫" in msg
    assert "必須在投資前釐清" not in msg          # 不提供正確答案


def test_retry_message_for_missing_quote_uses_different_wording():
    msg = build_quote_retry_message({"downgrade_reason": REASON_NO_QUOTE, "quote": None, "missing_pieces": []})
    assert "沒有給出足夠長的引文" in msg and "逐字複製" in msg


# ---------------------------------------------------------------------------
# detect_veto()：什麼時候打第二次
# ---------------------------------------------------------------------------

def test_real_a1_failure_is_recovered_by_one_retry_with_verbatim_quote(monkeypatch):
    result, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(GOOD_QUOTE))
    assert len(fake.calls) == 2
    assert result["status"] == "admitted" and result["is_veto_triggered"] is True
    assert result["quote"] == GOOD_QUOTE and result["basis"] == f"「{GOOD_QUOTE}」"
    assert result["retried"] is True and result["first_quote"] == BAD_QUOTE_REAL
    assert result["rule"] == RULE and result["downgrade_reason"] is None


def test_retry_request_carries_original_conversation_and_the_failed_quote(monkeypatch):
    _, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(GOOD_QUOTE))
    first, second = fake.calls
    assert second["messages"][:2] == first["messages"]                      # system 與 user 原封不動
    assert second["messages"][2] == {"role": "assistant", "content": _admitted(BAD_QUOTE_REAL)}
    assert second["messages"][3]["role"] == "user"
    assert BAD_QUOTE_REAL in second["messages"][3]["content"]
    for key in ("model", "temperature", "max_tokens", "chat_template_kwargs"):
        assert second[key] == first[key]                                    # 呼叫參數一樣，沒有放寬


def test_retry_that_still_fails_verification_downgrades_and_stops_at_two_calls(monkeypatch):
    result, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(BAD_QUOTE_REAL))
    assert len(fake.calls) == 2                                             # 絕不會有第 3 次
    assert result["status"] == "silent" and result["is_veto_triggered"] is False
    assert result["downgrade_reason"] == REASON_QUOTE_NOT_IN_CONTEXT
    assert result["retried"] is True and result["first_quote"] == BAD_QUOTE_REAL


@pytest.mark.parametrize("retry_quote", [
    "需在投資前釐清這件事：IP 轉移鏈是否完整？",       # 又改寫
    "核心專利仍登記於 Stillwater，移轉鏈不完整",         # 編出一句（文件別處的說法，這份片段裡沒有）
    "文件中未提及智財歸屬",                              # 系統回答的說法
    "未揭露",                                            # 太短
    None,
])
def test_retry_cannot_bring_an_unverifiable_quote_through(monkeypatch, retry_quote):
    result, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(retry_quote))
    assert len(fake.calls) == 2
    assert result["status"] == "silent" and result["is_veto_triggered"] is False
    assert result["basis"] is None and result["retried"] is True


def test_accepted_retry_quote_is_always_a_verbatim_part_of_the_context(monkeypatch):
    result, _ = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(GOOD_QUOTE))
    assert veto_check.verify_quote_in_context(result["quote"], CONTEXT)[0] is True


def test_retry_model_backing_off_to_silent_is_accepted_as_silent(monkeypatch):
    result, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), SILENT)
    assert len(fake.calls) == 2
    assert result["status"] == "silent" and result["retried"] is True and result["downgrade_reason"] is None


def test_no_quote_first_answer_is_also_retried_once(monkeypatch):
    result, fake = _run(monkeypatch, _admitted(None), _admitted(GOOD_QUOTE))
    assert len(fake.calls) == 2 and result["status"] == "admitted"
    assert "沒有給出足夠長的引文" in fake.calls[1]["messages"][3]["content"]


@pytest.mark.parametrize("first", [SILENT, CLEAR, _admitted(GOOD_QUOTE)])
def test_no_second_model_call_for_silent_clear_or_verified_admission(monkeypatch, first):
    result, fake = _run(monkeypatch, first)
    assert len(fake.calls) == 1
    assert result["retried"] is False and result["first_quote"] is None


@pytest.mark.parametrize("first", ["not json", '["admitted"]', '{"status": "yes"}', None, {"unexpected": 1}])
def test_no_second_model_call_when_first_output_is_broken(monkeypatch, first):
    result, fake = _run(monkeypatch, first)
    assert len(fake.calls) == 1
    assert result["status"] == "undetermined" and result["retried"] is False


# ---------------------------------------------------------------------------
# 重試失敗：保留第一次的降級結果
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("second,error_fragment", [
    (None, "HTTP 500"),
    ({"unexpected": 1}, "格式不如預期"),
    ("not json", "無法解析"),
    ('{"status": "maybe"}', "無法解析"),
])
def test_failed_retry_keeps_the_first_downgrade_and_records_why(monkeypatch, second, error_fragment):
    result, fake = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), second)
    assert len(fake.calls) == 2
    assert result["status"] == "silent" and result["is_veto_triggered"] is False
    assert result["downgrade_reason"] == REASON_QUOTE_NOT_IN_CONTEXT
    assert result["retried"] is True and result["first_quote"] == BAD_QUOTE_REAL
    assert error_fragment in result["retry_error"]


# ---------------------------------------------------------------------------
# 只有有旗標的規則才重試
# ---------------------------------------------------------------------------

def test_default_five_rules_never_retry_and_keep_their_original_shape(monkeypatch):
    fake = FakeHttp('{"is_veto_triggered": false, "phenomenon": null}')
    monkeypatch.setattr(veto_check, "http_json", fake)
    result = detect_veto("利益不明", "問題", "回答", CONTEXT, "u", "m")
    assert len(fake.calls) == 1
    assert "retried" not in result and "status" not in result

    fake = FakeHttp("這不是 JSON")
    monkeypatch.setattr(veto_check, "http_json", fake)
    result = detect_veto("人不明", "問題", "回答", CONTEXT, "u", "m")
    assert len(fake.calls) == 1 and result["parse_error"]


def test_template_rule_with_flag_off_never_retries(monkeypatch):
    rules = copy.deepcopy(RULES)
    rules[RULE]["verify_quote"] = False
    fake = FakeHttp('{"is_veto_triggered": false}')
    monkeypatch.setattr(veto_check, "http_json", fake)
    result = detect_veto(RULE, "問題", "回答", CONTEXT, "u", "m", rules=rules)
    assert len(fake.calls) == 1 and "retried" not in result


# ---------------------------------------------------------------------------
# summarize：沉默明細的原因
# ---------------------------------------------------------------------------

def _entry(detection):
    return {"question": "Q", "rule": RULE, "detection": detection, "heading": "H"}


def test_summarize_marks_retried_downgrades_in_the_reason(monkeypatch):
    result, _ = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(BAD_QUOTE_REAL))
    entry = summarize_veto_results([_entry(result)], rules=RULES)[RULE]["silent_entries"][0]
    assert entry["reason"] == REASON_QUOTE_NOT_IN_CONTEXT + "（已請模型重新逐字複製一次，仍無法驗證）"


def test_summarize_reason_unchanged_when_not_retried():
    det = {"status": "silent", "is_veto_triggered": False, "downgrade_reason": REASON_QUOTE_NOT_IN_CONTEXT}
    entry = summarize_veto_results([_entry(det)], rules=RULES)[RULE]["silent_entries"][0]
    assert entry["reason"] == REASON_QUOTE_NOT_IN_CONTEXT


def test_summarize_recovered_retry_is_a_normal_trigger(monkeypatch):
    result, _ = _run(monkeypatch, _admitted(BAD_QUOTE_REAL), _admitted(GOOD_QUOTE))
    r = summarize_veto_results([_entry(result)], rules=RULES)[RULE]
    assert r["status"] == "triggered" and r["entries"][0]["basis"] == f"「{GOOD_QUOTE}」"
    assert r["silent_entries"] == []
