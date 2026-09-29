"""
測試重點（Gate 3 誤觸發修正：三態＋逐字引文＋程式驗證；規則設定 verify_quote=true 才啟用）：

背景：問答系統被要求「文件沒寫就回答『文件中未提及』」，舊的缺失型 prompt 又把回答裡的
「未提及／未說明」當成「文件承認不完整」，沉默被誤判成觸發（2026-09-29 用 PSF EIM 公開
文件實測：4 個沉默案例 2 個誤觸發）。

- parse_verified_absence_response()：三態解析；admitted 必須附「在檢索片段裡逐字找得到」
  的引文，缺漏／太短／改寫／只在系統回答裡出現／簡體化，一律降級成 silent（並記原因）；
  JSON 壞掉、status 不合法 → undetermined（不是「未觸發」）
- verify_quote_in_context()：比對規則（去空白、標點、Markdown *；分段比對；至少 6 個字）
- detection_status()／summarize_veto_results()：規則狀態優先順序
  觸發 > 無法判定 > 文件未提及 > 未觸發；舊格式（五不合作，沒有 status）相容，但沒觸發
  又有 parse_error 的現在算「無法判定」，不再默默當成「未觸發」
- render_veto_section()／build_veto_banner_line()：新狀態的顯示，沉默與無法判定絕不出現
  「✅ 未觸發」
- detect_veto()：有 verify_quote 的規則走新 prompt，其他規則走舊路徑、回傳 dict 不含新欄位
- **五不合作完全不受影響**：五個預設 prompt 與預設報告顯示的雜湊值，是在修改之前的程式碼
  （HEAD c33369f）上算出來的基準，這裡逐一比對；任何一個字改動都會被抓到
- validate_scoring_template()：verify_quote 必須是布林值、只適用缺失型

真實資料：引文案例用的是真實文件裡的句子（遮蔽後的 Branes.AI DD 報告、PSF EIM 公開文件），
以及 2026-09-29 實測時模型真的輸出過的誤觸發依據。
"""

import copy
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import veto_check
from generate_report import build_veto_banner_line, render_veto_section, resolve_veto_config
from scoring import validate_scoring_template
from veto_check import (
    MIN_QUOTE_CHARS,
    REASON_NO_QUOTE,
    REASON_QUOTE_NOT_IN_CONTEXT,
    VETO_RULES,
    _build_veto_system_prompt,
    detect_veto,
    detection_status,
    parse_verified_absence_response,
    summarize_veto_results,
    verify_quote_in_context,
)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    DEEP = json.load(_f)["深科技"]
NINE = json.load(open(_TEMPLATES_PATH, encoding="utf-8"))["九格"]
RULE = "智財歸屬不完整"
CONSEQUENCE = DEEP["veto_rules"][RULE]["consequence"]


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 五不合作完全不受影響：基準雜湊（在修改之前的 HEAD c33369f 上算出來的）
# ---------------------------------------------------------------------------

BASELINE_PROMPT_HASHES = {
    "人不明": "c8a740d40936a69c5ef33744c81d004ec304f82ef3258f21b08510466a9afc3a",
    "資源不實": "d33824c76048352a29a9e54db5086d3eac18805238890a7edaf1f706785d44ad",
    "權責不清": "ddb23464b86bda35c86c104f2d392a5a493cd8b48320c0ad601844a3f2bd6b98",
    "利益不明": "413665e077be619ee0ac95efcdc7a1395ca8a4d23fb8ac47545d8407b0dea482",
    "風險不揭露": "e33559696957439b97a3bcf34bda4f9ddcdfce32632c0743921c3cc0c51b4a33",
}
BASELINE_TEMPLATE_LEGACY_PROMPT = "05161ac3d04b9cfbd3b7f474822d998015e1b7db0ef51a29bacff10d7df6e078"
BASELINE_RENDER = {
    "not_triggered_only": "799c16857f791648b1065ee1658fc4c01966ee239424643ccb5d67e3697d9b74",
    "triggered_and_not": "e2ee4e0de4938a7639f3f514aca4598b4e19270729d0059b55a8398a662ab603",
    "multi": "a0175d9e7d6605d311f91181ecb5d743c75c4d24b70219dbe2642a064384f6bf",
}
BASELINE_BANNER = {
    "not_triggered_only": "50542b3306039b8712fb06ff5330f63faa91df3244e92c5504ea2931a12efd18",
    "triggered_and_not": "1f30d14afa366de16d9bd9b0b97d55772ac46790e013f79f055e3ca4239f8d74",
    "multi": "1f30d14afa366de16d9bd9b0b97d55772ac46790e013f79f055e3ca4239f8d74",
}
BASELINE_BANNER_EMPTY = "dc937b59892604f5a86ac96936cd7ff09e25f18ae6b758e8014a24c7fa039e91"
BASELINE_DEFAULT_CONFIG = "113a33835919deb2274fbe2d0c620a728ce352339c25b083bd49ac4d91a23965"

_ENTRY = {"question": "Q", "phenomenon": "P", "basis": "B", "suggested_action": "A", "heading": "H"}
_BASELINE_RESULTS = {
    "not_triggered_only": {"人不明": {"triggered": False}},
    "triggered_and_not": {"人不明": {"triggered": True, "entries": [_ENTRY]},
                          "資源不實": {"triggered": False, "entries": []}},
    "multi": {"人不明": {"triggered": True, "entries": [_ENTRY, dict(_ENTRY, question="Q2", heading=None)]},
              "權責不清": {"triggered": False}},
}


@pytest.mark.parametrize("rule", list(VETO_RULES))
def test_default_five_rule_prompts_are_byte_identical_to_baseline(rule):
    assert _sha(_build_veto_system_prompt(rule)) == BASELINE_PROMPT_HASHES[rule]


def test_template_legacy_prompt_builder_unchanged():
    assert _sha(_build_veto_system_prompt(RULE, DEEP["veto_rules"])) == BASELINE_TEMPLATE_LEGACY_PROMPT


@pytest.mark.parametrize("name", list(_BASELINE_RESULTS))
def test_default_render_and_banner_are_byte_identical_to_baseline(name):
    assert _sha(render_veto_section(_BASELINE_RESULTS[name])) == BASELINE_RENDER[name]
    assert _sha(str(build_veto_banner_line(_BASELINE_RESULTS[name]))) == BASELINE_BANNER[name]


def test_default_banner_empty_and_config_unchanged():
    assert _sha(str(build_veto_banner_line({}))) == BASELINE_BANNER_EMPTY
    cfg = resolve_veto_config(None)
    assert _sha(json.dumps({k: v for k, v in cfg.items() if k != "rules"}, ensure_ascii=False,
                           sort_keys=True)) == BASELINE_DEFAULT_CONFIG
    assert resolve_veto_config(NINE)["rules"] is None


# ---------------------------------------------------------------------------
# verify_quote_in_context()
# ---------------------------------------------------------------------------

# 真實文件（遮蔽後的 Branes.AI DD 報告 2.3 節表格）：版面把欄位用大量空白隔開
REAL_CONTEXT = (
    "[文件片段 4] Branes.AI 商業模式暨投資盡職調查報告 > 貳 公司定位與團隊結構 > 2.3 團隊結構風險\n"
    "  已具備                                            明顯缺口\n"
    "     軟體堆疊部分已建（宣稱 ~30%）                             團隊規模未揭露，資源配置不明\n"
    "[文件片段 5] 陸 估值合理性分析 > 6.2\n"
    " 稀釋計算透明度    USD 8M ÷ (USD 27M + USD 8M) = 22.9%，計算正確。惟未揭露 option pool 是否\n"
    "                                   已計入 pre-money；若未計入，實際稀釋將更高。\n"
)


@pytest.mark.parametrize("quote", [
    "團隊規模未揭露，資源配置不明",
    "團隊規模未揭露",
    "未揭露 option pool 是否已計入 pre-money；若未計入，實際稀釋將更高。",   # 原文在版面上換了行
    "惟未揭露 option pool 是否已計入 pre-money",
])
def test_verify_quote_accepts_verbatim_text_despite_layout(quote):
    ok, reason, missing = verify_quote_in_context(quote, REAL_CONTEXT)
    assert (ok, reason, missing) == (True, None, [])


def test_verify_quote_ignores_markdown_bold_on_either_side():
    ctx = "現狀：核心專利受讓人**仍為 Stillwater Supercomputing**。"
    assert verify_quote_in_context("核心專利受讓人仍為 Stillwater Supercomputing", ctx)[0] is True
    assert verify_quote_in_context("核心專利受讓人**仍為** Stillwater", "核心專利受讓人仍為 Stillwater")[0] is True


@pytest.mark.parametrize("quote", [
    "團隊規模並未公開，人力配置不清楚",          # 改寫
    "团队规模未揭露，资源配置不明",              # 簡體化
    "團隊規模未揭露，並且資源配置不明",          # 多插了字
    "文件中未提及該公司的團隊規模",              # 系統回答的說法，不在文件裡
])
def test_verify_quote_rejects_paraphrase_simplified_and_answer_text(quote):
    ok, reason, missing = verify_quote_in_context(quote, REAL_CONTEXT)
    assert ok is False
    assert reason == REASON_QUOTE_NOT_IN_CONTEXT
    assert missing


@pytest.mark.parametrize("quote", [None, "", "   ", 3, "未揭露"])
def test_verify_quote_rejects_missing_or_too_short(quote):
    ok, reason, missing = verify_quote_in_context(quote, REAL_CONTEXT)
    assert (ok, reason, missing) == (False, REASON_NO_QUOTE, [])


def test_verify_quote_length_boundary_is_normalized_characters():
    assert MIN_QUOTE_CHARS == 6
    ctx = "尚待確認事項一覽表"
    assert verify_quote_in_context("尚待確認事項", ctx)[0] is True            # 6 個字
    assert verify_quote_in_context("尚待確認事", ctx)[0] is False              # 5 個字
    assert verify_quote_in_context("尚 待 確 認 事 項", ctx)[0] is True        # 空白不算字數，仍是 6 個字


# ---------------------------------------------------------------------------
# parse_verified_absence_response()
# ---------------------------------------------------------------------------

def _raw(**fields):
    return json.dumps(fields, ensure_ascii=False)


def _parse(raw, context=REAL_CONTEXT):
    return parse_verified_absence_response(raw, context, CONSEQUENCE)


def test_parse_clear_and_silent_are_not_triggered():
    for status in ("clear", "silent"):
        r = _parse(_raw(status=status, quote=None, phenomenon=None, suggested_action=None))
        assert r["status"] == status and r["is_veto_triggered"] is False
        assert r["basis"] is None and r["parse_error"] is None and r["downgrade_reason"] is None


def test_parse_admitted_with_verifiable_quote_triggers_and_uses_quote_as_basis():
    r = _parse(_raw(status="admitted", quote="團隊規模未揭露，資源配置不明",
                    phenomenon="團隊規模與資源配置未揭露", suggested_action="補齊團隊資料前維持不可投資"))
    assert r["status"] == "admitted" and r["is_veto_triggered"] is True
    assert r["basis"] == "「團隊規模未揭露，資源配置不明」"
    assert r["quote"] == "團隊規模未揭露，資源配置不明"
    assert r["phenomenon"] == "團隊規模與資源配置未揭露"
    assert r["suggested_action"] == "補齊團隊資料前維持不可投資"
    assert r["parse_error"] is None and r["downgrade_reason"] is None


def test_parse_admitted_fills_defaults_instead_of_discarding_a_verified_quote():
    r = _parse(_raw(status="admitted", quote="團隊規模未揭露，資源配置不明"))
    assert r["is_veto_triggered"] is True
    assert r["phenomenon"] and CONSEQUENCE in r["suggested_action"]


@pytest.mark.parametrize("quote,reason", [
    ("團隊規模並未公開，人力配置不清楚", REASON_QUOTE_NOT_IN_CONTEXT),   # 改寫
    ("團隊规模未揭露，资源配置不明", REASON_QUOTE_NOT_IN_CONTEXT),       # 簡體
    (None, REASON_NO_QUOTE),
    ("", REASON_NO_QUOTE),
    ("未揭露", REASON_NO_QUOTE),                                          # 太短
])
def test_parse_admitted_with_unverifiable_quote_downgrades_to_silent(quote, reason):
    r = _parse(_raw(status="admitted", quote=quote, phenomenon="x", suggested_action="y"))
    assert r["status"] == "silent"
    assert r["is_veto_triggered"] is False
    assert r["downgrade_reason"] == reason
    assert r["basis"] is None and r["parse_error"] is None


def test_real_false_trigger_2026_09_29_is_downgraded():
    """2026-09-29 實測：PSF EIM 公開文件對智財完全沉默，現行判斷跑兩次都觸發，
    模型寫的『依據』引用的是問答系統自己的回答，不是文件。用真實的文件片段與那次
    模型真的輸出過的引文重現：新判斷必須把它降級成沉默。"""
    psf_context = (
        "[文件片段 1] PSF Enterprise Intelligence Matrix™（PSF EIM）\n"
        "PSF EIM 是：企業智慧基礎設施（Enterprise Intelligence Infrastructure）。\n"
        "[文件片段 2] PSF EIM 五大產品矩陣\n"
        "① PSF EIM Decision Matrix™（原 CEO AI HUB）　② PSF EIM Service Matrix™（原 Black Concierge）\n"
    )
    for answer_sentence in ("文件中未提及", "無法從現有資訊推斷技術智財的持有者",
                            "提供的文件片段中，僅描述了PSF EIM的產品結構、核心價值主張、定位與功能模組，但未說明其技術智財"):
        r = _parse(_raw(status="admitted", quote=answer_sentence, phenomenon="智財歸屬未揭露",
                        suggested_action="補充智財文件"), context=psf_context)
        assert r["status"] == "silent" and r["is_veto_triggered"] is False, answer_sentence


def test_real_gate3_positive_quote_from_document_still_triggers():
    """Branes.AI 報告 Gate 3 那段不是用『未揭露』這類字，而是『要求取得…移轉文件』；
    這類（要求對方取得＝現在還沒有）算承認不完整，引文在文件裡逐字存在就必須觸發。"""
    ctx = ("[文件片段 2] 玖 投資前必要前置條件（Gate 查核清單） > Gate 3 — 完整智財清單與歸屬確認\n"
           "    取得 22+ 專利之完整清單、claim chart，以及 Stillwater → Branes 的移轉文件\n"
           "    現狀：公開檢索僅見約 11 筆，且集中於單一專利家族；核心專利受讓人仍為 Stillwater Supercomputing。\n")
    r = _parse(_raw(status="admitted", quote="取得 22+ 專利之完整清單、claim chart，以及 Stillwater → Branes 的移轉文件",
                    phenomenon="智財移轉文件尚未取得", suggested_action="取得後再議"), context=ctx)
    assert r["status"] == "admitted" and r["is_veto_triggered"] is True


@pytest.mark.parametrize("raw", [
    "not json",
    '["admitted"]',
    '{"quote": "團隊規模未揭露，資源配置不明"}',                      # 沒有 status
    '{"status": "yes", "quote": "團隊規模未揭露，資源配置不明"}',       # status 不合法
    '{"status": null}',
    '{"status": "admitted", "quote": "團隊規模未揭露，資源配置不明"',   # 物件不完整
    "",
])
def test_parse_broken_output_is_undetermined_not_clear(raw):
    r = _parse(raw)
    assert r["status"] == "undetermined"
    assert r["is_veto_triggered"] is False
    assert r["parse_error"]


@pytest.mark.parametrize("raw", [
    '{"status": "ADMITTED ", "quote": "團隊規模未揭露，資源配置不明"}',                 # 大小寫與空白
    '{"status": "admitted", "quote": "團隊規模未揭露，資源配置不明"}}',                 # 完整物件後多一個括號（驗證時的真實輸出形狀）
    '<think>先看片段</think>\n{"status": "admitted", "quote": "團隊規模未揭露，資源配置不明"}',
    '```json\n{"status": "admitted", "quote": "團隊規模未揭露，資源配置不明"}\n```',
])
def test_parse_tolerates_case_trailing_brace_think_and_fence(raw):
    r = _parse(raw)
    assert r["status"] == "admitted" and r["is_veto_triggered"] is True


# ---------------------------------------------------------------------------
# detection_status() / summarize_veto_results()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("detection,expected", [
    ({"status": "admitted", "is_veto_triggered": True}, "triggered"),
    ({"status": "silent"}, "silent"),
    ({"status": "clear"}, "clear"),
    ({"status": "undetermined", "parse_error": "x"}, "undetermined"),
    ({"is_veto_triggered": True}, "triggered"),                          # 舊格式：觸發
    ({"is_veto_triggered": False}, "clear"),                             # 舊格式：未觸發
    ({"is_veto_triggered": False, "parse_error": "壞掉"}, "undetermined"),   # 舊格式：解析失敗不再當成未觸發
    ({}, "clear"),
])
def test_detection_status(detection, expected):
    assert detection_status(detection) == expected


def _entry(rule, state, question="Q"):
    det = {
        "triggered": {"status": "admitted", "is_veto_triggered": True, "phenomenon": "p", "basis": "「b」", "suggested_action": "a"},
        "silent": {"status": "silent", "is_veto_triggered": False, "downgrade_reason": "原因S"},
        "clear": {"status": "clear", "is_veto_triggered": False},
        "undetermined": {"status": "undetermined", "is_veto_triggered": False, "parse_error": "解析失敗X"},
    }[state]
    return {"question": question, "rule": rule, "detection": det, "heading": "H"}


@pytest.mark.parametrize("states,expected", [
    (["clear"], "clear"),
    (["silent"], "silent"),
    (["clear", "silent"], "silent"),
    (["silent", "undetermined"], "undetermined"),
    (["clear", "silent", "undetermined", "triggered"], "triggered"),
    (["triggered", "silent"], "triggered"),
])
def test_summarize_rule_status_precedence(states, expected):
    entries = [_entry(RULE, s, question=f"Q{i}") for i, s in enumerate(states)]
    result = summarize_veto_results(entries, rules=DEEP["veto_rules"])[RULE]
    assert result["status"] == expected
    assert result["triggered"] is (expected == "triggered")


def test_summarize_keeps_each_kind_of_entry_with_its_reason():
    entries = [_entry(RULE, "triggered", "Q1"), _entry(RULE, "silent", "Q2"), _entry(RULE, "undetermined", "Q3")]
    r = summarize_veto_results(entries, rules=DEEP["veto_rules"])[RULE]
    assert [e["question"] for e in r["entries"]] == ["Q1"]
    assert r["silent_entries"] == [{"question": "Q2", "heading": "H", "reason": "原因S"}]
    assert r["undetermined_entries"] == [{"question": "Q3", "heading": "H", "reason": "解析失敗X"}]


def test_summarize_truncates_long_parse_error_reason():
    e = _entry(RULE, "undetermined")
    e["detection"]["parse_error"] = "x" * 500
    r = summarize_veto_results([e], rules=DEEP["veto_rules"])[RULE]
    assert len(r["undetermined_entries"][0]["reason"]) == 160


def test_summarize_legacy_parse_failure_is_undetermined_not_clear():
    """舊格式（五不合作）：JSON 壞掉時 parse_veto_response() 回傳 未觸發＋parse_error。
    以前報告會顯示 ✅ 未觸發（假放心），現在算無法判定。觸發與否的判斷本身沒有改。"""
    entries = [{"question": "Q", "rule": "人不明", "heading": None,
                "detection": {"is_veto_triggered": False, "parse_error": "veto判斷結果無法解析成 JSON"}}]
    r = summarize_veto_results(entries)["人不明"]
    assert r["status"] == "undetermined" and r["triggered"] is False


def test_summarize_legacy_normal_results_keep_old_shape_and_meaning():
    entries = [
        {"question": "Q1", "rule": "人不明", "heading": "h",
         "detection": {"is_veto_triggered": True, "phenomenon": "p", "basis": "b", "suggested_action": "a"}},
        {"question": "Q2", "rule": "資源不實", "heading": None, "detection": {"is_veto_triggered": False}},
    ]
    r = summarize_veto_results(entries)
    assert r["人不明"]["triggered"] is True and r["人不明"]["status"] == "triggered"
    assert r["資源不實"]["triggered"] is False and r["資源不實"]["status"] == "clear"


# ---------------------------------------------------------------------------
# 報告顯示
# ---------------------------------------------------------------------------

def _results(state, question="PSF EIM 的技術智財是誰持有？"):
    return summarize_veto_results([_entry(RULE, state, question)], rules=DEEP["veto_rules"])


CFG = resolve_veto_config(DEEP)
SILENT_LINE = "- ⏸ **智財歸屬不完整**：檢索到的片段未提及此主題，需人工確認"


def _rule_lines(section: str):
    """章節裡「規則結果」的那幾行（- 開頭的項目與 ### 標題）。章節開頭的驗證邊界說明
    會用文字描述各種狀態（包括「✅ 未觸發」這幾個字），不算規則結果。"""
    return [l for l in section.splitlines() if l.startswith("- ") or l.startswith("### ")]


def test_render_silent_shows_pause_not_checkmark():
    out = render_veto_section(_results("silent"), CFG)
    assert SILENT_LINE in out
    assert "  - 「PSF EIM 的技術智財是誰持有？」（原因S）" in out
    assert _rule_lines(out) == [SILENT_LINE]        # 只有這一行，沒有 ✅ 也沒有 🚫


def test_render_undetermined_shows_warning_not_checkmark():
    out = render_veto_section(_results("undetermined"), CFG)
    assert "- ⚠️ **智財歸屬不完整**：無法判定（模型輸出無法解析），需人工確認" in out
    assert "  - 「PSF EIM 的技術智財是誰持有？」（解析失敗X）" in out
    assert _rule_lines(out) == ["- ⚠️ **智財歸屬不完整**：無法判定（模型輸出無法解析），需人工確認"]


def test_render_clear_still_shows_checkmark():
    assert "- ✅ **智財歸屬不完整**：未觸發" in render_veto_section(_results("clear"), CFG)


def test_render_triggered_still_lists_quote_as_basis():
    out = render_veto_section(_results("triggered"), CFG)
    assert "### 🚫 智財歸屬不完整" in out and "**依據**：「b」" in out


def test_banner_states_and_precedence():
    label = CFG["label"]
    assert build_veto_banner_line(_results("clear"), CFG) == f"**{label}**：✅ 未觸發"
    assert build_veto_banner_line(_results("silent"), CFG) == f"**{label}**：⏸ 檢索到的片段未提及「智財歸屬不完整」——需人工確認"
    assert build_veto_banner_line(_results("undetermined"), CFG) == f"**{label}**：⚠️ 無法判定「智財歸屬不完整」——需人工確認"
    assert "🚫 觸發「智財歸屬不完整」" in build_veto_banner_line(_results("triggered"), CFG)
    two = {**_results("silent"), "另一條": {"triggered": False, "entries": [], "status": "undetermined",
                                           "silent_entries": [], "undetermined_entries": []}}
    assert "⚠️ 無法判定「另一條」" in build_veto_banner_line(two, CFG)   # 無法判定優先於文件未提及


def test_banner_and_section_never_show_checkmark_for_unconfirmed_states():
    for state in ("silent", "undetermined"):
        assert "✅" not in build_veto_banner_line(_results(state), CFG)
        assert not any("✅" in l or "🚫" in l for l in _rule_lines(render_veto_section(_results(state), CFG)))


def test_legacy_results_without_status_keep_old_display():
    assert "- ✅ **人不明**：未觸發" in render_veto_section({"人不明": {"triggered": False}})
    assert build_veto_banner_line({"人不明": {"triggered": False}}) == "**不合作紅線**：✅ 未觸發"


def test_template_section_note_mentions_verification_and_default_note_does_not():
    assert "逐字複製、程式已驗證的引文" in CFG["section_note"]
    assert "⏸" in CFG["section_note"] and "需人工確認" in CFG["section_note"]
    assert "逐字複製" not in resolve_veto_config(None)["section_note"]


# ---------------------------------------------------------------------------
# detect_veto()：依規則旗標分流
# ---------------------------------------------------------------------------

def _fake_http(content, sent):
    def fake(method, url, payload, timeout=None):
        sent.append(payload)
        return 200, {"choices": [{"message": {"content": content}}]}
    return fake


def test_detect_veto_with_flag_uses_three_state_prompt_and_verifies_quote(monkeypatch):
    sent = []
    monkeypatch.setattr(veto_check, "http_json", _fake_http(
        _raw(status="admitted", quote="團隊規模未揭露，資源配置不明", phenomenon="p", suggested_action="a"), sent))
    result = detect_veto(RULE, "問題", "回答", REAL_CONTEXT, "u", "m", rules=DEEP["veto_rules"])
    assert result["status"] == "admitted" and result["is_veto_triggered"] is True and result["rule"] == RULE
    payload = sent[0]
    assert "三選一" in payload["messages"][0]["content"]
    assert "不是文件承認資訊不完整" in payload["messages"][0]["content"]
    assert (payload["temperature"], payload["max_tokens"]) == (0, 2048)
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert "檢索到的文件片段全文" in payload["messages"][1]["content"]


def test_detect_veto_with_flag_downgrades_quote_that_is_only_in_the_answer(monkeypatch):
    monkeypatch.setattr(veto_check, "http_json", _fake_http(
        _raw(status="admitted", quote="文件中未提及智財歸屬", phenomenon="p", suggested_action="a"), []))
    result = detect_veto(RULE, "問題", "文件中未提及智財歸屬", REAL_CONTEXT, "u", "m", rules=DEEP["veto_rules"])
    assert result["status"] == "silent" and result["is_veto_triggered"] is False
    assert result["downgrade_reason"] == REASON_QUOTE_NOT_IN_CONTEXT


@pytest.mark.parametrize("http_result", [(500, {"error": "x"}), (200, {"unexpected": 1})])
def test_detect_veto_with_flag_model_failure_is_undetermined(monkeypatch, http_result):
    monkeypatch.setattr(veto_check, "http_json", lambda *a, **k: http_result)
    result = detect_veto(RULE, "問題", "回答", REAL_CONTEXT, "u", "m", rules=DEEP["veto_rules"])
    assert result["status"] == "undetermined" and result["rule"] == RULE and result["parse_error"]


def test_detect_veto_without_flag_uses_original_path_and_dict_shape(monkeypatch):
    sent = []
    monkeypatch.setattr(veto_check, "http_json", _fake_http(
        '{"is_veto_triggered": true, "phenomenon": "p", "basis": "b", "suggested_action": "a"}', sent))
    result = detect_veto("利益不明", "問題", "回答", REAL_CONTEXT, "u", "m")
    assert sent[0]["messages"][0]["content"] == _build_veto_system_prompt("利益不明")
    assert _sha(sent[0]["messages"][0]["content"]) == BASELINE_PROMPT_HASHES["利益不明"]
    assert result == {"is_veto_triggered": True, "phenomenon": "p", "basis": "b", "suggested_action": "a",
                      "parse_error": None, "rule": "利益不明"}
    assert "status" not in result


def test_detect_veto_template_rule_with_flag_off_uses_original_path(monkeypatch):
    rules = copy.deepcopy(DEEP["veto_rules"])
    rules[RULE]["verify_quote"] = False
    sent = []
    monkeypatch.setattr(veto_check, "http_json", _fake_http('{"is_veto_triggered": false}', sent))
    result = detect_veto(RULE, "問題", "回答", REAL_CONTEXT, "u", "m", rules=rules)
    assert sent[0]["messages"][0]["content"] == _build_veto_system_prompt(RULE, rules)
    assert "status" not in result


def test_only_the_gate3_template_rule_has_the_flag():
    assert DEEP["veto_rules"][RULE]["verify_quote"] is True
    assert not any("verify_quote" in r for r in VETO_RULES.values())
    assert "veto_rules" not in NINE


# ---------------------------------------------------------------------------
# 設定檢查
# ---------------------------------------------------------------------------

def test_real_template_with_flag_passes_validation():
    assert validate_scoring_template(DEEP) == []


@pytest.mark.parametrize("mutator,keyword", [
    (lambda t: t["veto_rules"][RULE].__setitem__("verify_quote", "yes"), "布林"),
    (lambda t: t["veto_rules"][RULE].__setitem__("verify_quote", 1), "布林"),
    (lambda t: t["veto_rules"][RULE].__setitem__("mode", "矛盾"), "缺失"),
])
def test_validation_rejects_bad_verify_quote(mutator, keyword):
    broken = copy.deepcopy(DEEP)
    mutator(broken)
    errors = validate_scoring_template(broken)
    assert any("verify_quote" in e and keyword in e for e in errors), errors


def test_validation_allows_flag_false_on_contradiction_rule():
    t = copy.deepcopy(DEEP)
    t["veto_rules"][RULE].update({"mode": "矛盾", "verify_quote": False})
    assert validate_scoring_template(t) == []
