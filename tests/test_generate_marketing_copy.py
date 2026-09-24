"""
測試重點：
- parse_marketing_copy_content()：驗證三種樣板各自的必要欄位檢查
  （重點條列型/故事敘述型/數據導向型），title/cta共用的空白檢查，
  以及JSON解析失敗時的處理
- render_marketing_html()：驗證三種樣板各自套版正確、html.escape()有
  正確套用（避免生成內容裡的<,>,&弄壞版面）、CSS裡的花括號不會被
  str.format()誤判成佔位符
- render_marketing_plaintext()：驗證三種樣板各自轉出的純文字內容正確，
  不含HTML標籤
- load_marketing_templates()：檔案不存在時die()
- build_warning_banner()/flag_map()/render_marketing_html()：兩級警示——紅
  （無依據、整行漏檢、格式損壞、查核失敗、跳過檢查）跟黃（循環引用、
  依據無法自動核實）在橫幅與行內標記上要看得出差別；全部有依據時沒有橫幅
- tone_gate_reason()/format_tone_report()：忠實度有任何紅/黃、失敗或被
  跳過時，語氣評分不顯示分數
- main() 接線：monkeypatch 檢索與模型呼叫，確認檢查結果真的決定 HTML
  橫幅、結束碼，以及事實檢查沒過時根本不呼叫語氣評分器
- 真實模板結構讀 scripts/data/marketing_templates.json，不虛構一套
  不相容的模板結構——跟 tests/test_report_helpers.py 處理
  scoring_templates.json 同一個原則
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from generate_marketing_copy import (
    TONE_GATED_MESSAGE,
    apply_warning_banner,
    build_warning_banner,
    flag_map,
    format_tone_report,
    load_marketing_templates,
    tone_gate_reason,
    parse_marketing_copy_content,
    render_marketing_html,
    render_marketing_plaintext,
)
from marketing_faithfulness import ALERT_RED, ALERT_YELLOW

TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "data", "marketing_templates.json",
)


@pytest.fixture(scope="module")
def real_templates():
    with open(TEMPLATES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# parse_marketing_copy_content()
# ---------------------------------------------------------------------------

def test_parse_marketing_copy_content_bullets_valid():
    raw = json.dumps({"title": "標題", "bullets": ["賣點一", "賣點二"], "cta": "立即聯絡"})
    result = parse_marketing_copy_content(raw, "重點條列型")
    assert result["parse_error"] is None
    assert result["content"]["bullets"] == ["賣點一", "賣點二"]


def test_parse_marketing_copy_content_story_valid():
    raw = json.dumps({"title": "標題", "story_paragraphs": ["段一", "段二"], "cta": "聯絡我們"})
    result = parse_marketing_copy_content(raw, "故事敘述型")
    assert result["parse_error"] is None
    assert result["content"]["story_paragraphs"] == ["段一", "段二"]


def test_parse_marketing_copy_content_stats_valid():
    raw = json.dumps({
        "title": "標題",
        "stats": [{"number": "99%", "label": "滿意度"}],
        "body_paragraphs": ["補充說明"],
        "cta": "預約諮詢",
    })
    result = parse_marketing_copy_content(raw, "數據導向型")
    assert result["parse_error"] is None
    assert result["content"]["stats"] == [{"number": "99%", "label": "滿意度"}]


@pytest.mark.parametrize("raw,description", [
    ("not json", "純文字"),
    ('{"title": "X"', "JSON語法不完整"),
    ('["title", "X"]', "頂層是清單不是物件"),
])
def test_parse_marketing_copy_content_json_error(raw, description):
    result = parse_marketing_copy_content(raw, "重點條列型")
    assert result["parse_error"] is not None, description
    assert result["content"] is None, description


@pytest.mark.parametrize("layout,payload,description", [
    ("重點條列型", {"title": "標題", "cta": "聯絡"}, "缺bullets"),
    ("重點條列型", {"title": "標題", "bullets": [], "cta": "聯絡"}, "bullets是空清單"),
    ("重點條列型", {"title": "標題", "bullets": ["", ""], "cta": "聯絡"}, "bullets裡是空字串"),
    ("重點條列型", {"title": "標題", "bullets": "不是清單", "cta": "聯絡"}, "bullets不是清單"),
    ("重點條列型", {"title": "", "bullets": ["賣點"], "cta": "聯絡"}, "title是空字串"),
    ("重點條列型", {"title": "標題", "bullets": ["賣點"], "cta": ""}, "cta是空字串"),
    ("故事敘述型", {"title": "標題", "cta": "聯絡"}, "缺story_paragraphs"),
    ("故事敘述型", {"title": "標題", "story_paragraphs": [], "cta": "聯絡"}, "story_paragraphs是空清單"),
    ("數據導向型", {"title": "標題", "body_paragraphs": ["說明"], "cta": "聯絡"}, "缺stats"),
    ("數據導向型", {"title": "標題", "stats": [{"number": "1"}], "body_paragraphs": ["說明"], "cta": "聯絡"}, "stats缺label"),
    ("數據導向型", {"title": "標題", "stats": [{"number": "1", "label": "L"}], "cta": "聯絡"}, "缺body_paragraphs"),
])
def test_parse_marketing_copy_content_invalid_fields(layout, payload, description):
    result = parse_marketing_copy_content(json.dumps(payload), layout)
    assert result["parse_error"] is not None, description
    assert result["content"] is None, description


# ---------------------------------------------------------------------------
# render_marketing_html()：用真實模板結構，不虛構
# ---------------------------------------------------------------------------

def test_render_marketing_html_bullets_layout(real_templates):
    content = {"title": "測試標題", "bullets": ["賣點一", "賣點二"], "cta": "立即聯絡"}
    html = render_marketing_html("重點條列型", real_templates["重點條列型"], content)
    assert "<li>賣點一</li>" in html
    assert "<li>賣點二</li>" in html
    assert "測試標題" in html
    assert "立即聯絡" in html
    assert "<style>" in html and real_templates["重點條列型"]["css"] in html


def test_render_marketing_html_story_layout(real_templates):
    content = {"title": "測試標題", "story_paragraphs": ["段一", "段二"], "cta": "聯絡我們"}
    html = render_marketing_html("故事敘述型", real_templates["故事敘述型"], content)
    assert "<p>段一</p>" in html
    assert "<p>段二</p>" in html


def test_render_marketing_html_stats_layout(real_templates):
    content = {
        "title": "測試標題",
        "stats": [{"number": "99%", "label": "滿意度"}],
        "body_paragraphs": ["補充說明"],
        "cta": "預約諮詢",
    }
    html = render_marketing_html("數據導向型", real_templates["數據導向型"], content)
    assert "mc-stat-num\">99%</div>" in html
    assert "mc-stat-label\">滿意度</div>" in html
    assert "<p>補充說明</p>" in html


def test_render_marketing_html_escapes_html_special_chars(real_templates):
    content = {"title": "<b>標題</b> & 測試", "bullets": ["賣點<script>"], "cta": "聯絡"}
    html = render_marketing_html("重點條列型", real_templates["重點條列型"], content)
    assert "<script>" not in html
    assert "&lt;b&gt;" in html and "&amp;" in html


def test_render_marketing_html_css_braces_do_not_break_format(real_templates):
    # CSS本身包含大量花括號——確認 str.format() 不會誤把 css 值裡的 {} 當
    # 成 html_template 自己的佔位符去重新解析（如果會，這裡會直接拋
    # KeyError/IndexError，不需要額外斷言內容）
    content = {"title": "標題", "bullets": ["賣點"], "cta": "聯絡"}
    render_marketing_html("重點條列型", real_templates["重點條列型"], content)  # 不拋例外就算過


def test_render_marketing_html_unknown_layout_raises(real_templates):
    with pytest.raises(ValueError):
        render_marketing_html("不存在的樣板", real_templates["重點條列型"], {"title": "x", "cta": "y"})


# ---------------------------------------------------------------------------
# render_marketing_plaintext()：不含HTML標籤
# ---------------------------------------------------------------------------

def test_render_marketing_plaintext_bullets_layout():
    content = {"title": "標題", "bullets": ["賣點一", "賣點二"], "cta": "聯絡"}
    text = render_marketing_plaintext("重點條列型", content)
    assert text == "標題\n賣點一\n賣點二\n聯絡"
    assert "<" not in text


def test_render_marketing_plaintext_stats_layout():
    content = {
        "title": "標題",
        "stats": [{"number": "99%", "label": "滿意度"}],
        "body_paragraphs": ["補充"],
        "cta": "聯絡",
    }
    text = render_marketing_plaintext("數據導向型", content)
    assert text == "標題\n99% 滿意度\n補充\n聯絡"


def test_render_marketing_plaintext_unknown_layout_raises():
    with pytest.raises(ValueError):
        render_marketing_plaintext("不存在的樣板", {"title": "x", "cta": "y"})


# ---------------------------------------------------------------------------
# load_marketing_templates()
# ---------------------------------------------------------------------------

def test_load_marketing_templates_missing_file_dies(tmp_path):
    with pytest.raises(SystemExit):
        load_marketing_templates(str(tmp_path / "nope.json"))


def test_load_marketing_templates_real_file_has_three_layouts():
    templates = load_marketing_templates(TEMPLATES_PATH)
    assert set(templates) == {"重點條列型", "故事敘述型", "數據導向型"}
    for layout in templates.values():
        assert "css" in layout and "html_template" in layout


# ---------------------------------------------------------------------------
# build_warning_banner() / apply_warning_banner()：兩級警示
# ---------------------------------------------------------------------------

def _faith(claims=(), malformed=0, uncovered=(), parse_error=None):
    return {"claims": None if parse_error else list(claims), "malformed_count": malformed,
            "uncovered_lines": list(uncovered), "parse_error": parse_error}


def _unsupported(claim, alert=ALERT_RED, reason="文件沒提到"):
    return {"claim": claim, "kind": "fact", "supported": False, "evidence": None,
            "reason_unsupported": reason, "alert": alert}


def _supported(claim):
    return {"claim": claim, "kind": "fact", "supported": True, "evidence": "有",
            "reason_unsupported": None, "alert": None}


RHETORIC = {"claim": "迎向未來", "kind": "rhetoric", "supported": None,
            "evidence": None, "reason_unsupported": None, "alert": None}
RED_BORDER = "#dc2626"
YELLOW_BORDER = "#ca8a04"


def test_banner_absent_when_all_facts_supported():
    assert build_warning_banner(_faith([_supported("A"), RHETORIC]), skipped=False) == ""


def test_banner_red_lists_unsupported_claim_with_reason():
    banner = build_warning_banner(
        _faith([_supported("有依據的主張甲"), _unsupported("服務超過500家企業")]), skipped=False)
    assert "服務超過500家企業" in banner and "文件沒提到" in banner
    assert "有依據的主張甲" not in banner
    assert "紅色｜無依據（1）" in banner and "黃色" not in banner
    assert RED_BORDER in banner and YELLOW_BORDER not in banner
    assert "⚠ 警告" in banner


def test_banner_yellow_only_is_visually_distinct():
    banner = build_warning_banner(
        _faith([_unsupported("整合五大產品矩陣", alert=ALERT_YELLOW, reason="循環引用")]), skipped=False)
    assert "黃色｜依據無法自動核實（1）" in banner and "紅色" not in banner
    assert YELLOW_BORDER in banner and RED_BORDER not in banner
    assert "dashed" in banner
    assert "? 注意" in banner and "⚠ 警告" not in banner


def test_banner_red_and_yellow_both_listed_red_headline_wins():
    banner = build_warning_banner(_faith([
        _unsupported("效率提升300%"), _unsupported("整合五大產品矩陣", alert=ALERT_YELLOW),
    ]), skipped=False)
    assert "紅色｜無依據（1）" in banner and "黃色｜依據無法自動核實（1）" in banner
    assert banner.index("紅色｜") < banner.index("黃色｜")
    assert "⚠ 警告" in banner


def test_banner_uncovered_lines_are_red():
    banner = build_warning_banner(_faith([_supported("A")], uncovered=["…的終極方案"]), skipped=False)
    assert "紅色｜無依據（1）" in banner and "…的終極方案" in banner and "檢查器整行漏掉" in banner


def test_banner_for_malformed_items():
    banner = build_warning_banner(_faith([_supported("A")], malformed=2), skipped=False)
    assert "另有 2 條" in banner and RED_BORDER in banner


def test_banner_when_check_failed():
    banner = build_warning_banner(_faith(parse_error="被截斷"), skipped=False)
    assert "忠實度檢查失敗" in banner and "被截斷" in banner and RED_BORDER in banner


def test_banner_when_check_skipped():
    banner = build_warning_banner(None, skipped=True)
    assert "未經忠實度檢查" in banner and RED_BORDER in banner


def test_banner_escapes_claim_text():
    banner = build_warning_banner(_faith([_unsupported("<script>x</script>")]), skipped=False)
    assert "<script>" not in banner and "&lt;script&gt;" in banner


def test_apply_warning_banner_goes_right_after_body():
    html = "<html><head></head><body><h1>T</h1></body></html>"
    out = apply_warning_banner(html, "<div>WARN</div>")
    assert out == "<html><head></head><body><div>WARN</div><h1>T</h1></body></html>"


def test_apply_warning_banner_noop_when_empty():
    html = "<html><body>x</body></html>"
    assert apply_warning_banner(html, "") == html


# ---------------------------------------------------------------------------
# flag_map() / render_marketing_html() 的兩級標記
# ---------------------------------------------------------------------------

def test_flag_map_levels_and_uncovered_lines_red():
    faith = _faith([_supported("A"), _unsupported("紅的"), _unsupported("黃的", alert=ALERT_YELLOW)],
                   uncovered=["漏掉的行"])
    assert flag_map(faith) == {"紅的": ALERT_RED, "黃的": ALERT_YELLOW, "漏掉的行": ALERT_RED}


def test_flag_map_empty_when_skipped_or_failed():
    assert flag_map(None) == {}
    assert flag_map(_faith(parse_error="壞")) == {}


def test_render_html_red_and_yellow_marks_and_plain_title_tag(real_templates):
    content = {"title": "終極方案", "bullets": ["賣點一", "服務超過500家企業，值得信賴", "整合五大產品矩陣"],
               "cta": "聯絡"}
    html = render_marketing_html("重點條列型", real_templates["重點條列型"], content, flagged={
        "服務超過500家企業": ALERT_RED, "終極方案": ALERT_RED, "整合五大產品矩陣": ALERT_YELLOW,
    })
    assert "⚠ 無依據：服務超過500家企業，值得信賴</mark></li>" in html
    assert "? 待核實：整合五大產品矩陣</mark></li>" in html
    assert "dashed #ca8a04" in html and "solid #dc2626" in html
    assert "<li>賣點一</li>" in html
    assert "<title>終極方案</title>" in html
    assert '<h1 class="mc-title"><mark' in html


def test_render_html_red_wins_when_element_hits_both_levels(real_templates):
    content = {"title": "T", "bullets": ["甲乙丙丁"], "cta": "C"}
    html = render_marketing_html("重點條列型", real_templates["重點條列型"], content,
                                 flagged={"甲乙": ALERT_YELLOW, "丙丁": ALERT_RED})
    assert "⚠ 無依據：甲乙丙丁" in html and "待核實" not in html


def test_render_html_stat_block_flagged_by_number_plus_label(real_templates):
    # 檢查器看到的是「100+ 企業智慧協作案例」一整行，HTML 裡數字跟說明分在兩格
    content = {"title": "T", "stats": [{"number": "100+", "label": "企業智慧協作案例"},
                                       {"number": "5", "label": "大產品矩陣"}],
               "body_paragraphs": ["說明"], "cta": "C"}
    html = render_marketing_html("數據導向型", real_templates["數據導向型"], content,
                                 flagged={"100+ 企業智慧協作案例": ALERT_RED})
    assert '<div class="mc-stat" style="background:#fee2e2;outline:2px solid #dc2626;">' in html
    assert "⚠ 無依據：企業智慧協作案例" in html
    assert '<div class="mc-stat"><div class="mc-stat-num">5</div>' in html


def test_render_html_no_marks_without_flags(real_templates):
    content = {"title": "標題", "bullets": ["賣點"], "cta": "聯絡"}
    html = render_marketing_html("重點條列型", real_templates["重點條列型"], content)
    assert "<mark" not in html


# ---------------------------------------------------------------------------
# 語氣評分綁定忠實度檢查
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("faith,skipped,expect_gated", [
    (_faith([_supported("A"), RHETORIC]), False, False),
    (_faith([_unsupported("A")]), False, True),
    (_faith([_unsupported("A", alert=ALERT_YELLOW)]), False, True),
    (_faith([_supported("A")], uncovered=["漏掉"]), False, True),
    (_faith([_supported("A")], malformed=1), False, True),
    (_faith(parse_error="壞"), False, True),
    (None, True, True),
])
def test_tone_gate_reason(faith, skipped, expect_gated):
    reason = tone_gate_reason(faith, skipped)
    if expect_gated:
        assert reason.startswith(TONE_GATED_MESSAGE)
    else:
        assert reason is None


def test_format_tone_report_shows_total():
    tone = {"scores": {"開頭吸引力": {"score": 4, "rationale": "r"}, "具體性": {"score": 3, "rationale": "r"}},
            "parse_error": None}
    lines = format_tone_report(tone)
    assert lines[-1] == "合計：7/10"


def test_format_tone_report_failure_has_no_total():
    lines = format_tone_report({"scores": None, "parse_error": "壞"})
    assert lines == ["[評分失敗] 壞"] and not any("合計" in l for l in lines)


# ---------------------------------------------------------------------------
# main() 接線：檢查結果真的決定 HTML 橫幅、結束碼、以及語氣評分要不要算
# （monkeypatch 檢索與模型呼叫，不打 Qdrant/vLLM）
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, faith_result, extra_args=()):
    import generate_marketing_copy as gmc
    content = {"title": "標題", "bullets": ["服務超過500家企業"], "cta": "聯絡"}
    tone_calls = []
    monkeypatch.setattr(gmc, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(gmc, "embed_one", lambda text: [0.0])
    monkeypatch.setattr(gmc, "search", lambda *a, **k: [{"payload": {"heading_path": "h", "body": "b"}}])
    monkeypatch.setattr(gmc, "generate_marketing_copy_content",
                        lambda *a, **k: {"content": content, "parse_error": None})
    monkeypatch.setattr(gmc, "check_marketing_claims", lambda *a, **k: faith_result)

    def fake_tone(*a, **k):
        tone_calls.append(1)
        return {"scores": {"開頭吸引力": {"score": 5, "rationale": "r"}}, "parse_error": None}

    monkeypatch.setattr(gmc, "score_marketing_tone", fake_tone)
    out = tmp_path / "copy.html"
    monkeypatch.setattr(sys, "argv", ["generate_marketing_copy.py", "--collection", "c", "--brief", "b",
                                      "--layout", "重點條列型", "--output", str(out), *extra_args])
    code = 0
    try:
        gmc.main()
    except SystemExit as e:
        code = e.code
    return code, out.read_text(encoding="utf-8"), tone_calls


def test_main_red_alert_banner_exit3_and_no_tone_call(monkeypatch, tmp_path, capsys):
    code, html, tone_calls = _run_main(monkeypatch, tmp_path, _faith([_unsupported("服務超過500家企業")]))
    assert code == 3
    assert "紅色｜無依據" in html and "⚠ 無依據：服務超過500家企業" in html
    assert tone_calls == []
    out = capsys.readouterr().out
    assert TONE_GATED_MESSAGE in out and "合計" not in out


def test_main_yellow_alert_also_exit3_and_no_tone_call(monkeypatch, tmp_path, capsys):
    faith = _faith([_unsupported("服務超過500家企業", alert=ALERT_YELLOW)])
    code, html, tone_calls = _run_main(monkeypatch, tmp_path, faith)
    assert code == 3 and "黃色｜依據無法自動核實" in html and tone_calls == []
    assert TONE_GATED_MESSAGE in capsys.readouterr().out


def test_main_skip_quality_check_banner_exit3_no_tone(monkeypatch, tmp_path, capsys):
    code, html, tone_calls = _run_main(monkeypatch, tmp_path, None, ["--skip-quality-check"])
    assert code == 3 and "未經忠實度檢查" in html and tone_calls == []
    assert TONE_GATED_MESSAGE in capsys.readouterr().out


def test_main_clean_copy_no_banner_exit0_tone_with_total(monkeypatch, tmp_path, capsys):
    code, html, tone_calls = _run_main(monkeypatch, tmp_path, _faith([_supported("服務超過500家企業")]))
    assert code == 0 and "<mark" not in html and "警告" not in html
    assert tone_calls == [1]
    out = capsys.readouterr().out
    assert "合計：5/5" in out and TONE_GATED_MESSAGE not in out
