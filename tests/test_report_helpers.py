"""
測試重點：
- filter_citations() 測試相關度門檻值0.6的邊界情況與空列表情境
- slugify_heading() 測試中文/特殊符號處理與slug衝突現象
- load_questions() 的清除語法（`## group: none`／`## dimension: none`／
  `## veto: none`）：清除後回到 None、清除不是永久關閉（清除後可以重新
  宣告）——這是整合驗證（五種能力同時啟用）第一次跑跨主題問題清單時
  抓到的資料污染缺陷的修正，見 load_questions() docstring 記錄的兩個
  真實案例（veto誤判、評分依據污染）
- render_tag_table()：`--dry-run` 用的標記對照表，只斷言關鍵子字串存在
  （欄位寬度是排版細節，不斷言整行相等）
- compute_scoring_summary()／build_scoring_banner_line()：用真實的九格
  模板結構（讀 scripts/data/scoring_templates.json）驗證，不是虛構一套
  跟真實系統不相容的百分制模板——分數範圍、grade_bands 結構、
  max_score_per_dimension 都要跟真實系統一致，否則測不到真正的行為
- render_section() 新增的 dimension_name 參數（🎯 標記）：跟既有三種標記
  （🔗🚩🚫）同時出現時各自獨立一行，不互相覆蓋
- render_red_flags_section()：驗證紅旗清單格式、severity圖示對應（含未知
  severity的預設圖示）、headings去重排序、matched_questions格式
- render_consistency_section()：驗證數字一致性檢查三種狀態（conflict/
  consistent/其他）處理、僅顯示extraction.found=true的項目、source標註、
  needs_review複核警語、涵蓋文件清單去重排序
- render_scoring_section()：驗證評分表格生成、加權小計計算、子章節生成
  條件、PSF總評顯示條件（含缺分/評分失敗時不顯示）、警告訊息文字跟順序；
  template一律用真實九格schema（含grade_bands），不虛構不相容結構——這
  是第二次在測試草稿裡踩到「虛構跟真實系統不相容的資料結構」，詳見
  docs/rag-findings.md 的記錄
- render_veto_section()：驗證未觸發規則的一行式顯示、觸發規則的完整四行
  格式（現象/依據/建議/出現於）、同一規則多筆entries全部列出、heading
  缺失時的提示文字位置（放在「相關片段」括號裡，不取代question本身）
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from generate_report import (
    apply_full_dd_report_defaults,
    build_scoring_banner_line,
    check_deterministic,
    check_full_dd_report_tag_coverage,
    compute_scoring_summary,
    filter_citations,
    load_questions,
    load_scoring_template,
    render_consistency_section,
    render_red_flags_section,
    render_scoring_section,
    render_section,
    render_tag_table,
    render_veto_section,
    slugify_heading,
)

MIN_CITATION_SCORE = 0.6

# 真實的九格模板結構（跟 scripts/data/scoring_templates.json 的「九格」
# 逐字一致）——不能虛構一套百分制/不同 schema 的模板，compute_weighted_
# total()/lookup_grade() 是照這個 schema 寫的（weights全部=1、
# max_score_per_dimension=5、grade_bands 是由高到低排列的 min 門檻）。
NINE_GRID_TEMPLATE = {
    "dimensions": {
        "信用": 1, "資源": 1, "市場定位（權力）": 1, "動機": 1,
        "能力": 1, "風險": 1, "契合": 1, "貢獻": 1, "共業": 1,
    },
    "max_score_per_dimension": 5,
    "grade_bands": [
        {"min": 40, "grade": "A", "label": "核心夥伴", "action": "可進入核心合作"},
        {"min": 34, "grade": "B", "label": "策略夥伴", "action": "可專案合作"},
        {"min": 27, "grade": "C", "label": "資源夥伴", "action": "可單點合作"},
        {"min": 20, "grade": "D", "label": "觀察名單", "action": "低成本接觸"},
        {"min": 0, "grade": "E", "label": "風險名單", "action": "不宜合作"},
    ],
}


def _all_scored(score):
    return {dim: {"score": score, "rationale": "r", "evidence": ["e"]}
            for dim in NINE_GRID_TEMPLATE["dimensions"]}


# filter_citations() 測試案例
@pytest.mark.parametrize("hits,expected", [
    # 剛好等於門檻值的邊界情況
    ([{"score": 0.6}], [{"score": 0.6}]),
    # 過濾後清單為空的情境
    ([{"score": 0.5}, {"score": 0.59}], []),
    # 正常過濾情境
    ([{"score": 0.6}, {"score": 0.7}], [{"score": 0.6}, {"score": 0.7}]),
    # 零個項目的情境
    ([], []),
])
def test_filter_citations_boundary_and_empty(hits, expected):
    result = filter_citations(hits)
    assert result == expected, (
        f"輸入 {hits} 應返回 {expected}，實際得到 {result}"
    )


# slugify_heading() 測試案例
@pytest.mark.parametrize("text,expected_slug", [
    # 中文處理測試
    ("這是一個標題", "這是一個標題"),
    # 特殊符號處理
    ("Hello! World®?", "hello-world"),
    # 這是與 GitHub 渲染器行為一致的已知 upstream 限制（句點被直接刪除、
    # 非轉為連字符），非本函式缺陷——GitHub 自己的錨點演算法也是直接刪句
    # 點，所以「3.10」跟「3.11」在真實 GitHub 上一樣會撞出同一個錨點
    # 「python-310-vs-311」。曾評估改成把句點轉成連字符，但那樣會讓
    # slug 跟真實 GitHub/VS Code 產生的錨點不一致，違背這個函式想要
    # 「貼合真實渲染器行為」的設計目的，所以維持現況、不修這個技術債。
    ("Python 3.10 vs 3.11", "python-310-vs-311"),
    # 空白處理
    ("   多個   空格   ", "多個-空格"),
    # 混合情境
    ("Section #1: Introduction", "section-1-introduction"),
])
def test_slugify_heading_special_characters(text, expected_slug):
    result = slugify_heading(text)
    assert result == expected_slug, (
        f"「{text}」應轉換為 {expected_slug}，實際得到 {result}"
    )


def test_slugify_heading_collision():
    # 已知限制：不同標題可能撞出相同 slug，slugify_heading 目前沒有防碰撞
    # 機制。這裡用「空格」跟「已有的連字符」都會被轉成 "-" 的事實構造一組
    # 真的會碰撞的例子——不是假設,是這兩個輸入實際上就是會撞。
    slug1 = slugify_heading("Hello World")
    slug2 = slugify_heading("Hello-World")
    assert slug1 == slug2 == "hello-world", (
        f"「Hello World」與「Hello-World」預期撞出相同 slug "
        f"（已知限制,目前沒有防碰撞機制），實際得到 {slug1!r} 跟 {slug2!r}"
    )


# load_questions() 清除語法測試案例——`## group:` 一定要帶 `| metric:`
# 才能被 GROUP_HEADER_RE 配對到（跟 DIMENSION_HEADER_RE／VETO_HEADER_RE
# 不一樣，那兩個標記只要名稱就夠），這裡的測試資料要照這個既有語法寫，
# 不能只寫 `## group: A`（那樣配不到任何規則，會被當成普通註解跳過）。
@pytest.mark.parametrize("content,expected_groups", [
    (
        "## group: A | metric: 指標一\n"
        "問題1\n"
        "## group: none\n"
        "問題2\n"
        "## group: B | metric: 指標二\n"
        "問題3\n",
        ["A", None, "B"],
    ),
    (
        # 清除後又重新宣告同一個 id——清除不是永久關閉
        "## group: A | metric: 指標一\n"
        "問題1\n"
        "## group: none\n"
        "問題2\n"
        "## group: A | metric: 指標一\n"
        "問題3\n",
        ["A", None, "A"],
    ),
])
def test_load_questions_group_clear_syntax(tmp_path, content, expected_groups):
    path = tmp_path / "questions.txt"
    path.write_text(content, encoding="utf-8")
    questions = load_questions(str(path))
    assert [q["group"] for q in questions] == expected_groups


@pytest.mark.parametrize("content,expected_dims", [
    (
        "## dimension: 信用\n"
        "問題1\n"
        "## dimension: none\n"
        "問題2\n"
        "## dimension: 資源\n"
        "問題3\n",
        ["信用", None, "資源"],
    ),
])
def test_load_questions_dimension_clear_syntax(tmp_path, content, expected_dims):
    path = tmp_path / "questions.txt"
    path.write_text(content, encoding="utf-8")
    questions = load_questions(str(path))
    assert [q["dimension"] for q in questions] == expected_dims


@pytest.mark.parametrize("content,expected_vetoes", [
    (
        "## veto: 風險不揭露\n"
        "問題1\n"
        "## veto: none\n"
        "問題2\n"
        "## veto: 資源不實\n"
        "問題3\n",
        ["風險不揭露", None, "資源不實"],
    ),
])
def test_load_questions_veto_clear_syntax(tmp_path, content, expected_vetoes):
    path = tmp_path / "questions.txt"
    path.write_text(content, encoding="utf-8")
    questions = load_questions(str(path))
    assert [q["veto"] for q in questions] == expected_vetoes


def test_load_questions_clear_syntax_is_independent_per_marker(tmp_path):
    """三種標記各自獨立清除——清掉 dimension 不應該連帶清掉還在生效的
    group/veto（整合驗證抓到的污染案例正是三種標記互相干擾，這裡反過來
    驗證「清除語法本身」不會矯枉過正、變成清一個就把全部清空）。"""
    content = (
        "## group: G | metric: 指標\n"
        "## dimension: 信用\n"
        "## veto: 資源不實\n"
        "問題1\n"
        "## dimension: none\n"
        "問題2\n"
    )
    path = tmp_path / "questions.txt"
    path.write_text(content, encoding="utf-8")
    questions = load_questions(str(path))
    assert questions[0] == {
        "question": "問題1", "group": "G", "metric": "指標",
        "source": None, "dimension": "信用", "veto": "資源不實",
    }
    assert questions[1] == {
        "question": "問題2", "group": "G", "metric": "指標",
        "source": None, "dimension": None, "veto": "資源不實",
    }


def test_load_questions_topic_switch_regression(tmp_path):
    """直接對應整合驗證抓到的真實事故，不是清除語法的一般邊界案例：
    日羿智能那段用 `## dimension: 信用` + `## veto: 風險不揭露` 標記了
    股權題，換到 Branes.AI 的跨文件比對題時，兩個標記都要明確清除——
    這條測試釘的正是事故發生時那個轉折點的原始形狀（換 source、
    dimension+veto 同時要清除、換過去的題目完全沒有任何新標記）。

    這條測試故意不驗證「沒有標記的題目該不該繼承」——那是宣告後持續
    生效機制本身要做的事，`## group: ar | metric: 應收帳款` 底下兩題
    共用同一個 group 就是靠這個機制運作，不能因為要防範這次事故就把
    它斷言掉，那樣反而會鎖死一個已經上線、驗證過的正確行為。這條測試
    只釘住「有明確清除時，換主題的題目必須乾淨，不能繼承前段標記」，
    對應的正是事故裡「清除語法原本不存在」這個真正的缺陷，不是繼承
    機制本身。"""
    content = (
        "## dimension: 信用\n"
        "## veto: 風險不揭露\n"
        "[source: ruiyi_report.md] 公司合計揭露的股權比例是多少？\n"
        "## dimension: none\n"
        "## veto: none\n"
        "## group: crossdoc | metric: 2026年營收預測\n"
        "[source: branes_deck_excerpt.md] Pitch Deck 列出的營收預測金額是多少？\n"
    )
    path = tmp_path / "questions.txt"
    path.write_text(content, encoding="utf-8")
    questions = load_questions(str(path))

    ruiyi_q, branes_q = questions
    assert ruiyi_q["dimension"] == "信用"
    assert ruiyi_q["veto"] == "風險不揭露"

    # 這是事故的核心：換到 Branes 的題目，沒有寫任何新的 dimension/veto
    # 標記，必須是 None，不能繼承前一段日羿的標記——這正是整合驗證
    # 抓到「風險不揭露被誤判觸發」跟「信用維度評分依據被污染」兩個真實
    # 案例的成因。group 則正常設成新宣告的 crossdoc，不受影響（三種
    # 標記各自獨立，group 沒有被清除過，是正常換成新值）。
    assert branes_q["dimension"] is None
    assert branes_q["veto"] is None
    assert branes_q["group"] == "crossdoc"


# render_tag_table() 測試案例——只斷言關鍵子字串存在，不斷言整行/整表
# 字串相等，欄位寬度（padding）是排版細節不是行為保證。
def test_render_tag_table_shows_values_and_dashes():
    questions = [
        {"question": "測試1", "source": "file1.md", "group": "A",
         "metric": "指標一", "dimension": "信用", "veto": "資源不實"},
        {"question": "測試2", "source": None, "group": None,
         "metric": None, "dimension": None, "veto": None},
    ]
    table = render_tag_table(questions)
    assert "file1.md" in table
    assert "信用" in table
    assert "資源不實" in table
    assert "測試1" in table
    assert "測試2" in table
    # 第二題沒有任何標記，每個欄位都要顯示 "-" 而不是空白或 "None"
    lines = table.splitlines()
    second_row = next(l for l in lines if "測試2" in l)
    assert "None" not in second_row
    assert "-" in second_row


def test_render_tag_table_empty_list():
    # 零題時只印表頭跟分隔線，不噴例外
    table = render_tag_table([])
    assert "source" in table
    assert "問題" in table


# compute_scoring_summary() 測試案例——用真實九格模板 schema
def test_compute_scoring_summary_all_dimensions_scored():
    dimension_results = _all_scored(3)
    summary = compute_scoring_summary(dimension_results, NINE_GRID_TEMPLATE)
    assert summary == {
        "total": 27, "max_total": 45,
        "grade": {"grade": "C", "label": "資源夥伴", "action": "可單點合作"},
    }


def test_compute_scoring_summary_missing_dimension_returns_none():
    dimension_results = _all_scored(3)
    dimension_results.pop("信用")
    assert compute_scoring_summary(dimension_results, NINE_GRID_TEMPLATE) is None


def test_compute_scoring_summary_failed_dimension_returns_none():
    dimension_results = _all_scored(3)
    dimension_results["信用"] = {"score": None, "rationale": None, "evidence": None}
    assert compute_scoring_summary(dimension_results, NINE_GRID_TEMPLATE) is None


def test_compute_scoring_summary_empty_dict_returns_none():
    assert compute_scoring_summary({}, NINE_GRID_TEMPLATE) is None


# build_scoring_banner_line() 測試案例
def test_build_scoring_banner_line_matches_render_scoring_section_total():
    """跟 render_scoring_section() 算出的總評行必須是同一個數字——兩處
    共用 compute_scoring_summary()，這條測試釘住「不會各自維護一份重複
    邏輯、以後改了一邊忘記改另一邊」這件事本身。"""
    dimension_results = _all_scored(2)
    line = build_scoring_banner_line(dimension_results, NINE_GRID_TEMPLATE)
    assert line == "**PSF 總評**：18/45 — E（風險名單） → 不宜合作"


@pytest.mark.parametrize("dimension_results,template", [
    ({}, NINE_GRID_TEMPLATE),
    (None, NINE_GRID_TEMPLATE),
    (_all_scored(3), None),
])
def test_build_scoring_banner_line_returns_none_when_incomplete(dimension_results, template):
    assert build_scoring_banner_line(dimension_results, template) is None


def test_build_scoring_banner_line_returns_none_when_dimension_missing():
    incomplete = _all_scored(3)
    incomplete.pop("風險")
    assert build_scoring_banner_line(incomplete, NINE_GRID_TEMPLATE) is None


# render_section() 的 dimension_name 參數（🎯 標記）測試案例——result/qa
# 要給完整、真實形狀的 mock，不能只給 {"question": ...} 這種殘缺字典，
# 否則函式內部存取 result["visible"]/qa["needs_review"] 等既有欄位時會
# 直接 KeyError，測不到 dimension_name 本身的邏輯。
def _mock_result(question="測試問題"):
    return {
        "question": question, "visible": "這是答案內容",
        "hits": [], "think_used": False, "total": 1.0,
    }


def _mock_qa():
    return {"needs_review": False, "faithfulness": 9, "relevance": 9}


@pytest.mark.parametrize("dimension_name,should_appear", [
    ("信用", True),
    (None, False),
])
def test_render_section_dimension_marker(dimension_name, should_appear):
    result = render_section(1, _mock_result(), _mock_qa(), dimension_name=dimension_name)
    line = f"🎯 屬於評分維度：{dimension_name}"
    if should_appear:
        assert line in result
    else:
        assert "🎯" not in result


def test_render_section_all_four_markers_coexist_independently():
    """同一題同時帶四種標記（一致性檢查群組、評分維度、紅旗、veto）時，
    四行都要各自獨立出現，不能互相覆蓋或漏掉任何一行——這是整合驗證
    第一次同時開啟五種能力時要檢查的重點之一。"""
    result = render_section(
        1, _mock_result(), _mock_qa(),
        group_metric="應收帳款", red_flag_title="簡報財報落差",
        veto_title="資源不實", dimension_name="資源",
    )
    assert "🔗 屬於一致性檢查群組：應收帳款" in result
    assert "🎯 屬於評分維度：資源" in result
    assert "🚩 此題內容被判定為紅旗：簡報財報落差" in result
    assert "🚫 此題內容被判定觸發不合作紅線：資源不實" in result


def test_render_section_no_markers_when_all_none():
    result = render_section(1, _mock_result(), _mock_qa())
    assert "🔗" not in result
    assert "🎯" not in result
    assert "🚩" not in result
    assert "🚫" not in result


# render_red_flags_section()：red_flags為空清單時回傳空字串
def test_render_red_flags_section_empty():
    assert render_red_flags_section([]) == ""


# render_red_flags_section()：單筆紅旗且severity分別是high/medium/low時要出現對應圖示
@pytest.mark.parametrize("severity,expected_icon", [
    ("high", "🔴"),
    ("medium", "🟡"),
    ("low", "🟢"),
    ("unknown", "⚪"),
])
def test_render_red_flags_section_severity(severity, expected_icon):
    red_flags = [{
        "title": "測試紅旗",
        "severity": severity,
        "phenomenon": "現象",
        "why_it_matters": "原因",
        "required_action": "動作",
        "matched_questions": ["問題1"],
        "headings": []
    }]
    result = render_red_flags_section(red_flags)
    assert expected_icon in result, f"應顯示 {expected_icon} 圖示"


# render_red_flags_section()：headings為空清單時該筆要顯示「（無法定位到具體片段）」
def test_render_red_flags_section_headings_empty():
    red_flags = [{
        "title": "測試紅旗",
        "severity": "medium",
        "phenomenon": "現象",
        "why_it_matters": "原因",
        "required_action": "動作",
        "matched_questions": ["問題1"],
        "headings": []
    }]
    result = render_red_flags_section(red_flags)
    assert "（無法定位到具體片段）" in result, "headings 空時應顯示預期提示"


# render_red_flags_section()：headings有重複值時輸出要去重且排序後用、串接
def test_render_red_flags_section_headings_deduplicated():
    red_flags = [{
        "title": "測試紅旗",
        "severity": "medium",
        "phenomenon": "現象",
        "why_it_matters": "原因",
        "required_action": "動作",
        "matched_questions": ["問題1"],
        "headings": ["A", "B", "A"]
    }]
    result = render_red_flags_section([red_flags[0]])
    assert "A、B" in result, "headings 應去重排序後用「、」連接"


# render_red_flags_section()：matched_questions每個問題要用「」包住再用、串接
def test_render_red_flags_section_questions_quoted():
    red_flags = [{
        "title": "測試紅旗",
        "severity": "medium",
        "phenomenon": "現象",
        "why_it_matters": "原因",
        "required_action": "動作",
        "matched_questions": ["問題1", "問題2"],
        "headings": []
    }]
    result = render_red_flags_section([red_flags[0]])
    assert "「問題1」、「問題2」" in result, "matched_questions 應用「」包覆後用「、」連接"


# render_consistency_section()：consistency_results為空清單時回傳空字串
def test_render_consistency_section_empty():
    assert render_consistency_section([]) == ""


# render_consistency_section()：status="conflict"時只列出entries裡extraction.found=true的項目
# ——needs_review/heading/source/question 都是 entry 自己的欄位，不是巢狀在
# extraction 裡面（真正的函式讀 e["needs_review"]/e["heading"]/e["question"]，
# extraction 裡只放 found/raw_value）
def test_render_consistency_section_conflict_filtered():
    results = [{
        "metric": "測試指標",
        "status": "conflict",
        "entries": [
            {"extraction": {"found": False}, "question": "問題A",
             "needs_review": False, "heading": "章節A", "source": "文件A"},
            {"extraction": {"found": True, "raw_value": "100"}, "question": "問題B",
             "needs_review": False, "heading": "章節", "source": "文件1"},
        ]
    }]
    result = render_consistency_section(results)
    assert "測試指標 — 數字不一致" in result, "應顯示 conflict 指標"
    assert "**100**（來源：章節）" in result, "僅顯示 found=True 的項目"
    assert "問題A" not in result, "found=False 的項目不該出現在輸出裡"


# render_consistency_section()：needs_review=true的entry要出現複核警語
def test_render_consistency_section_needs_review():
    results = [{
        "metric": "測試指標",
        "status": "conflict",
        "entries": [{
            "extraction": {"found": True, "raw_value": "100"}, "question": "問題1",
            "needs_review": True, "heading": "章節", "source": "文件1",
        }]
    }]
    result = render_consistency_section(results)
    assert "（⚠️ 此題答案本身待人工複核，數字可信度打折扣）" in result, "needs_review 應顯示警語"


# render_consistency_section()：source有值時要出現［來源文件：X］標註
def test_render_consistency_section_source_annotation():
    results = [{
        "metric": "測試指標",
        "status": "conflict",
        "entries": [{
            "extraction": {"found": True, "raw_value": "100"}, "question": "問題1",
            "needs_review": False, "heading": "章節", "source": "文件1",
        }]
    }]
    result = render_consistency_section(results)
    assert "［來源文件：文件1］" in result, "source 應顯示標註"


# render_consistency_section()：status="consistent"時顯示找到的第一筆raw_value跟found筆數，
# 且涵蓋文件清單要去重排序——兩筆entry故意用重複的文件名測去重
def test_render_consistency_section_consistent():
    results = [{
        "metric": "測試指標",
        "status": "consistent",
        "entries": [
            {"extraction": {"found": True, "raw_value": "100"}, "question": "問題1",
             "needs_review": False, "heading": "章節", "source": "文件2"},
            {"extraction": {"found": True, "raw_value": "100"}, "question": "問題2",
             "needs_review": False, "heading": "章節", "source": "文件1"},
            {"extraction": {"found": True, "raw_value": "100"}, "question": "問題3",
             "needs_review": False, "heading": "章節", "source": "文件1"},
        ]
    }]
    result = render_consistency_section(results)
    assert "✅ **測試指標**：3 筆記錄皆得到一致數字（100）" in result, "應顯示一致結果"
    assert "涵蓋文件：文件1、文件2" in result, "涵蓋文件清單應去重（文件1只出現一次）且排序"


# render_consistency_section()：status是其他值時顯示ℹ️訊息
def test_render_consistency_section_other_status():
    results = [{
        "metric": "測試指標",
        "status": "unknown",
        "entries": []
    }]
    result = render_consistency_section(results)
    assert "ℹ️ **測試指標**：資料不足以比對" in result, "其他狀態應顯示提示訊息"


def test_render_scoring_section_empty_dimension_results():
    """dimension_results為空字典時回傳空字串"""
    result = render_scoring_section({}, {})
    assert result == "", "空輸入應返回空字串"


def test_render_scoring_section_all_success():
    """所有維度都評分成功時顯示PSF總評行"""
    dimension_results = {
        "dim1": {"score": 3, "rationale": "test1", "evidence": ["a"]},
        "dim2": {"score": 4, "rationale": "test2", "evidence": ["b"]}
    }
    template = {
        "dimensions": {"dim1": 2, "dim2": 3},
        "max_score_per_dimension": 5,
        "grade_bands": [
            {"min": 15, "grade": "A", "label": "優", "action": "可合作"},
            {"min": 0, "grade": "B", "label": "差", "action": None},
        ],
    }
    expected_summary = compute_scoring_summary(dimension_results, template)
    expected_grade = expected_summary["grade"]
    expected_line = f"**PSF 總評：{expected_summary['total']}/{expected_summary['max_total']} — "
    expected_line += f"{expected_grade['grade']}（{expected_grade['label']}）"
    if expected_grade.get("action"):
        expected_line += f" → {expected_grade['action']}"
    expected_line += "**"

    result = render_scoring_section(dimension_results, template)
    assert expected_line in result, "應顯示PSF總評行"


def test_render_scoring_section_failure_and_missing():
    """有評分失敗與未評分維度時顯示警告訊息"""
    dimension_results = {
        "dim1": {"score": None, "rationale": "", "evidence": []},
        "dim2": {"score": 4, "rationale": "test", "evidence": ["b"]}
    }
    template = {
        "dimensions": {"dim1": 1, "dim2": 2, "dim3": 3},
        "max_score_per_dimension": 5
    }

    result = render_scoring_section(dimension_results, template)
    assert "| dim1 | ⚠️ 評分失敗，需人工判斷 | ×1 | — |" in result, "評分失敗維度應顯示警告"
    assert "| dim3 | ⚠️ 未評分（問題清單未涵蓋此維度） | ×3 | — |" in result, "未評分維度應顯示提示"
    # 原斷言的文字順序跟措辭都跟真正的程式碼對不上：真正的程式碼先組
    # missing_dims 的理由再組 failed_dims 的理由（順序固定，不是依字母或
    # 出現順序），措辭是「問題清單未涵蓋：」不是「未評分（問題清單未
    # 涵蓋）：」——這兩處都是生成階段的既有錯誤，不是這次修正引入的。
    assert "> ⚠️ **無法計算總分**——問題清單未涵蓋：dim3；評分失敗：dim1。" in result, "應顯示無法計算總分警告"


def test_render_scoring_section_weighted_subtotal():
    """驗證加權小計是否正確計算"""
    dimension_results = {
        "dim1": {"score": 3, "rationale": "", "evidence": []},
        "dim2": {"score": 5, "rationale": "", "evidence": []}
    }
    template = {
        "dimensions": {"dim1": 2, "dim2": 3},
        "max_score_per_dimension": 5,
        "grade_bands": [{"min": 0, "grade": "A", "label": "測試等級", "action": None}],
    }

    result = render_scoring_section(dimension_results, template)
    assert "| dim1 | 3/5 | ×2 | 6 |" in result, "加權小計應為3×2=6"
    assert "| dim2 | 5/5 | ×3 | 15 |" in result, "加權小計應為5×3=15"


def test_render_scoring_section_success_subsections():
    """評分成功的維度生成子章節，失敗的不生成"""
    dimension_results = {
        "dim1": {"score": 3, "rationale": "reason", "evidence": ["item1", "item2"]},
        "dim2": {"score": None, "rationale": "", "evidence": []}
    }
    template = {"dimensions": {"dim1": 1, "dim2": 1}, "max_score_per_dimension": 5}

    result = render_scoring_section(dimension_results, template)
    assert "### dim1（3/5）" in result, "成功評分維度應生成子章節"
    assert "### dim2（" not in result, "評分失敗維度不生成子章節"
    assert "**評分依據**：reason" in result, "應顯示評分依據"
    assert "- item1" in result and "- item2" in result, "應顯示引用清單"


def test_render_scoring_section_missing_dims():
    """缺少template裡的維度時顯示未評分提示"""
    dimension_results = {"dim1": {"score": 3, "rationale": "", "evidence": []}}
    template = {"dimensions": {"dim1": 1, "dim2": 1}, "max_score_per_dimension": 5}

    result = render_scoring_section(dimension_results, template)
    assert "| dim2 | ⚠️ 未評分（問題清單未涵蓋此維度） | ×1 | — |" in result, "應顯示未評分維度提示"


def test_render_scoring_section_failure_only():
    """只有評分失敗維度時不顯示總評"""
    dimension_results = {"dim1": {"score": None, "rationale": "", "evidence": []}}
    template = {"dimensions": {"dim1": 1}, "max_score_per_dimension": 5}

    result = render_scoring_section(dimension_results, template)
    assert "> ⚠️ **無法計算總分**——評分失敗：dim1。" in result, "應顯示無法計算總分警告"
    # 原本斷言 "**PSF 總評**" not in result 是恒真的假陽性——真正的總評行
    # 格式是 "**PSF 總評：X/Y — ..."，"總評" 後面接的是全角冒號不是 "**"，
    # 所以 "**PSF 總評**" 這個字面組合永遠不會出現在真正的總評行裡，不管
    # 邏輯對不對，斷言都會通過，測不到「評分失敗時不該顯示總評」這件事。
    # 改成比對真正會出現在總評行開頭的字面文字。
    assert "PSF 總評：" not in result, "評分失敗時不顯示總評行"


# render_veto_section()：veto_results為空字典時回傳空字串
def test_render_veto_section_empty():
    result = render_veto_section({})
    assert result == "", "空字典輸入應返回空字串"


# render_veto_section()：triggered=false的規則顯示「- ✅ **規則名**：未觸發」
@pytest.mark.parametrize("veto_results,expected_line", [
    ({"rule1": {"triggered": False}}, "- ✅ **rule1**：未觸發"),
    ({"rule2": {"triggered": False, "entries": []}}, "- ✅ **rule2**：未觸發"),
])
def test_render_veto_section_not_triggered(veto_results, expected_line):
    result = render_veto_section(veto_results)
    assert expected_line in result, f"應包含未觸發規則顯示：{expected_line}"


# render_veto_section()：triggered=true的規則顯示「### 🚫 規則名」標題及所有entries
# ——entries 裡每一筆都要有 question 欄位（真正的函式讀 entry['question']，
# 沒有這個欄位會直接 KeyError，不是選填的）
@pytest.mark.parametrize("veto_results,expected_count", [
    (
        {
            "rule1": {
                "triggered": True,
                "entries": [
                    {"question": "q1", "phenomenon": "p1", "basis": "b1",
                     "suggested_action": "a1", "heading": "h1"},
                    {"question": "q2", "phenomenon": "p2", "basis": "b2",
                     "suggested_action": "a2", "heading": None},
                ]
            }
        },
        1  # 只有一條規則觸發，標題只會出現一次；entries 有兩筆是同一條規則底下的兩筆證據
    ),
    (
        {
            "rule2": {
                "triggered": True,
                "entries": [
                    {"question": "q3", "phenomenon": "p3", "basis": "b3",
                     "suggested_action": "a3", "heading": None}
                ]
            }
        },
        1
    ),
])
def test_render_veto_section_triggered(veto_results, expected_count):
    result = render_veto_section(veto_results)
    assert "### 🚫" in result, "應包含觸發規則標題"
    assert result.count("### 🚫") == expected_count, "應正確顯示觸發規則的標題數量"


def test_render_veto_section_multiple_entries_all_listed():
    """一條規則底下有多筆 entries 時，每一筆都要各自出現現象/依據/建議/
    出現於四行，不能只列第一筆——這是這條規則跟其他規則的關鍵差異，值得
    獨立一條測試單獨釘住，不要跟「標題數量」的測試混在一起。"""
    veto_results = {
        "rule1": {
            "triggered": True,
            "entries": [
                {"question": "q1", "phenomenon": "p1", "basis": "b1",
                 "suggested_action": "a1", "heading": "h1"},
                {"question": "q2", "phenomenon": "p2", "basis": "b2",
                 "suggested_action": "a2", "heading": None},
            ]
        }
    }
    result = render_veto_section(veto_results)
    assert "**現象**：p1" in result and "**現象**：p2" in result
    assert "**依據**：b1" in result and "**依據**：b2" in result
    assert "**建議**：a1" in result and "**建議**：a2" in result
    assert "「q1」" in result and "「q2」" in result


# render_veto_section()：entry的heading是None時「出現於」那行要顯示
# 「（相關片段：（無法定位到具體片段））」——無法定位的提示文字放在
# 「相關片段」那個括號裡面，不是取代掉question本身，question不管heading
# 是不是None都照樣顯示
@pytest.mark.parametrize("heading,expected_fragment_snippet", [
    (None, "（相關片段：（無法定位到具體片段））"),
    ("實際片段標題", "（相關片段：實際片段標題）"),
])
def test_render_veto_section_heading_handling(heading, expected_fragment_snippet):
    veto_results = {
        "rule1": {
            "triggered": True,
            "entries": [
                {"question": "測試問題", "phenomenon": "test",
                 "basis": "basis", "suggested_action": "action", "heading": heading}
            ]
        }
    }
    result = render_veto_section(veto_results)
    assert "「測試問題」" in result, "question 不管 heading 是不是 None 都要照樣顯示"
    assert expected_fragment_snippet in result, f"應顯示：{expected_fragment_snippet}"


# render_veto_section()：完整格式驗證（含邊界情況組合）
def test_render_veto_section_full_format():
    veto_results = {
        "矛盾型資源不實": {"triggered": False},
        "缺失型人不明": {
            "triggered": True,
            "entries": [
                {
                    "question": "合作方的主事者是誰？",
                    "phenomenon": "未明確說明合作方背景",
                    "basis": "文件未提及合作方資歷",
                    "suggested_action": "要求補充合作方資歷說明",
                    "heading": None,
                }
            ]
        },
        "缺失型權責不清": {
            "triggered": True,
            "entries": [
                {
                    "question": "誰負責什麼決策？",
                    "phenomenon": "未說明決策權限",
                    "basis": "協議未明確權責分工",
                    "suggested_action": "補充權責說明章節",
                    "heading": "權責分工",
                }
            ]
        }
    }

    result = render_veto_section(veto_results)

    # 驗證各部分存在性
    assert "## 不合作紅線檢查" in result
    assert "> ⚠️ **驗證邊界**：" in result
    assert "- ✅ **矛盾型資源不實**：未觸發" in result
    assert "### 🚫 缺失型人不明" in result
    assert "### 🚫 缺失型權責不清" in result
    assert "「合作方的主事者是誰？」（相關片段：（無法定位到具體片段））" in result
    assert "「誰負責什麼決策？」（相關片段：權責分工）" in result
    assert "---" in result

    # 驗證換行格式——splitlines() 不會為結尾的換行符另外產生一個空字串
    # 元素，所以分隔線是最後一個元素，不是倒數第二個（實測驗證過，不是
    # 憑印象推論字串 join/splitlines 的邊界行為）
    lines = result.splitlines()
    assert lines[-1] == "---", "結尾應為分隔線"


# ---------------------------------------------------------------------------
# check_deterministic()：批次F。截斷判斷用是否撞到 max_tokens 上限（門檻是
# max_tokens - 8，含），不是猜結尾標點；completion_tokens 不是 int 時永遠
# 不算撞上限；空白/空字串答案視為截斷；亂碼判斷：含「�」或連續同一字元
# 10 次以上（9 次不算）。
# ---------------------------------------------------------------------------

def test_check_deterministic_token_cap_boundary():
    # max_tokens=108 → 門檻是 100（含）
    assert check_deterministic("test", 99, 108)["not_truncated"] is True
    assert check_deterministic("test", 100, 108)["not_truncated"] is False
    assert check_deterministic("test", 108, 108)["not_truncated"] is False


@pytest.mark.parametrize("completion_tokens", ["200", None])
def test_check_deterministic_non_int_tokens_never_hit_cap(completion_tokens):
    assert check_deterministic("test", completion_tokens, 108)["not_truncated"] is True


@pytest.mark.parametrize("answer", ["", "   \n\t"])
def test_check_deterministic_empty_answer_is_truncated(answer):
    # completion_tokens 遠低於門檻，確保判定只來自空白答案本身
    assert check_deterministic("test", 10, 108)["not_truncated"] is True
    assert check_deterministic(answer, 10, 108)["not_truncated"] is False


def test_check_deterministic_replacement_char_is_garbled():
    assert check_deterministic("a\ufffdb", 10, 108)["no_garbled"] is False


def test_check_deterministic_repeat_run_boundary():
    assert check_deterministic("x" + "a" * 9 + "y", 10, 108)["no_garbled"] is True
    assert check_deterministic("x" + "a" * 10 + "y", 10, 108)["no_garbled"] is False


def test_check_deterministic_pass_requires_both():
    assert check_deterministic("正常回答", 10, 108)["deterministic_pass"] is True
    assert check_deterministic("正常回答", 100, 108)["deterministic_pass"] is False
    assert check_deterministic("正常\ufffd回答", 10, 108)["deterministic_pass"] is False


# ---------------------------------------------------------------------------
# load_scoring_template()：批次F。模板名稱打錯字要在讀模板階段直接死掉，不
# 是跑到評分階段才發現；跟 verify_sources_exist() 的 typo 前置檢查同一個
# 原則。用 tmp_path 建立真實檔案，不 mock 檔案系統；真實 scoring_templates.
# json 最上層直接就是模板名稱，沒有額外的 "templates" 包裝層。
# ---------------------------------------------------------------------------

def _write_templates(tmp_path):
    path = tmp_path / "scoring_templates.json"
    path.write_text(json.dumps({
        "九格": {"dimensions": [{"name": "A", "weight": 1.0}], "grade_bands": []},
        "深科技": {"dimensions": [{"name": "B", "weight": 1.0}], "grade_bands": []},
    }, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_scoring_template_returns_named_template(tmp_path):
    path = _write_templates(tmp_path)
    assert load_scoring_template(path, "深科技") == {
        "dimensions": [{"name": "B", "weight": 1.0}], "grade_bands": [],
    }


def test_load_scoring_template_missing_file_dies(tmp_path, capsys):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(SystemExit) as excinfo:
        load_scoring_template(missing, "九格")
    assert excinfo.value.code == 1
    assert f"找不到評分模板設定檔：{missing}" in capsys.readouterr().err


def test_load_scoring_template_unknown_name_dies_listing_available(tmp_path, capsys):
    path = _write_templates(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        load_scoring_template(path, "九宮格")
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "評分模板 '九宮格' 不存在" in err
    assert "['九格', '深科技']" in err


# ---------------------------------------------------------------------------
# apply_full_dd_report_defaults()：任務三。--full-dd-report 只補使用者
# 沒有主動選擇的部分——已經明確指定 --detect-red-flags 或 --scoring-template
# 時不覆蓋，不管使用者指定的是不是「九格」。用 SimpleNamespace 模擬
# argparse.Namespace，不需要真的跑 argparse。
# ---------------------------------------------------------------------------

def test_apply_full_dd_report_defaults_fills_both_when_unset():
    args = SimpleNamespace(full_dd_report=True, detect_red_flags=False, scoring_template=None)
    apply_full_dd_report_defaults(args)
    assert args.detect_red_flags is True
    assert args.scoring_template == "九格"


def test_apply_full_dd_report_defaults_does_not_override_explicit_scoring_template():
    args = SimpleNamespace(full_dd_report=True, detect_red_flags=False, scoring_template="深科技")
    apply_full_dd_report_defaults(args)
    assert args.detect_red_flags is True
    assert args.scoring_template == "深科技"


def test_apply_full_dd_report_defaults_does_not_override_explicit_detect_red_flags():
    # 已經是 True 時不該被重新賦值成別的東西（雖然結果一樣是 True，重點是
    # 不誤觸發沒必要的自動套用邏輯，跟上一條測試同一個「不覆蓋」原則對稱）
    args = SimpleNamespace(full_dd_report=True, detect_red_flags=True, scoring_template=None)
    apply_full_dd_report_defaults(args)
    assert args.detect_red_flags is True
    assert args.scoring_template == "九格"


def test_apply_full_dd_report_defaults_noop_when_flag_off():
    args = SimpleNamespace(full_dd_report=False, detect_red_flags=False, scoring_template=None)
    apply_full_dd_report_defaults(args)
    assert args.detect_red_flags is False
    assert args.scoring_template is None


# ---------------------------------------------------------------------------
# check_full_dd_report_tag_coverage()：任務三。純邏輯，檢查問題清單裡
# 三種tag的覆蓋率，缺什麼就回傳對應提示——不是規則檢查，只是「這份清單
# 可能沒有用滿DD報告四個章節」的非阻斷提醒。
# ---------------------------------------------------------------------------

def _q(dimension=None, veto=None, group=None):
    return {"question": "Q", "group": group, "metric": None, "source": None,
            "dimension": dimension, "veto": veto}


def test_check_tag_coverage_all_missing_when_scoring_template_set():
    args = SimpleNamespace(scoring_template="九格")
    hints = check_full_dd_report_tag_coverage([_q()], args)
    assert len(hints) == 3
    assert any("dimension" in h for h in hints)
    assert any("veto" in h for h in hints)
    assert any("group" in h for h in hints)


def test_check_tag_coverage_dimension_hint_absent_without_scoring_template():
    # 沒開評分模板時，即使沒有 dimension 標記也不用提示——評分本來就沒開
    args = SimpleNamespace(scoring_template=None)
    hints = check_full_dd_report_tag_coverage([_q()], args)
    assert not any("dimension" in h for h in hints)
    assert any("veto" in h for h in hints)
    assert any("group" in h for h in hints)


def test_check_tag_coverage_all_present_gives_no_hints():
    args = SimpleNamespace(scoring_template="九格")
    questions = [_q(dimension="動機", veto="人不明", group="定價")]
    assert check_full_dd_report_tag_coverage(questions, args) == []


def test_check_tag_coverage_only_one_question_needs_the_tag():
    # any()：清單裡只要有一題帶了某種標記就不提示，不用每題都標
    args = SimpleNamespace(scoring_template="九格")
    questions = [_q(), _q(dimension="動機", veto="人不明", group="定價")]
    assert check_full_dd_report_tag_coverage(questions, args) == []
