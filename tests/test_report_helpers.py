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
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from generate_report import (
    build_scoring_banner_line,
    compute_scoring_summary,
    filter_citations,
    load_questions,
    render_section,
    render_tag_table,
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
