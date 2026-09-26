"""
測試重點（generate_report.py 的字母等級報告部分，深科技模板）：
- build_letter_scoring_banner_line()：報頭只寫綜合等級由人工判斷，不出現
  任何總分
- render_letter_scoring_section()：七個維度依模板順序；未評分／評分失敗
  照實標示；Gate 列接在對應維度後面；綜合列固定「由人工判斷」；表格儲存格
  的直線與換行不能切壞表格；不出現加權或分數
- load_questions()：`## gate:` 標記的延續與 `## gate: none` 清除，跟
  dimension／veto 標記互相獨立
- load_scoring_template()：設定有誤的模板在讀取階段就死掉並列出錯誤

模板一律讀 scripts/data/scoring_templates.json 的深科技真實模板。
"""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from generate_report import (
    build_letter_scoring_banner_line,
    load_questions,
    load_scoring_template,
    render_letter_scoring_section,
)

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    DEEP = json.load(_f)["深科技"]
DIMS = list(DEEP["dimensions"])


def _ok(grade, rationale="r"):
    return {"grade": grade, "rationale": rationale, "evidence": ["e1", "e2"], "parse_error": None}


def _failed():
    return {"grade": None, "rationale": None, "evidence": None, "parse_error": "bad json"}


# build_letter_scoring_banner_line()：有結果時只寫由人工判斷
def test_banner_with_results():
    assert (build_letter_scoring_banner_line({"團隊": _ok("A")}, DEEP)
            == "**綜合評等**：由人工判斷（深科技模板不訂權重、不計算綜合等級）")


# build_letter_scoring_banner_line()：沒有結果或沒有模板時不顯示
@pytest.mark.parametrize("results,template", [({}, DEEP), ({"團隊": _ok("A")}, None)])
def test_banner_absent(results, template):
    assert build_letter_scoring_banner_line(results, template) is None


# render_letter_scoring_section()：空結果回傳空字串
def test_section_empty():
    assert render_letter_scoring_section({}, DEEP) == ""


def _table_rows(content):
    return [l for l in content.splitlines() if l.startswith("| ") and not l.startswith("| 維度")]


# render_letter_scoring_section()：七個維度依模板順序，未評分與評分失敗照實標示
def test_section_rows_follow_template_order():
    results = {"市場驗證": _ok("C"), "團隊": _failed()}
    rows = _table_rows(render_letter_scoring_section(results, DEEP))
    dim_rows = [r for r in rows if not r.startswith("| ↳") and not r.startswith("| 綜合")]
    assert [r.split(" | ")[0][2:] for r in dim_rows] == DIMS
    row = {r.split(" | ")[0][2:]: r for r in dim_rows}
    assert "| C |" in row["市場驗證"]
    assert "評分失敗，需人工判斷" in row["團隊"]
    assert "未評分" in row["問題真實性"]
    assert rows[-1].startswith("| 綜合 | 由人工判斷 |")


# render_letter_scoring_section()：rationale 的直線與換行不能切壞表格
def test_section_escapes_table_cell():
    content = render_letter_scoring_section({"團隊": _ok("A", "甲|乙\n丙")}, DEEP)
    row = next(r for r in _table_rows(content) if r.startswith("| 團隊 |"))
    assert "甲｜乙 丙" in row
    assert row.count("|") == 4


# render_letter_scoring_section()：Gate 1 接在市場驗證後、財務紀律前；Gate 2 接在技術差異化後
def test_section_gate_rows_positioned_after_their_dimension():
    gate_results = {"Gate 1": {"status": "not_met", "entries": [
        {"question": "q", "heading": "玖", "status": "not_met", "basis": "b", "quote": "Customer 1", "parse_error": None}]}}
    content = render_letter_scoring_section({"市場驗證": _ok("C"), "技術差異化": _ok("B+")}, DEEP, gate_results)
    rows = _table_rows(content)
    names = [r.split(" | ")[0][2:] for r in rows]
    assert names.index("↳ Gate 1") == names.index("市場驗證") + 1
    assert names.index("↳ Gate 1") < names.index("財務紀律")
    assert names.index("↳ Gate 2") == names.index("技術差異化") + 1
    assert "問題清單未標記 Gate 2，未檢查" in content
    assert "### Gate 查核明細" in content
    assert "「Customer 1」" in content


# render_letter_scoring_section()：沒有 gate_results 時不出現查核明細
def test_section_no_gate_details_without_results():
    content = render_letter_scoring_section({"團隊": _ok("A")}, DEEP)
    assert "### Gate 查核明細" not in content


# render_letter_scoring_section()：只列字母，不出現加權、總分或分數格式
def test_section_has_no_numeric_scoring():
    results = {dim: _ok("B") for dim in DIMS}
    content = render_letter_scoring_section(results, DEEP)
    assert "加權" not in content
    assert "總分" not in content
    assert "/5" not in content
    assert "### 團隊（B）" in content


def _write(tmp_path, text):
    path = tmp_path / "q.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# load_questions()：gate 標記延續、清除，且跟 dimension／veto 互相獨立
def test_load_questions_gate_marker(tmp_path):
    path = _write(tmp_path, (
        "## dimension: 市場驗證\n"
        "## veto: 智財歸屬不完整\n"
        "## gate: Gate 1\n"
        "問題1\n"
        "問題2\n"
        "## gate: none\n"
        "問題3\n"
        "## gate: Gate 2\n"
        "## dimension: none\n"
        "問題4\n"
    ))
    qs = load_questions(path)
    assert [q["gate"] for q in qs] == ["Gate 1", "Gate 1", None, "Gate 2"]
    assert [q["dimension"] for q in qs] == ["市場驗證", "市場驗證", "市場驗證", None]
    assert [q["veto"] for q in qs] == ["智財歸屬不完整"] * 4


# load_questions()：沒有 gate 標記的舊檔案，每題 gate 都是 None
def test_load_questions_without_gate_marker(tmp_path):
    qs = load_questions(_write(tmp_path, "問題1\n## dimension: 信用\n問題2\n"))
    assert [q["gate"] for q in qs] == [None, None]


# load_scoring_template()：真實深科技模板正常讀取
def test_load_scoring_template_real_deep_tech():
    assert load_scoring_template(_TEMPLATES_PATH, "深科技") == DEEP


# load_scoring_template()：設定有誤的模板在讀取階段就死掉，列出具體錯誤
def test_load_scoring_template_invalid_dies(tmp_path, capsys):
    broken = copy.deepcopy(DEEP)
    broken["grade_bands"] = [{"min": 0, "grade": "E", "label": "L"}]
    path = tmp_path / "templates.json"
    path.write_text(json.dumps({"深科技": broken}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        load_scoring_template(str(path), "深科技")
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "設定有誤" in err
    assert "grade_bands" in err


# render_letter_scoring_section()：必須揭露等級會浮動、字母等級僅供參考、最終由人工判斷
# （驗證時同一份輸入、溫度 0 重跑，同一維度出現過 A 與 A−、C+ 與 B− 的差異）
def test_section_discloses_grade_variability():
    content = render_letter_scoring_section({"團隊": _ok("A")}, DEEP)
    assert "**等級會浮動**" in content
    assert "僅供參考" in content
    assert "最終由人工判斷" in content
