"""
測試重點：
- parse_scoring_response()：驗證 JSON 解析、欄位完整性檢查、code fence/配對
  think 標籤剝殼、錯誤處理
- compute_weighted_total()：驗證加權總分計算邏輯、最大可能分數計算方式
  （維度總數 × 上限，不是權重總和 × 上限）、真實案例驗證
- lookup_grade()：驗證等級判斷邏輯、邊界值處理（含新增的 action 欄位）；
  真實案例驗證直接讀取 scripts/data/scoring_templates.json，不硬編一份
  複本——設定檔的等級邊界之後如果調整，這條測試會直接反映真實內容，不會
  安靜地跟設定檔脫鉤
- JSON_OUTPUT_REMINDER 是否真的接進三個 system prompt（numeric_consistency/
  red_flag_detection/scoring）——共用修正是否三處都生效的驗證
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from scoring import (
    compute_weighted_total,
    lookup_grade,
    parse_scoring_response,
)


# parse_scoring_response()：合法JSON(score是1-5的整數、rationale有值、evidence是非空list)時,回傳score/rationale/evidence都保留,parse_error=None
@pytest.mark.parametrize("raw,description", [
    ('{"score": "3", "rationale": "test", "evidence": ["a"]}', "合法輸入"),
    ('{"score": 5, "rationale": "test", "evidence": ["a", "b"]}', "數字格式score"),
])
def test_parse_scoring_response_valid(raw, description):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is None, f"{description}：parse_error 應為 None"
    assert "score" in result and result["score"] is not None, f"{description}：score 應存在且非空"
    assert "rationale" in result and result["rationale"], f"{description}：rationale 應存在且非空"
    assert "evidence" in result and result["evidence"], f"{description}：evidence 應存在且非空"


# parse_scoring_response()：score不是1-5範圍內的整數(例如0、6、或無法轉成int的字串)時,視為抽取失敗,回傳score=None且parse_error要有值
@pytest.mark.parametrize("raw,description", [
    ('{"score": 0}', "score=0"),
    ('{"score": "6"}', "score=6"),
    ('{"score": "invalid"}', "非數字字串"),
    ('{"score": 3.5}', "浮點數"),
])
def test_parse_scoring_response_invalid_score(raw, description):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"
    assert result["score"] is None, f"{description}：score 應為 None"


# parse_scoring_response()：rationale缺漏或是空字串時,視為抽取失敗,parse_error要有值
@pytest.mark.parametrize("raw,description", [
    ('{"score": 3}', "rationale 缺漏"),
    ('{"score": 3, "rationale": ""}', "rationale 空字串"),
])
def test_parse_scoring_response_missing_rationale(raw, description):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"
    assert result["score"] is None, f"{description}：score 應為 None"


# parse_scoring_response()：evidence缺漏、不是list、或是空list時,視為抽取失敗,parse_error要有值
@pytest.mark.parametrize("raw,description", [
    ('{"score": 3, "rationale": "test"}', "evidence 缺漏"),
    ('{"score": 3, "rationale": "test", "evidence": "not a list"}', "evidence 非list"),
    ('{"score": 3, "rationale": "test", "evidence": []}', "evidence 空list"),
])
def test_parse_scoring_response_invalid_evidence(raw, description):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"
    assert result["score"] is None, f"{description}：score 應為 None"


# parse_scoring_response()：模型輸出被配對的<think>...</think>標籤或三個反引號加json包住時,要能正確剝殼後解析出合法JSON
@pytest.mark.parametrize("raw,expected_score", [
    ("```json\n{ \"score\": \"3\", \"rationale\": \"test\", \"evidence\": [\"a\"] }\n```", 3),
    ("<think>先確認負面事實再打分</think>\n{ \"score\": \"5\", \"rationale\": \"test\", \"evidence\": [\"x\"] }\n", 5),
])
def test_parse_scoring_response_fences(raw, expected_score):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is None, "應成功解析"
    assert result["score"] == expected_score, "score 值應正確"


# parse_scoring_response()：模型輸出根本不是合法JSON時,parse_error要有值,不能讓例外往外拋
@pytest.mark.parametrize("raw,description", [
    ("invalid json", "純文字"),
    ('"score": 1, "rationale": "test", "evidence": ["x"]}', "缺開頭大括號"),
    ("{ \"score\": \"3\", \"rationale\": \"test\" }", "缺 evidence"),
    ("{ \"score\": \"invalid\" }", "無法轉成整數"),
])
def test_parse_scoring_response_json_error(raw, description):
    result = parse_scoring_response(raw)
    assert result["parse_error"] is not None, f"{description}：parse_error 應有值"


# compute_weighted_total()：純邏輯,回傳(加權總分,最大可能分數)。最大可能分數是「維度總數 × 5」
@pytest.mark.parametrize("dimension_scores,template,expected", [
    (
        {"dim1": 3, "dim2": 4},
        {"dimensions": {"dim1": 2, "dim2": 3}, "max_score_per_dimension": 5},
        (3 * 2 + 4 * 3, 2 * 5)
    ),
    (
        {"dim1": 5},
        {"dimensions": {"dim1": 1, "dim2": 1}, "max_score_per_dimension": 5},
        (5 * 1, 2 * 5)
    ),
])
def test_compute_weighted_total_basic(dimension_scores, template, expected):
    result = compute_weighted_total(dimension_scores, template)
    assert result == expected, "加權總分與最大可能分數計算錯誤"


# compute_weighted_total()：真實案例驗證——日羿智能九個維度分數搭配九格模板權重。
#
# 這裡的權重（全部是1）跟預期值（16, 45）是訂正過的答案，不是原本測試沒過
# 才改成這樣規避測試：《PSF六壬合夥生態系統》原始文件「七、九格評分制度」
# 明文「每格1～5分」「總分45分」，完全沒有權重倍數的描述。這條測試原本用
# 的權重（市場定位×2/能力×2/風險×2/貢獻×2/共業×3，算出26）是在沒看過原始
# 文件時反推估計出來的，已確認無依據。訂正後的16分，用日羿智能報告柒章的
# 九個原始分數直接加總驗證：2+1+1+2+1+1+3+3+2=16。
@pytest.mark.parametrize("dimension_scores,template,expected", [
    (
        {
            "信用": 2, "資源": 1, "市場定位（權力）": 1, "動機": 2, "能力": 1,
            "風險": 1, "契合": 3, "貢獻": 3, "共業": 2
        },
        {
            "dimensions": {
                "信用": 1, "資源": 1, "市場定位（權力）": 1, "動機": 1, "能力": 1,
                "風險": 1, "契合": 1, "貢獻": 1, "共業": 1
            },
            "max_score_per_dimension": 5
        },
        (16, 45)
    ),
])
def test_compute_weighted_total_real_case(dimension_scores, template, expected):
    result = compute_weighted_total(dimension_scores, template)
    assert result == expected, "真實案例計算結果錯誤"


# lookup_grade()：等級判斷邏輯驗證。這幾條用合成模板，不帶 action 欄位，
# 驗證 lookup_grade() 用 .get("action") 讀取時，沒有這個欄位就回傳 None，
# 不會因為舊模板沒有 action 而報錯。
@pytest.mark.parametrize("total,template,expected", [
    (26, {"grade_bands": [{"min": 25, "grade": "P2", "label": "良好"}, {"min": 20, "grade": "P3", "label": "普通"}]}, {"grade": "P2", "label": "良好", "action": None}),
    (21, {"grade_bands": [{"min": 25, "grade": "P2", "label": "良好"}, {"min": 20, "grade": "P3", "label": "普通"}]}, {"grade": "P3", "label": "普通", "action": None}),
    (19, {"grade_bands": [{"min": 25, "grade": "P2", "label": "良好"}, {"min": 20, "grade": "P3", "label": "普通"}]}, {"grade": None, "label": "無法判定等級", "action": None}),
])
def test_lookup_grade_basic(total, template, expected):
    result = lookup_grade(total, template)
    assert result == expected, "等級判斷邏輯錯誤"


# lookup_grade()：邊界值處理驗證
@pytest.mark.parametrize("total,template,expected", [
    (25, {"grade_bands": [{"min": 25, "grade": "P2", "label": "良好"}]}, {"grade": "P2", "label": "良好", "action": None}),
    (20, {"grade_bands": [{"min": 25, "grade": "P2", "label": "良好"}, {"min": 20, "grade": "P3", "label": "普通"}]}, {"grade": "P3", "label": "普通", "action": None}),
])
def test_lookup_grade_boundary(total, template, expected):
    result = lookup_grade(total, template)
    assert result == expected, "邊界值處理錯誤"


# lookup_grade()：真實案例驗證——直接讀取 scripts/data/scoring_templates.json
# 的九格模板，不硬編複本。日羿智能 16/45→E、宣捷幹細胞 21/45→D 是這個
# 模板等級表的兩個真實資料點來源，此處對照的就是真實設定檔內容。
#
# 這裡的預期值（16→E、21→D）是訂正過的答案，不是規避測試：
# - 日羿智能：九個原始分數（2+1+1+2+1+1+3+3+2）直接加總是16，不是套權重
#   算出來的26。對照《PSF六壬合夥生態系統》的A-E表，16落在「19以下→E級
#   風險名單→不宜合作」。這個訂正不只是數字校準——日羿智能報告自己的文字
#   結論是「不建議純財務投資/不建議股權投資/不建議技術合作/不建議合資」，
#   四個維度全部「不建議」，跟E級「不宜合作」的方向一致；原本套權重算出
#   的26分對應舊版P2「高風險，附條件合作」，反而讓判斷顯得比報告本身的
#   立場更溫和——訂正後的方向更準確，不是單純改數字過測試。
# - 宣捷幹細胞：九個原始分數（2+3+3+2+2+1+3+2+3）直接加總是21，跟報告
#   本身顯示的「21/45」完全吻合——這是文件之外，第二個獨立支持「無權重」
#   結論的真實資料點，不是單一案例的巧合。21落在「20–26→D級觀察名單→
#   低成本接觸」。
def _load_jiuge_template():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "data", "scoring_templates.json",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["九格"]


@pytest.mark.parametrize("total,expected_grade,expected_label,expected_action", [
    (16, "E", "風險名單", "不宜合作"),
    (21, "D", "觀察名單", "低成本接觸"),
])
def test_lookup_grade_real_case(total, expected_grade, expected_label, expected_action):
    template = _load_jiuge_template()
    result = lookup_grade(total, template)
    assert result["grade"] == expected_grade, (
        f"total={total} 應對應 {expected_grade}，實際 {result['grade']}"
    )
    assert result["label"] == expected_label, (
        f"total={total} 的 label 應為「{expected_label}」，實際「{result['label']}」"
    )
    assert result["action"] == expected_action, (
        f"total={total} 的 action 應為「{expected_action}」，實際「{result['action']}」"
    )


# JSON_OUTPUT_REMINDER 共用修正：驗證真的接進三個 system prompt，不是只
# 修了其中一個或兩個——避免同一類 bug 以後在別的地方用不同方式重現。
def test_json_output_reminder_wired_into_all_three_prompts():
    from rag_common import JSON_OUTPUT_REMINDER
    from numeric_consistency import EXTRACT_SYSTEM_PROMPT
    from red_flag_detection import RED_FLAG_SYSTEM_PROMPT
    from scoring import SCORING_SYSTEM_PROMPT

    assert JSON_OUTPUT_REMINDER in EXTRACT_SYSTEM_PROMPT, (
        "numeric_consistency.EXTRACT_SYSTEM_PROMPT 沒有接上 JSON_OUTPUT_REMINDER"
    )
    assert JSON_OUTPUT_REMINDER in RED_FLAG_SYSTEM_PROMPT, (
        "red_flag_detection.RED_FLAG_SYSTEM_PROMPT 沒有接上 JSON_OUTPUT_REMINDER"
    )
    assert JSON_OUTPUT_REMINDER in SCORING_SYSTEM_PROMPT, (
        "scoring.SCORING_SYSTEM_PROMPT 沒有接上 JSON_OUTPUT_REMINDER"
    )
