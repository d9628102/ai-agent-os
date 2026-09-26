"""
測試重點（Gate 3 沿用 veto 機制：模板自帶的否決規則）：
- _mode_block_with_consequence()：只替換「建議」那一句的前提，模式區塊
  其餘判斷規則原封不動；替換後不能殘留「建議不合作」
- _build_veto_system_prompt()：rules=None 時五不合作的 prompt 逐字不變；
  傳入模板規則時改用規則自己的出處、原文判準與後果
- summarize_veto_results()：傳入模板規則時依模板規則順序
- resolve_veto_config()：沒有模板或模板沒有 veto_rules 時回到五不合作；
  深科技模板改用 Gate 3 的標籤與後果
- build_veto_banner_line()／render_veto_section()：不傳 config 時輸出
  跟加入模板規則之前逐字相同；深科技 config 顯示「完成智財歸屬確認前
  不可投資」，不顯示「建議不合作」，也不宣稱判斷依據是五不合作條款
  （說明裡可以提「沿用五不合作的 veto 判斷機制」，那是機制來源）

模板規則一律讀 scripts/data/scoring_templates.json 的深科技真實模板。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from veto_check import (
    VETO_RULES,
    _ABSENCE_MODE_BLOCK,
    _CONTRADICTION_MODE_BLOCK,
    _build_veto_system_prompt,
    _mode_block_with_consequence,
    summarize_veto_results,
)
from generate_report import build_veto_banner_line, render_veto_section, resolve_veto_config

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "data", "scoring_templates.json"
)
with open(_TEMPLATES_PATH, encoding="utf-8") as _f:
    _TEMPLATES = json.load(_f)
DEEP = _TEMPLATES["深科技"]
NINE = _TEMPLATES["九格"]
RULES = DEEP["veto_rules"]
RULE = "智財歸屬不完整"
CONSEQUENCE = RULES[RULE]["consequence"]


# _mode_block_with_consequence()：兩種模式都只換掉「建議」前提句
@pytest.mark.parametrize("mode,original,verb", [
    ("矛盾", _CONTRADICTION_MODE_BLOCK, "提供/證明"),
    ("缺失", _ABSENCE_MODE_BLOCK, "補充/揭露"),
])
def test_mode_block_with_consequence(mode, original, verb):
    block = _mode_block_with_consequence(mode, CONSEQUENCE)
    assert "建議不合作" not in block
    assert "不合作的建議" not in block
    assert (f"以「{CONSEQUENCE}」為前提，具體寫出除非對方能{verb}什麼，"
            f"否則維持「{CONSEQUENCE}」的建議") in block
    # 除了那一句，其他內容原封不動：把新句子換回原句要得到原區塊
    restored = block.replace(
        f"以「{CONSEQUENCE}」為前提，具體寫出除非對方能{verb}什麼，否則維持「{CONSEQUENCE}」的建議",
        f"以「建議不合作」為前提，具體寫出除非對方能{verb}什麼，否則維持不合作的建議",
    )
    assert restored == original


# _build_veto_system_prompt()：rules=None 時五條規則都跟原本一樣
@pytest.mark.parametrize("rule_name", list(VETO_RULES))
def test_default_prompt_is_five_rules(rule_name):
    prompt = _build_veto_system_prompt(rule_name)
    expected_block = _CONTRADICTION_MODE_BLOCK if VETO_RULES[rule_name]["mode"] == "矛盾" else _ABSENCE_MODE_BLOCK
    assert "五不合作" in prompt
    assert expected_block in prompt
    assert VETO_RULES[rule_name]["description"] in prompt


# _build_veto_system_prompt()：模板規則改用自己的出處、定義、原文判準與後果
def test_template_rule_prompt():
    prompt = _build_veto_system_prompt(RULE, RULES)
    rule = RULES[RULE]
    assert rule["source_label"] in prompt
    assert rule["description"] in prompt
    assert rule["criterion_verbatim"] in prompt
    assert CONSEQUENCE in prompt
    assert "五不合作" not in prompt
    assert "建議不合作" not in prompt


# _build_veto_system_prompt()：規則名稱不在傳入的規則裡要丟 KeyError（名稱在跑報告前就該被擋下）
def test_template_rule_prompt_unknown_rule():
    with pytest.raises(KeyError):
        _build_veto_system_prompt("人不明", RULES)


def _veto_entry(rule, triggered):
    detection = {"is_veto_triggered": triggered, "phenomenon": "p" if triggered else None,
                 "basis": "b" if triggered else None, "suggested_action": "s" if triggered else None,
                 "parse_error": None}
    return {"question": "q", "rule": rule, "detection": detection, "heading": "玖"}


# summarize_veto_results()：傳入模板規則時依模板規則彙總
def test_summarize_with_template_rules():
    results = summarize_veto_results([_veto_entry(RULE, False), _veto_entry(RULE, True)], RULES)
    assert list(results) == [RULE]
    assert results[RULE]["triggered"] is True
    assert len(results[RULE]["entries"]) == 1


# summarize_veto_results()：傳入模板規則時，五不合作的規則名稱不會被收進結果
def test_summarize_with_template_rules_ignores_other_rules():
    results = summarize_veto_results([_veto_entry("人不明", True)], RULES)
    assert results == {}


# resolve_veto_config()：沒有模板、或九格（沒有 veto_rules）都回到五不合作
@pytest.mark.parametrize("template", [None, NINE])
def test_resolve_veto_config_default(template):
    config = resolve_veto_config(template)
    assert config["rules"] is None
    assert config["label"] == "不合作紅線"
    assert config["section_title"] == "不合作紅線檢查"
    assert config["consequences"] == {name: "建議不合作" for name in VETO_RULES}


# resolve_veto_config()：深科技模板改用 Gate 3 的標籤與後果
def test_resolve_veto_config_deep_tech():
    config = resolve_veto_config(DEEP)
    assert config["rules"] is RULES
    assert config["label"] == DEEP["veto_label"]
    assert config["section_title"] == f"{DEEP['veto_label']}檢查"
    assert config["consequences"] == {RULE: "完成智財歸屬確認前不可投資"}
    # 說明可以提到「沿用五不合作的 veto 判斷機制」，但不能宣稱判斷依據是五不合作條款
    assert "依《PSF六壬合夥生態系統》「五不合作」條款" not in config["section_note"]
    assert "沿用五不合作的 veto 判斷機制" in config["section_note"]


# build_veto_banner_line()：不傳 config 時跟原本逐字相同
def test_banner_default_unchanged():
    assert build_veto_banner_line({"人不明": {"triggered": True}}) == "**不合作紅線**：🚫 觸發「人不明」——建議不合作"
    assert build_veto_banner_line({"人不明": {"triggered": False}}) == "**不合作紅線**：✅ 未觸發"
    assert build_veto_banner_line({}) is None


# build_veto_banner_line()：深科技 config
def test_banner_deep_tech():
    config = resolve_veto_config(DEEP)
    assert (build_veto_banner_line({RULE: {"triggered": True}}, config)
            == "**投資前必要條件（Gate 3）**：🚫 觸發「智財歸屬不完整」——完成智財歸屬確認前不可投資")
    assert build_veto_banner_line({RULE: {"triggered": False}}, config) == "**投資前必要條件（Gate 3）**：✅ 未觸發"
    assert build_veto_banner_line({}, config) is None


def _section_results(triggered):
    entries = [{"question": "q", "heading": "玖", "phenomenon": "p", "basis": "b", "suggested_action": "s"}]
    return {RULE: {"triggered": triggered, "entries": entries if triggered else []}}


# render_veto_section()：深科技 config 的章節標題與說明不提五不合作
def test_section_deep_tech():
    content = render_veto_section(_section_results(True), resolve_veto_config(DEEP))
    assert content.startswith("## 投資前必要條件（Gate 3）檢查")
    assert "依《PSF六壬合夥生態系統》「五不合作」條款" not in content
    assert "### 🚫 智財歸屬不完整" in content


# render_veto_section()：不傳 config 時標題與說明照舊
def test_section_default_unchanged():
    results = {"人不明": {"triggered": False, "entries": []}}
    content = render_veto_section(results)
    assert content.startswith("## 不合作紅線檢查")
    assert "依《PSF六壬合夥生態系統》「五不合作」條款" in content
    assert "- ✅ **人不明**：未觸發" in content
