"""
測試重點：
- detect_think_reason() 的 8 個案例全部來自 merge 前(DEFAULT_THINKING 還不存在,
  detect_think_reason() 單獨決定思考模式)那次 8 題問答的真實結果 —— 不是編出來的
  假設情境,是這幾輪已經在 GX10 上跑過、有明確已知答案的真實問題。
  合併之後跑的測試(兩輪8題穩定性測試、10題 race condition 自我測試)因為
  DEFAULT_THINKING=True 已經讓所有題目預設走思考模式,無法拿來當這支函式的
  ground truth,所以不用。
- split_think() 的 4 個案例照它自己 docstring 列的 4 種已知情境寫:完整
  <think>...</think>、只有結尾 </think>、只有開頭沒結尾(截斷)、完全沒有標籤。
- build_context()：驗證多筆hits的編號與換行連接、heading_path/body/text
  欄位的預設值處理、hit完全缺payload時的容錯。body為空的案例直接比對
  完整block內容，不是只檢查空字串是否為子字串（那樣不管body有沒有洩漏
  多餘內容都會通過，等於沒測到）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from rag_answer import build_context, detect_think_reason, split_think


# 這 8 筆是 merge 前那次「8 題問答」的真實結果，見
# docs/rag-findings.md Known Issue #7。
KNOWN_THINK_CASES = [
    ("PSF EIM 跟一般 ERP/CRM 有什麼不同？", True, "比較性詞彙「不同」"),
    ("PSF EIM 裡負責客戶服務的 Agent 是誰？", True, "2個專有名詞(Agent、PSF EIM)"),
    ("PSF EIM Service Matrix™ 跟 PSF Customer Success Agent™ 有什麼關聯？", True,
     "2個以上專有名詞"),
    ("PSF EIM 的五大產品矩陣是什麼？", False, "一般查詢"),
    ("企業智慧架構總共分成幾層？", False, "一般查詢，無英文專有名詞"),
    ("PSF EIM 的定價是多少？", False, "負向對照"),
    ("PSF EIM 的導入時程大概要多久？", False, "負向對照"),
    ("PSF EIM 的核心價值主張是什麼？", False, "一般查詢"),
]


@pytest.mark.parametrize("question,expected,note", KNOWN_THINK_CASES)
def test_detect_think_reason_known_cases(question, expected, note):
    should_think, reason = detect_think_reason(question)
    assert should_think == expected, (
        f"「{question}」預期 should_think={expected}({note})，"
        f"實際得到 {should_think}，reason={reason!r}"
    )
    if expected:
        assert reason is not None
    else:
        assert reason is None


def test_detect_think_reason_comparison_keyword_present_in_reason():
    _, reason = detect_think_reason("PSF EIM 跟一般 ERP/CRM 有什麼不同？")
    assert "不同" in reason


def test_detect_think_reason_entity_count_present_in_reason():
    _, reason = detect_think_reason("PSF EIM 裡負責客戶服務的 Agent 是誰？")
    assert "PSF EIM" in reason and "Agent" in reason


# ---------------------------------------------------------------------------
# split_think(): 照它自己 docstring 列的 4 種情境
# ---------------------------------------------------------------------------

def test_split_think_complete_block():
    visible, thinking = split_think("<think>先想一下</think>這是答案")
    assert visible == "這是答案"
    assert thinking == "先想一下"


def test_split_think_dangling_close_tag_no_open():
    # 有些 chat template 自己開 think block，模型輸出從沒出現過 <think>。
    visible, thinking = split_think("先想一下</think>這是答案")
    assert visible == "這是答案"
    assert thinking == "先想一下"


def test_split_think_unterminated_open_tag():
    # 生成在 max_tokens 內被截斷，<think> 沒有配對的結尾。
    visible, thinking = split_think("<think>想到一半就被截斷了")
    assert visible == ""
    assert thinking == "想到一半就被截斷了"


def test_split_think_no_tags_at_all():
    visible, thinking = split_think("這是答案，完全沒有推理標籤")
    assert visible == "這是答案，完全沒有推理標籤"
    assert thinking == ""


# ---------------------------------------------------------------------------
# build_context()
# ---------------------------------------------------------------------------

def test_build_context_empty_hits():
    assert build_context([]) == "", "空清單應返回空字串"


@pytest.mark.parametrize("payload,expected_heading", [
    ({"body": "content"}, "(無章節)"),
    ({"heading_path": "章節1", "body": "content"}, "章節1"),
    ({}, "(無章節)"),
])
def test_build_context_missing_heading_path(payload, expected_heading):
    hits = [{"payload": payload}]
    result = build_context(hits)
    assert f"[文件片段 1] {expected_heading}" in result, "應使用預設heading"


# body存在時使用body，否則用text，兩者都無時為空字串——後兩個 case 直接
# 比對整條 block 的完整內容，不是只檢查空字串是不是 result 的子字串（空
# 字串永遠是任何字串的子字串，那樣寫不管 body 是不是真的空的都會通過）。
@pytest.mark.parametrize("payload,expected_block", [
    ({"body": "body content"}, "[文件片段 1] (無章節)\nbody content"),
    ({"text": "text content"}, "[文件片段 1] (無章節)\ntext content"),
    ({"body": "", "text": ""}, "[文件片段 1] (無章節)\n"),
    ({}, "[文件片段 1] (無章節)\n"),
])
def test_build_context_body_text_handling(payload, expected_block):
    hits = [{"payload": payload}]
    result = build_context(hits)
    assert result == expected_block, "應正確處理body/text欄位，含body為空的情況"


@pytest.mark.parametrize("hit_count", [1, 2, 3])
def test_build_context_multiple_hits(hit_count):
    hits = [{"payload": {}} for _ in range(hit_count)]
    result = build_context(hits)
    blocks = result.split("\n\n")
    assert len(blocks) == hit_count, "應正確分割多個文件片段"
    for i, block in enumerate(blocks, 1):
        assert f"[文件片段 {i}]" in block, f"第{i}個片段應有正確編號"


def test_build_context_missing_payload():
    hits = [{}]
    result = build_context(hits)
    assert "[文件片段 1] (無章節)" in result, "無payload時應使用預設值"
