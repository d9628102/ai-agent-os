"""
測試重點：
- find_orphan_think_tags()：驗證孤立結尾標籤的偵測——這是測試生成代理在
  生成後做的機械攔截，抓的是「沒有配對開頭標籤的孤立結尾標籤」。八種輸入
  各自驗證回傳的 1-based 行號清單，不只驗證有沒有抓到東西
- strip_think_and_fence()：驗證剝殼的邊界判斷。這個函式連續修過三次，
  三個修正各自對應一種真實踩過的失敗，這裡每一種都留一條測試釘住：
  (1) 測試資料裡合法配對的標籤不能被連帶刪掉（原本全文取代會刪掉，草稿
      語法還是對的、測試還會過，但已經測不到剝殼行為——靜默失效）
  (2) 推理散文裡行內引用結尾標籤、後面同一行還有字，不能被誤認成區塊結尾
      （非貪婪取第一個會誤認，症狀是「invalid character」這種指錯方向的
      語法錯誤）
  (3) 推理散文把引用寫在行尾，同樣不能被誤認（「必須是該行最後一個東西」
      這個啟發式就是被這個形狀打敗的）
  現在的作法是用「切掉之後剩下的內容能不能當成 Python 解析」決定邊界，
  上面三種形狀都應該正確處理

這份檔案是手寫的，不是 test_generation_agent.py 生成的——實測過這兩個
目標函式沒辦法用生成代理產出草稿：它們的原始碼本身就在講 think 標籤，
模型推理時必然引用這些標籤字串，而這兩個標籤在對話模板裡是特殊標記，
生成會在中途斷掉（finish_reason 回報 stop、檔案只寫到一半）或被自動
換成別的標籤。githooks/pre-push 的註解本來就允許手寫測試檔，人工逐條
審核斷言這個把關點沒有因此被跳過。

標籤字串一律用字串相加組出來（OPEN/CLOSE 兩個常數），不在這個檔案裡直接
寫出完整字面值。這不是風格潔癖，是為了這個檔案之後被當成 --style-reference
餵給生成代理時，不會把「直接寫字面標籤」這個寫法示範出去——實測過模型看到
要輸出字面標籤時，會自己換成別的標籤（例如把 think 換成 x），產出語法正確
但測錯東西的假案例。

- main() 的孤立標籤重試控制流程（第一次孤立就重試一次、兩次都孤立就中止
  不寫檔、第一次乾淨就只呼叫一次）：這三個情境是這份檔案新增的部分，補回
  之前只用一次性腳本手動驗證過、沒有收進正式測試的案例
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from test_generation_agent import find_orphan_think_tags, strip_think_and_fence

OPEN = "<" + "think>"
CLOSE = "</" + "think>"


@pytest.mark.parametrize("code,expected_lines,description", [
    ("x = 1\n", [], "完全沒有標籤"),
    (f'A = "{OPEN}推理{CLOSE}"\n', [], "正常配對一組"),
    (f'A = "{CLOSE}"\n', [1], "單一孤立結尾標籤"),
    (f'A = "{OPEN}x{CLOSE}"\nB = "{CLOSE}"\n', [2], "配對一組之後又多一個孤立"),
    (f'A = "{CLOSE}"\nB = "{OPEN}x{CLOSE}"\n', [1],
     "孤立在前、配對在後——順序不能救它，前面那個仍算孤立"),
    (f'A = "{CLOSE}"\nB = "{CLOSE}"\n', [1, 2], "兩個孤立，回傳兩個行號"),
    (f'A = "{OPEN}外{OPEN}內{CLOSE}{CLOSE}"\n', [], "巢狀配對不算孤立"),
    (f'A = "{OPEN}沒有結尾"\n', [],
     "只有開頭沒結尾——刻意不攔，交給呼叫端的語法檢查處理"),
])
def test_find_orphan_think_tags(code, expected_lines, description):
    assert find_orphan_think_tags(code) == expected_lines, description


def test_find_orphan_think_tags_line_numbers_are_1_based():
    """行號要能真的定位到出問題的那一行——孤立標籤在第三行時要回傳 3，
    不是 2（0-based）也不是清單長度這種無意義的值。"""
    code = f'A = 1\nB = 2\nC = "{CLOSE}"\n'
    assert find_orphan_think_tags(code) == [3]


PAIRED_DATA = f'CASE = "{OPEN}推理{CLOSE}"'


@pytest.mark.parametrize("raw,expected,description", [
    ("Y = 2\n", "Y = 2", "沒有推理區塊，原樣回傳（只 strip 前後空白）"),
    ("```python\nZ = 3\n```", "Z = 3", "code fence 含 python 標記"),
    ("```\nZ = 4\n```", "Z = 4", "code fence 不含標記"),
    (f"{OPEN}\n推理內容。\n{CLOSE}\n\nX = 1\n", "X = 1", "推理區塊結尾自己佔一行"),
    (f"{OPEN}簡短推理{CLOSE}\nX = 2\n", "X = 2", "推理區塊開頭結尾同一行"),
    (f"\n\n{OPEN}推理{CLOSE}\nZ = 6", "Z = 6", "推理區塊前面有空白換行仍剝得掉"),
    (f"{OPEN}推理{CLOSE}\n```python\nZ = 5\n```", "Z = 5", "推理區塊加 code fence"),
])
def test_strip_think_and_fence_basic(raw, expected, description):
    assert strip_think_and_fence(raw) == expected, description


def test_strip_preserves_paired_tags_in_test_data():
    """修正 (1)：測試資料裡合法配對的標籤必須原封不動保留。

    原本是全文取代所有配對標籤，會把這種測試資料連同推理區塊一起刪掉——
    草稿語法還是對的、測試還會過，但已經測不到剝殼行為。這種靜默失效還讓
    管線系統性偏袒錯誤樣式：寫對的被刪掉、寫錯的（孤立結尾標籤）原封不動
    存活，是孤立標籤問題連續四輪重複出現的成因之一。
    """
    raw = f"{OPEN}\n推理。\n{CLOSE}\n\n{PAIRED_DATA}\n"
    assert strip_think_and_fence(raw) == PAIRED_DATA


def test_strip_ignores_inline_tag_mention_followed_by_more_text():
    """修正 (2)：推理散文裡行內引用結尾標籤、後面同一行還有字，不能被誤認
    成推理區塊的結尾。非貪婪取第一個結尾標籤就是踩在這裡，症狀是剩下的
    推理散文被當成程式碼、以「invalid character」這種指錯方向的語法錯誤
    收場。"""
    raw = f"{OPEN}\n要找沒有對應開頭的{CLOSE}）這種標籤。\n第二段。\n{CLOSE}\n\nX = 3\n"
    assert strip_think_and_fence(raw) == "X = 3"


def test_strip_ignores_inline_tag_mention_at_end_of_line():
    """修正 (3)：推理散文把引用寫在行尾，同樣不能被誤認。「結尾標籤必須是
    該行最後一個東西」這個啟發式就是被這個形狀打敗的，所以改成用「剩下的
    內容能不能當成 Python 解析」判斷。"""
    raw = f"{OPEN}\n這個函式找的是孤立的結束標籤 {CLOSE}\n再想一下。\n{CLOSE}\n\nX = 4\n"
    assert strip_think_and_fence(raw) == "X = 4"


def test_strip_handles_inline_mention_and_paired_test_data_together():
    """兩種情況同時出現：推理散文引用了標籤，而程式碼裡又有合法配對的測試
    資料。正確結果是推理整段消失、測試資料完整保留。"""
    raw = f"{OPEN}\n提到 {CLOSE} 這個標籤。\n{CLOSE}\n\n{PAIRED_DATA}\n"
    assert strip_think_and_fence(raw) == PAIRED_DATA


@pytest.mark.parametrize("raw,description", [
    (f"{OPEN}\n推理斷了", "被 max_tokens 截斷，沒有結尾標籤"),
    (f"{OPEN}\n推理。\n{CLOSE}\n\n這一段還是中文散文，不是程式碼。\n",
     "有結尾標籤但後面不是合法 Python"),
])
def test_strip_returns_input_unchanged_when_no_valid_code(raw, description):
    """剝不出合法 Python 時原樣回傳（只 strip 前後空白），讓呼叫端的語法
    檢查去報錯——不要自己猜一個切法、產出半殘的草稿。"""
    assert strip_think_and_fence(raw) == raw.strip(), description


# ---------------------------------------------------------------------------
# main() 的孤立標籤重試控制流程：抓到孤立標籤要重新生成一次；兩次都有就
# 中止、完全不寫草稿檔；第一次就乾淨的話只呼叫模型一次。用 monkeypatch
# 假的 call_llm（不燒真的模型呼叫）跑一次完整的 main()，直接驗證呼叫
# 次數、是否中止、草稿檔是否寫出/寫出的內容是否已經沒有孤立標籤。
#
# 這三個情境先前用一支一次性驗證腳本手動跑過一次、全部正確，但當時只有
# find_orphan_think_tags() 的 8 個單元案例被收進這份檔案，控制流程本身
# 的 3 個情境沒有——這裡補上，並補做原本沒有做過的反向驗證（mutation
# testing）。
# ---------------------------------------------------------------------------

import test_generation_agent as _tga

PAIRED_DRAFT = (
    '"""測試"""\n'
    'import pytest\n\n'
    '@pytest.mark.parametrize("raw", [\n'
    f'    "{OPEN}一些推理{CLOSE}\\n{{ ok }}",\n'
    '])\n'
    'def test_paired(raw):\n'
    '    assert raw\n'
)

ORPHAN_DRAFT = (
    '"""測試"""\n'
    'import pytest\n\n'
    '@pytest.mark.parametrize("raw", [\n'
    f'    "{CLOSE}\\n{{ ok }}",\n'
    '])\n'
    'def test_orphan(raw):\n'
    '    assert raw\n'
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FLOW_DRAFT_REL = "tests/_testgen_flow_DRAFT.py"
_FLOW_DRAFT_FULL = os.path.join(_REPO_ROOT, _FLOW_DRAFT_REL)


@pytest.fixture
def cleanup_flow_draft():
    yield
    if os.path.exists(_FLOW_DRAFT_FULL):
        os.remove(_FLOW_DRAFT_FULL)


def _run_main_with_fake_outputs(monkeypatch, fake_outputs):
    """用假的 call_llm 跑一次 main()，回傳 (是否中止, 呼叫次數, 草稿內容或None)。"""
    calls = {"n": 0}

    def fake_call_llm(messages, max_tokens=8192, timeout=300):
        calls["n"] += 1
        return fake_outputs[min(calls["n"] - 1, len(fake_outputs) - 1)]

    monkeypatch.setattr(_tga, "call_llm", fake_call_llm)
    monkeypatch.setattr(_tga, "run_pytest_on_draft", lambda *a, **k: (0, "(測試用：略過真的跑 pytest)"))
    if os.path.exists(_FLOW_DRAFT_FULL):
        os.remove(_FLOW_DRAFT_FULL)

    monkeypatch.setattr(sys, "argv", [
        "test_generation_agent.py",
        "--target", "scripts/test_generation_agent.py:find_orphan_think_tags",
        "--style-reference", "tests/test_scoring.py",
        "--output", _FLOW_DRAFT_REL,
        "--repo-root", _REPO_ROOT,
    ])

    died = False
    try:
        _tga.main()
    except SystemExit:
        died = True

    content = None
    if os.path.exists(_FLOW_DRAFT_FULL):
        with open(_FLOW_DRAFT_FULL, "r", encoding="utf-8") as f:
            content = f.read()
    return died, calls["n"], content


def test_main_retries_once_then_succeeds_on_clean_second_draft(monkeypatch, cleanup_flow_draft):
    died, calls, content = _run_main_with_fake_outputs(monkeypatch, [ORPHAN_DRAFT, PAIRED_DRAFT])
    assert died is False, "第二次生成乾淨，不該中止"
    assert calls == 2, "第一次孤立要觸發剛好一次重新生成"
    assert content is not None, "最終應該寫出草稿檔"
    assert find_orphan_think_tags(content) == [], "寫出的草稿不該再有孤立標籤"


def test_main_aborts_without_writing_draft_when_both_attempts_orphaned(monkeypatch, cleanup_flow_draft):
    died, calls, content = _run_main_with_fake_outputs(monkeypatch, [ORPHAN_DRAFT, ORPHAN_DRAFT])
    assert died is True, "兩次都孤立，人工審核不該花時間看這種機械性錯誤，要直接中止"
    assert calls == 2, "應該剛好重試一次（不是無限重試）"
    assert content is None, "中止時完全不該寫出草稿檔"


def test_main_calls_model_only_once_when_first_draft_is_clean(monkeypatch, cleanup_flow_draft):
    died, calls, content = _run_main_with_fake_outputs(monkeypatch, [PAIRED_DRAFT])
    assert died is False
    assert calls == 1, "第一次就乾淨時不該多花一次生成"
    assert content is not None
