"""三個變異檢查腳本共用的防呆（tests/mutate_console_status.py、tests/mutate_console_pull.py、tests/mutate_gx10_backup_mac.py）。

為什麼要有：變異檢查的「抓到」必須代表「測試真的偵測到行為改變」，不能是副作用造成的假象。以前的腳本只看
pytest 的結束碼不是 0 就算抓到，所以「複本少了一個檔案」「匯入失敗」「逾時」「突變沒有改變任何內容」都會被誤算成抓到。
這個模組補兩道保護，三個腳本共用：
  a. 跑突變之前，先跑「未突變的複本」，必須全部通過才繼續（run_baseline）；沒通過就中止（結束碼 3）。
  b. 每個突變的結果要分類（run_pytest）：
     - 抓到（caught）：至少一個測試斷言失敗。主要數字只報這一類。
     - 執行期例外（runtime_exception）：突變後的程式在測試本體執行期間拋出非斷言例外，使測試失敗（例如 TypeError、KeyError、
       複本完整時讀不到突變後才去讀的檔案）。它不是「抓到」也不是「無效結果」，單獨計數、逐個列出，但不使整體失敗。
     - 沒抓到（survived）：測試全部通過。
     - 無效結果（invalid）：收集錯誤、匯入失敗、逾時、結束碼 2 以上、設定錯誤、複本不完整或內容未改變等，
       使整體結果失敗（結束碼 4），不會被算成其他任何一類。
本模組只做複製、比對與執行 pytest，不會執行突變後的程式碼以外的任何東西；它自己的檢查（self_test、check_harness）
只用無害的小型假專案，不執行任何突變程式碼。
"""
import hashlib
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

OK, CAUGHT, SURVIVED, INVALID, RUNTIME = "ok", "caught", "survived", "invalid", "runtime_exception"
IMPORT_ERRORS = ("ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError")      # 匯入失敗類：算無效結果
ASSERT_PREFIXES = ("assert", "AssertionError", "Failed:")      # 斷言失敗在 junit 報告裡的訊息開頭
DEFAULT_TIMEOUT = 300


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def list_files(root, rel_dir):
    """資料夾內的所有檔案（相對於 root），排除 __pycache__ 與 .pyc。"""
    base = Path(root) / rel_dir
    return sorted(p.relative_to(root) for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def copy_files(root, td, rels, drop=None):
    for rel in rels:
        if drop is not None and Path(rel) == Path(drop):
            continue
        dst = Path(td) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(root) / rel, dst)


def verify_copy(root, td, rels, mutated_rel=None):
    """檢查複本是否完整：每個應有的檔案都在，且除了被突變的那一個，內容與原檔相同。回傳問題清單。"""
    problems = []
    for rel in rels:
        dst = Path(td) / rel
        if not dst.is_file():
            problems.append("缺少 %s" % rel)
        elif mutated_rel is None or Path(rel) != Path(mutated_rel):
            if _sha(dst) != _sha(Path(root) / rel):
                problems.append("內容與原檔不同 %s" % rel)
    return problems


def run_pytest(td, test_rel, timeout=DEFAULT_TIMEOUT):
    """在 td 裡跑 test_rel，回傳 (類別, 說明)；類別：OK（全部通過）、CAUGHT（至少一個斷言失敗）、RUNTIME（只有非斷言的執行期例外）、INVALID。

    複本完整性在呼叫前已由 verify_copy 確認，所以測試本體裡的 FileNotFoundError 來自突變後的行為，算執行期例外；
    匯入失敗（ImportError、ModuleNotFoundError、SyntaxError）算無效結果。
    """
    xml = Path(td) / "_junit.xml"
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml), str(Path(td) / test_rel)],
                           capture_output=True, text=True, timeout=timeout, cwd=str(td))
    except subprocess.TimeoutExpired:
        return INVALID, "逾時"
    if r.returncode == 0:
        return OK, "全部通過"
    if r.returncode not in (1,):
        return INVALID, "pytest 結束碼 %d（收集錯誤、匯入失敗、沒有收集到測試或內部錯誤）" % r.returncode
    if not xml.is_file():
        return INVALID, "沒有產生報告"
    try:
        root = ET.parse(str(xml)).getroot()
    except ET.ParseError:
        return INVALID, "報告無法解析"
    asserts, others, errors = 0, [], 0
    for case in root.iter("testcase"):
        for child in case:
            if child.tag == "error":
                errors += 1
            elif child.tag == "failure":
                msg = (child.get("message") or "").lstrip()
                if msg.startswith(ASSERT_PREFIXES):
                    asserts += 1
                else:
                    others.append(msg.split(":")[0][:40] or "?")
    if errors:
        return INVALID, "有 %d 個收集或設定錯誤" % errors
    if asserts:
        return CAUGHT, "%d 個測試斷言失敗" % asserts
    kinds = sorted(set(others))
    if any(k in IMPORT_ERRORS for k in kinds):
        return INVALID, "匯入失敗（%s）" % ", ".join(kinds[:3])
    return RUNTIME, "執行期例外（%s）" % ", ".join(kinds[:3])


def run_baseline(root, rels, test_rels, runner=None, timeout=DEFAULT_TIMEOUT):
    """未突變的複本必須全部通過。回傳 (是否通過, 說明)。"""
    runner = runner or run_pytest
    with tempfile.TemporaryDirectory() as td:
        copy_files(root, td, rels)
        problems = verify_copy(root, td, rels)
        if problems:
            return False, "複本不完整：" + "；".join(problems)
        for t in test_rels:
            cat, detail = runner(td, t, timeout)
            if cat != OK:
                return False, "未突變的複本沒有全部通過（%s：%s）" % (t, detail)
    return True, ""


def evaluate(root, rels, test_rel, target_rel, fn, drop=None, runner=None, timeout=DEFAULT_TIMEOUT):
    """套用一個突變並跑測試。回傳 (類別, 說明)；類別：CAUGHT、SURVIVED、INVALID。

    drop：測試用，故意讓複本少一個檔案（應回報無效結果，且不得執行測試）。
    """
    runner = runner or run_pytest
    with tempfile.TemporaryDirectory() as td:
        copy_files(root, td, rels, drop=drop)
        problems = verify_copy(root, td, rels)
        if problems:
            return INVALID, "複本不完整：" + "；".join(problems)
        target = Path(td) / target_rel
        is_new = Path(target_rel) not in [Path(r) for r in rels]
        original = "" if is_new else target.read_text(encoding="utf-8")
        mutated = fn(original)
        if mutated == original:
            return INVALID, "突變沒有改變任何內容"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(mutated, encoding="utf-8")
        problems = verify_copy(root, td, rels, mutated_rel=target_rel)
        if problems:
            return INVALID, "套用突變後複本不完整：" + "；".join(problems)
        cat, detail = runner(td, test_rel, timeout)
        if cat == OK:
            return SURVIVED, detail
        return cat, detail


def summarize(results):
    """results：[(標籤, 類別, 說明)]。印出摘要並回傳結束碼：0 沒有沒抓到也沒有無效結果；1 有沒抓到；4 有無效結果（優先）。

    執行期例外不使整體失敗，但會逐個列出；主要數字（「斷言抓到」）不含它。
    """
    caught = [r for r in results if r[1] == CAUGHT]
    runtime = [r for r in results if r[1] == RUNTIME]
    survived = [r for r in results if r[1] == SURVIVED]
    invalid = [r for r in results if r[1] == INVALID]
    print("\n總數 %d；斷言抓到 %d；執行期例外 %d；沒抓到 %d；無效結果 %d" % (len(results), len(caught), len(runtime), len(survived), len(invalid)))
    if runtime:
        print("執行期例外（突變後的程式在測試中拋出非斷言例外；不算斷言抓到，也不使整體失敗）：")
        for label, _, detail in runtime:
            print("  -", label, "：", detail)
    if survived:
        print("沒抓到：")
        for label, _, detail in survived:
            print("  -", label)
    if invalid:
        print("無效結果（不算抓到，整體結果失敗）：")
        for label, _, detail in invalid:
            print("  -", label, "：", detail)
    return 4 if invalid else (1 if survived else 0)


# ───────────────────────────── 自我檢查（只用無害的小型假專案）─────────────────────────────

SYNTHETIC = {
    "全部通過": ("def test_a():\n    assert 1 == 1\n", OK),
    "斷言失敗": ("def test_a():\n    assert 1 == 2\n", CAUGHT),
    "斷言失敗（附訊息）": ("def test_a():\n    assert 1 == 2, 'x'\n", CAUGHT),
    "pytest.fail": ("import pytest\n\ndef test_a():\n    pytest.fail('x')\n", CAUGHT),
    "匯入失敗": ("import module_that_does_not_exist_xyz\n\ndef test_a():\n    assert True\n", INVALID),
    "測試本體找不到檔案（複本完整時屬執行期例外）": ("def test_a():\n    open('no_such_file_xyz.txt')\n", RUNTIME),
    "測試本體匯入失敗": ("def test_a():\n    import module_that_does_not_exist_xyz\n", INVALID),
    "語法錯誤（收集錯誤）": ("def test_a(:\n    pass\n", INVALID),
    "非斷言例外": ("def test_a():\n    {}['k']\n", RUNTIME),
    "多個非斷言例外": ("def test_a():\n    {}['k']\n\ndef test_b():\n    1 / 0\n", RUNTIME),
    "斷言失敗加匯入失敗例外": ("def test_a():\n    assert 1 == 2\n\ndef test_b():\n    import module_that_does_not_exist_xyz\n", CAUGHT),
    "非斷言例外加匯入失敗例外": ("def test_a():\n    {}['k']\n\ndef test_b():\n    import module_that_does_not_exist_xyz\n", INVALID),
    "沒有收集到測試": ("x = 1\n", INVALID),
    "設定錯誤": ("import pytest\n\n@pytest.fixture\ndef f():\n    raise RuntimeError('x')\n\ndef test_a(f):\n    assert True\n", INVALID),
    "斷言失敗加非斷言例外": ("def test_a():\n    assert 1 == 2\n\ndef test_b():\n    {}['k']\n", CAUGHT),
    "逾時": ("import time\n\ndef test_a():\n    time.sleep(30)\n", INVALID),
}


def self_test():
    """用小型假專案驗證 run_pytest 的分類。回傳問題清單（空清單＝正常）。"""
    problems = []
    for name, (src, want) in SYNTHETIC.items():
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "test_x.py").write_text(src, encoding="utf-8")
            got, detail = run_pytest(td, "test_x.py", timeout=5 if name == "逾時" else 60)
        if got != want:
            problems.append("%s：預期 %s，實際 %s（%s）" % (name, want, got, detail))
    return problems


def _boom(*a, **k):
    raise AssertionError("不應該執行測試")


def check_harness(root, rels, test_rel, target_rel, drop_rel):
    """驗證一個變異檢查腳本的保護有效（不執行任何突變程式碼）。回傳問題清單。"""
    problems = []
    try:
        cat, detail = evaluate(root, rels, test_rel, target_rel, lambda t: t + "\n", drop=drop_rel, runner=_boom)
        if cat != INVALID or "複本不完整" not in detail:
            problems.append("複本少一個檔案時沒有回報無效結果（%s：%s）" % (cat, detail))
    except AssertionError:
        problems.append("複本少一個檔案時仍然去執行測試")
    try:
        cat, detail = evaluate(root, rels, test_rel, target_rel, lambda t: t, runner=_boom)
        if cat != INVALID or "沒有改變" not in detail:
            problems.append("突變沒有改變內容時沒有回報無效結果（%s：%s）" % (cat, detail))
    except AssertionError:
        problems.append("突變沒有改變內容時仍然去執行測試")
    ok, detail = run_baseline(root, rels, [test_rel], runner=lambda td, t, timeout: (INVALID, "x"))
    if ok:
        problems.append("未突變的複本沒通過時沒有中止")
    seen = []

    def spy(td, t, timeout):
        seen.append(verify_copy(root, td, rels))
        return OK, ""
    ok, detail = run_baseline(root, rels, [test_rel], runner=spy)
    if not ok or seen != [[]]:
        problems.append("未突變的複本不是原檔的完整複本")
    for forced, want in ((CAUGHT, CAUGHT), (OK, SURVIVED), (INVALID, INVALID), (RUNTIME, RUNTIME)):
        cat, _ = evaluate(root, rels, test_rel, target_rel, lambda t: t + "\n# m\n", runner=lambda td, t, timeout, f=forced: (f, ""))
        if cat != want:
            problems.append("結果分類錯誤：%s 應為 %s，實際 %s" % (forced, want, cat))
    return problems
