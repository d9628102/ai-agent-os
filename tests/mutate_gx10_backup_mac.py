#!/usr/bin/env python3
"""Mutation harness for tests/test_gx10_backup_mac.py.

Works on a temporary copy (scripts/gx10-backup-mac + the test); the real files are never touched.
Each mutation must make the test file fail with an assertion (see tests/mutation_guard.py: unmutated baseline must pass first;
collection/import errors, timeouts and incomplete copies are "invalid"; non-assertion runtime exceptions are a separate
"runtime_exception" category, never counted as caught).
Usage: python3 tests/mutate_gx10_backup_mac.py   (--check-only: static check only; --self-test: prove the guards work)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_guard as guard
from mutation_guard import CAUGHT, SURVIVED, INVALID, RUNTIME

REPL = []        # 所有突變的替換文字，供開跑前的靜態自我檢查掃描


def sub(old, new):
    REPL.append(new)

    def f(t):
        assert old in t, "mutation anchor missing: %r" % old
        return t.replace(old, new, 1)
    return f


HOST_LINE = 'HOST="${GX10_BACKUP_HOST:?請設定 GX10_BACKUP_HOST，例如 user@host}"'
SH, PL, RD = "gx10-backup.sh", "com.psf.gx10-backup.plist.example", "README.md"
M = [
    ("HOST 放回真實位址樣式", SH, sub(HOST_LINE, 'HOST="tester@192.168.1.10"')),
    ("HOST 加預設值", SH, sub(HOST_LINE, 'HOST="${GX10_BACKUP_HOST:-user@host}"')),
    ("HOST 拿掉必填檢查", SH, sub(HOST_LINE, 'HOST="${GX10_BACKUP_HOST}"')),
    ("HOST 寫死字串", SH, sub(HOST_LINE, 'HOST="user@host"')),
    ("必填檢查移到建立資料夾之後", SH, lambda t: sub(HOST_LINE + "\n", "")(t).replace("\nnow() {", "\n" + HOST_LINE + "\nnow() {", 1)),
    ("KEY 預設寫死使用者路徑", SH, sub('${GX10_BACKUP_KEY:-$HOME/.ssh/gx10_backup}', '${GX10_BACKUP_KEY:-/Users/someone/.ssh/gx10_backup}')),
    ("KEY 不再可由環境變數覆蓋", SH, sub('KEY="${GX10_BACKUP_KEY:-$HOME/.ssh/gx10_backup}"', 'KEY="$HOME/.ssh/gx10_backup"')),
    ("DEST 不再可由環境變數覆蓋", SH, sub('DEST="${GX10_BACKUP_DEST:-$HOME/GX10Backup}"', 'DEST="$HOME/GX10Backup"')),
    ("ssh 不再使用 HOST", SH, sub('"$HOST" latest', '"nobody@nowhere" latest')),
    ("腳本語法錯誤", SH, lambda t: t + "\nif then fi (\n"),
    ("腳本混入使用者名稱 psf01", SH, lambda t: t + "\n# owner: psf01\n"),
    ("腳本混入 /home 路徑", SH, lambda t: t + "\n# /home/someone/x\n"),
    ("plist 範例混入真實位址樣式", PL, sub("user@host", "tester@192.168.1.10")),
    ("plist 範例換成真實使用者路徑", PL, sub("YOUR_USER", "alice")),
    ("plist 範例拿掉 HOST 變數", PL, sub("GX10_BACKUP_HOST", "SOMETHING_ELSE")),
    ("plist 範例破壞 XML", PL, sub("</dict>\n</plist>", "</plist>")),
    ("plist 範例改成每小時執行", PL, sub("<key>Hour</key><integer>9</integer>", "<key>Hour</key><integer>10</integer>")),
    ("README 混入 psf01", RD, lambda t: t + "\n帳號 psf01\n"),
    ("README 混入內網位址", RD, lambda t: t + "\n主機 192.168.1.10\n"),
    ("README 拿掉強制指令說明", RD, lambda t: t.replace("gx10_backup_serve.py", "xxx")),
    ("混入多餘檔案", "extra.txt", lambda t: "x"),
]


# 會把輸入交給外殼或執行任意程式的寫法：任何突變的替換文字都不得含有
FORBIDDEN = re.compile(r"(?i)os\.system|\.system\s*\(|subprocess|popen|\beval\b|\bexec|shell|sh\s+-c")
PKG = Path("scripts") / "gx10-backup-mac"
TEST_REL = Path("tests") / "test_gx10_backup_mac.py"


def check_replacements():
    """執行任何突變之前的靜態自我檢查：掃描所有突變的替換文字，含危險字樣就中止，不執行任何測試。"""
    bad = [t[:60] for t in REPL if FORBIDDEN.search(t)]
    if bad:
        print("靜態自我檢查失敗：下列突變的替換文字含有會執行輸入的寫法，已中止，沒有執行任何測試：")
        for b in bad:
            print("  -", repr(b))
        sys.exit(2)
    print("靜態自我檢查通過：%d 個突變的替換文字都沒有執行輸入的寫法" % len(REPL))


def all_rels():
    return guard.list_files(ROOT, PKG) + [TEST_REL]


def baseline(runner=None):
    """未突變的複本必須全部通過。"""
    return guard.run_baseline(ROOT, all_rels(), [TEST_REL], runner)


def evaluate_mutation(fname, fn, drop=None, runner=None):
    return guard.evaluate(ROOT, all_rels(), TEST_REL, PKG / fname, fn, drop=drop, runner=runner)


def self_check():
    """證明兩道保護有效（不執行任何突變程式碼）。回傳問題清單。"""
    return guard.self_test() + guard.check_harness(ROOT, all_rels(), TEST_REL, PKG / SH, PKG / RD)


def main():
    check_replacements()
    if "--self-test" in sys.argv:
        problems = self_check()
        print("保護自我檢查：%s" % ("全部正常" if not problems else "有問題"))
        for q in problems:
            print("  -", q)
        sys.exit(0 if not problems else 5)
    if "--check-only" in sys.argv:
        return
    ok, detail = baseline()
    if not ok:
        print("中止：" + detail)
        sys.exit(3)
    print("未突變的複本全部通過，開始跑突變")
    results = []
    for label, fname, fn in M:
        cat, detail = evaluate_mutation(fname, fn)
        results.append((label, cat, detail))
        print("%s  %s" % ({CAUGHT: "抓到", RUNTIME: "執行期例外", SURVIVED: "沒抓到", INVALID: "無效結果"}[cat], label), flush=True)
    sys.exit(guard.summarize(results))


if __name__ == "__main__":
    main()
