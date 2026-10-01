#!/usr/bin/env python3
"""Mutation harness for tests/test_gx10_backup_mac.py.

Works on a temporary copy (scripts/gx10-backup-mac + the test); the real files are never touched.
Each mutation must make the test file fail. Usage: python3 tests/mutate_gx10_backup_mac.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sub(old, new):
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


def main():
    ok, survivors = 0, []
    for label, fname, fn in M:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "tests").mkdir()
            shutil.copytree(ROOT / "scripts" / "gx10-backup-mac", td / "scripts" / "gx10-backup-mac")
            shutil.copy(ROOT / "tests" / "test_gx10_backup_mac.py", td / "tests")
            target = td / "scripts" / "gx10-backup-mac" / fname
            target.write_text(fn(target.read_text(encoding="utf-8")) if target.exists() else fn(""), encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(td / "tests" / "test_gx10_backup_mac.py")], capture_output=True, text=True)
            killed = r.returncode != 0
            ok += killed
            if not killed:
                survivors.append(label)
            print("%s  %s" % ("抓到" if killed else "沒抓到", label))
    print("\n抓到 %d／總共 %d" % (ok, len(M)))
    if survivors:
        print("沒抓到：", survivors)
    sys.exit(0 if not survivors else 1)


if __name__ == "__main__":
    main()
