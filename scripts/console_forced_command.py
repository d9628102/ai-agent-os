#!/usr/bin/env python3
"""控制台（Mac 端）用的「強制指令」：放在 authorized_keys 那一行的 command="…" 裡（`gx10_console` 專用金鑰）。

只允許兩個動作，且必須「完全相同」的字串（不去頭尾空白、不分大小寫變化、不接受任何參數）：
  status    輸出已產生的 status.json
  summary   輸出已產生的 daily-summary.txt
其他一律拒絕（含空輸入、超長輸入、大小寫變化、含空白／換行／分號／反引號／路徑符號的輸入），
結束碼 2，並寫一行 log。
只讀已經產生的檔案：不呼叫 docker 或任何外部程式，不把 SSH_ORIGINAL_COMMAND 傳給任何 shell
（本檔不引用 subprocess 或 os.system，也不以外殼方式執行任何東西）。log 只記動作與結果；被拒絕的輸入只記截短、跳脫後的樣貌。
結束碼：0 成功；2 不允許的指令；3 檔案不存在或讀不到。

Usage:
  不是手動執行的腳本：由 sshd 依 ~/.ssh/authorized_keys 的 command="/usr/bin/python3 <本檔路徑>" 呼叫，
  動作從環境變數 SSH_ORIGINAL_COMMAND 讀取（Mac 端用 ssh 帶入下列其中一個動作）：
    ssh <帳號>@<GX10> status
    ssh <帳號>@<GX10> summary
  讀取目錄預設 ~/trial/console（可用環境變數 CONSOLE_OUT_DIR 覆蓋）；唯一寫入的是同目錄的 serve.log。
  執行身分：該 authorized_keys 所屬的一般使用者，不需要 root。
  這個檔案這次只放進 repo，沒有部署、沒有動 authorized_keys。
"""
import datetime
import os
import sys

MAX_BYTES = 1_000_000
FILES = {"status": "status.json", "summary": "daily-summary.txt"}


def _default_dir():
    return os.path.expanduser(os.environ.get("CONSOLE_OUT_DIR", "~/trial/console"))


def _safe(text, limit=40):
    """被拒絕的輸入：截短並用 ascii() 跳脫，確保 log 永遠只有一行、沒有控制字元。"""
    return ascii(text[:limit])


def _log(msg, log_path):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("%s %s\n" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    except OSError:
        pass


def serve(original_command, out, err, out_dir=None, log_path=None):
    out_dir = out_dir or _default_dir()
    log_path = log_path or os.path.join(out_dir, "serve.log")
    if not isinstance(original_command, str):
        original_command = ""
    if original_command in FILES:
        path = os.path.join(out_dir, FILES[original_command])
        try:
            with open(path, "rb") as f:
                body = f.read(MAX_BYTES)
        except OSError:
            err.write(b"not available\n")
            _log("%s 檔案不存在或讀不到" % original_command, log_path)
            return 3
        out.write(body)
        _log(original_command, log_path)
        return 0
    err.write(b"not allowed\n")
    _log("拒絕 %s" % _safe(original_command), log_path)
    return 2


def main():
    return serve(os.environ.get("SSH_ORIGINAL_COMMAND", ""), sys.stdout.buffer, sys.stderr.buffer)


if __name__ == "__main__":
    sys.exit(main())
