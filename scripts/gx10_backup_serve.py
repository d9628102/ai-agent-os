#!/usr/bin/env python3
"""Mac 拉取備份用的「強制指令」：放在 authorized_keys 那一行的 command="…" 裡。

為什麼不用 rrsync：macOS 內建的 rsync（openrsync）一定會多送 --dirs 選項，rrsync 3.2.7 會拒絕
（實測：invalid rsync-command syntax or options）。這支只允許三個唯讀動作，其他一律拒絕：
  latest           印出 STAGING/LATEST（最新一份備份的名稱）
  failed           印出 STAGING/FAILED（GX10 端整理備份失敗的標記）；不存在時結束碼 3
  get <名稱>       把 STAGING/<名稱>/ 打成 tar 串流輸出；名稱必須是 YYYYMMDD-HHMMSS、資料夾存在、
                   而且有 DONE 標記（沒完成的備份不給）
名稱用嚴格格式比對，所以不可能用 .. 或斜線讀到 STAGING 以外的東西；不寫入任何東西、不執行任何外部指令。
每次呼叫寫一行 log（時間、動作、結果，不含備份內容）。
結束碼：0 成功；2 不允許的指令或名稱不合法；3 沒有這個檔；4 備份沒有 DONE。
"""
import datetime
import os
import re
import sys
import tarfile

STAGING = "/home/psf01/backup-staging"
LOG = "/home/psf01/trial/backup/serve.log"
NAME_RE = re.compile(r"^[0-9]{8}-[0-9]{6}$")


def _log(msg, log_path):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def serve(original_command, out, err, staging=STAGING, log_path=LOG):
    parts = (original_command or "").split()
    action = parts[0] if parts else ""
    if action in ("latest", "failed") and len(parts) == 1:
        p = os.path.join(staging, "LATEST" if action == "latest" else "FAILED")
        if not os.path.isfile(p):
            _log(f"{action} 不存在", log_path)
            return 3
        out.write(open(p, "rb").read())
        _log(action, log_path)
        return 0
    if action == "get" and len(parts) == 2:
        name = parts[1]
        if not NAME_RE.match(name):
            err.write(b"invalid name\n")
            _log("get 名稱格式不合法", log_path)
            return 2
        d = os.path.join(staging, name)
        if not os.path.isdir(d):
            _log(f"get {name} 不存在", log_path)
            return 3
        if not os.path.isfile(os.path.join(d, "DONE")):
            _log(f"get {name} 沒有 DONE", log_path)
            return 4
        with tarfile.open(fileobj=out, mode="w|") as t:
            t.add(d, arcname=name)
        _log(f"get {name}", log_path)
        return 0
    err.write(b"not allowed\n")
    _log("拒絕不允許的指令", log_path)
    return 2


def main():
    return serve(os.environ.get("SSH_ORIGINAL_COMMAND", ""), sys.stdout.buffer, sys.stderr.buffer)


if __name__ == "__main__":
    sys.exit(main())
