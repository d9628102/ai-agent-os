#!/usr/bin/env python3
"""GX10 控制台階段 1 的唯讀狀態腳本：蒐集 14 個狀態項目，輸出 status.json 與每日摘要。

規格：console-spec-stage0-1-20261005.md（含「審閱補充」與「原型補齊規則」）。
- 只用 Python 標準函式庫；不用 sudo；不寫入任何服務、不呼叫模型、不呼叫會變更狀態的指令。
- 缺資料不可誤判為綠：任何項目取不到資料最好也只是黃（備份例外：無法確認視為紅）；
  整體燈號取最嚴，只有「所有必要項目都有資料且都綠」才是綠；資訊項目（3、8、10）只在缺資料時算黃。
- 第 12 項（防火牆規則是否實際載入）需要 root：固定顯示「未確認（需管理權限）」，不判紅綠、不計入整體燈號，
  並在輸出固定帶「整體燈號不含防火牆規則檢查（需管理權限）」。
- status.json 超過 15 分鐘沒更新時，整體燈號直接紅（is_stale / evaluate）。
- 蒐集函式都從 env 取資料（指令、HTTP、檔案），測試用假 env，不碰正式服務；每個蒐集函式有逾時，
  丟例外或逾時時該項視為「無資料」。
- 輸出只含狀態、版本、數量與時間：寫出前一律經 sanitize() 白名單過濾，不含任何問答原文、金鑰、網址簽章或密碼值。
- n8n 統計（第 13 項）：在容器內以唯讀模式查資料庫、只取數量、固定逾時、每小時最多一次（結果快取）。
  這個蒐集函式只寫好；此版本沒有在任何正式服務上執行過，SQL 欄位名稱需要在部署前驗證。

Usage:
  python3 console_status.py --out <status.json 路徑>            # 蒐集並寫出 status.json（預設每 5 分鐘）
  python3 console_status.py --daily --out-dir <目錄>             # 每日摘要（預設每天 08:00），保留 14 份
  預設輸出目錄：~/trial/console（可用 --out、--out-dir 或環境變數 CONSOLE_OUT_DIR 覆蓋）。
  本檔不是由這次工作執行的；部署與 cron 另案核准。
"""
import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

NOTE = "整體燈號不含防火牆規則檢查（需管理權限）"
STALE_SECONDS = 15 * 60
FUTURE_SKEW_SECONDS = 60
COLLECT_TIMEOUT = 20
N8N_TIMEOUT = 10
N8N_CACHE_SECONDS = 3600
SUMMARY_KEEP = 14
SCHEMA_VERSION = 1

# (id, key, 名稱, 取得方式, 需要 sudo, 必要項目, 資訊項目)
ITEMS = [
    {"id": 1, "key": "containers", "name": "容器狀態、啟動時間、重啟次數", "info": False},
    {"id": 2, "key": "health", "name": "n8n、vLLM、嵌入、Qdrant 健康檢查", "info": False},
    {"id": 3, "key": "vllm_queue", "name": "vLLM 進行中／排隊請求", "info": True},
    {"id": 4, "key": "qdrant", "name": "Qdrant collection 點數與狀態", "info": False},
    {"id": 5, "key": "gpu", "name": "GPU 溫度、使用率", "info": False},
    {"id": 6, "key": "resources", "name": "系統記憶體、磁碟", "info": False},
    {"id": 7, "key": "backup", "name": "夜間備份最新狀態與失敗標記", "info": False},
    {"id": 8, "key": "retention", "name": "備份保留與磁碟用量", "info": True},
    {"id": 9, "key": "mac_pull", "name": "Mac 最近一次拉取時間", "info": False},
    {"id": 10, "key": "images", "name": "映像固定標籤是否仍在", "info": True},
    {"id": 11, "key": "fw_service", "name": "防火牆服務 active／啟用", "info": False},
    {"id": 12, "key": "fw_rules", "name": "防火牆規則是否實際載入", "info": False},
    {"id": 13, "key": "n8n_exec", "name": "n8n 24 小時執行統計（只取數量）", "info": False},
    {"id": 14, "key": "tests", "name": "還原驗證與測試結果", "info": False},
]

# 蒐集設定（預設值；部署時可由參數覆蓋）。不含任何憑證。
DEFAULT_CONFIG = {
    "containers": ("n8n", "vllm-server", "vllm-embed", "qdrant"),
    "health_urls": (
        ("n8n", "http://127.0.0.1:5678/healthz"),
        ("vllm", "http://127.0.0.1:8000/health"),
        ("embed", "http://127.0.0.1:8001/health"),
        ("qdrant", "http://127.0.0.1:6333/healthz"),
    ),
    "vllm_metrics_url": "http://127.0.0.1:8000/metrics",
    "qdrant_url": "http://127.0.0.1:6333",
    "qdrant_collection": "psf_eim_kb",
    "qdrant_baseline": 31,
    "pinned_images": ("qdrant/qdrant:v1.19.1", "n8nio/n8n:2.39.6"),
    "fw_unit": "gx10-firewall.service",
    "fw_fail_flag": "/run/gx10-fw-FAILED",
    "n8n_container": "n8n",
    "n8n_db_path": "/home/node/.n8n/database.sqlite",
    "n8n_logger_workflow": "PsfEimErrLog0001",
    "staging": "~/backup-staging",
    "serve_log": "~/trial/backup/serve.log",
    "restore_dir": "~/trial/backup",
    "disk_path": "/",
}


# ───────────────────────────── 判斷（紅黃綠）─────────────────────────────

def _is_bool(v):
    return isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0)


def _is_pos(v):
    """溫度、磁碟用量、點數基準這類「實際上不可能是 0」的值：0 視為取值失敗，不當作正常資料。"""
    return _is_num(v) and v > 0


def _nodata(extra=""):
    return {"color": "nodata", "text": "無資料" + extra}


def is_stale(generated_at_ms, now_ms):
    """狀態檔是否過期：沒有時間戳記、超過 15 分鐘沒更新、或時間戳記在未來（時鐘不可信）。"""
    if isinstance(generated_at_ms, bool) or not isinstance(generated_at_ms, (int, float)):
        return True
    if not math.isfinite(generated_at_ms):
        return True
    if now_ms - generated_at_ms > STALE_SECONDS * 1000:
        return True
    if generated_at_ms > now_ms + FUTURE_SKEW_SECONDS * 1000:
        return True
    return False


def evaluate_item(item, data):
    iid = item["id"]
    if iid == 12:
        return {"color": "unknown", "text": "未確認（需管理權限）"}
    v = data.get(item["key"]) if isinstance(data, dict) else None
    if not isinstance(v, dict):
        if iid == 7:
            return {"color": "red", "text": "無資料（備份無法確認，視為紅）"}
        return _nodata()

    def bad():
        if iid == 7:
            return {"color": "red", "text": "資料格式不正確（備份無法確認，視為紅）"}
        return _nodata("（格式不正確）")

    if iid == 1:
        if not (_is_bool(v.get("all_running")) and _is_bool(v.get("restart_increase")) and _is_bool(v.get("oom"))):
            return bad()
        names = "、".join(v["names"]) if isinstance(v.get("names"), list) else ""
        if not v["all_running"] or v["oom"]:
            return {"color": "red", "text": "有容器不是 running 或發生 OOM　" + names}
        if v["restart_increase"]:
            return {"color": "yellow", "text": "重啟次數增加　" + names}
        return {"color": "green", "text": "容器都 running，重啟次數不變　" + names}
    if iid == 2:
        checks = v.get("checks")
        if not isinstance(checks, list) or not checks:
            return bad()
        worst, parts = "green", []
        for c in checks:
            if not isinstance(c, dict) or not isinstance(c.get("name"), str):
                return bad()
            code, ms = c.get("code"), c.get("ms")
            if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
                return bad()
            if ms is not None and not _is_num(ms):
                return bad()
            parts.append("%s %s／%s" % (c["name"], "無回應" if code is None else code,
                                       "逾時" if ms is None else "%.1fs" % (ms / 1000)))
            if code != 200 or ms is None or ms > 15000:
                worst = "red"
            elif ms >= 5000 and worst != "red":
                worst = "yellow"
        return {"color": worst, "text": "；".join(parts)}
    if iid == 3:
        if not (_is_num(v.get("running")) and _is_num(v.get("waiting"))):
            return bad()
        return {"color": "info", "text": "進行中 %d、排隊 %d" % (v["running"], v["waiting"])}
    if iid == 4:
        if not (_is_bool(v.get("reachable")) and _is_bool(v.get("exists")) and isinstance(v.get("status"), str)
                and _is_num(v.get("points")) and _is_pos(v.get("baseline"))):
            return bad()
        t = "點數 %d（基準 %d）、狀態 %s" % (v["points"], v["baseline"], v["status"])
        if not v["reachable"] or not v["exists"] or v["status"] != "green":
            return {"color": "red", "text": t + "；連不上、collection 缺少或狀態不是 green"}
        if v["points"] != v["baseline"]:
            return {"color": "yellow", "text": t + "；點數與基準不同"}
        return {"color": "green", "text": t}
    if iid == 5:
        t = v.get("temp_c")
        if not _is_pos(t):
            return bad()
        color = "red" if t > 92 else "yellow" if t >= 85 else "green"
        return {"color": color, "text": "溫度 %s 度" % t}
    if iid == 6:
        d, a = v.get("disk_pct"), v.get("avail_gb")
        if not (_is_pos(d) and _is_num(a)) or d > 100:
            return bad()
        t = "磁碟 %s%%、可用記憶體 %s GB" % (d, a)
        if d > 90 or a < 3:
            return {"color": "red", "text": t}
        if d >= 80 or a <= 10:
            return {"color": "yellow", "text": t}
        return {"color": "green", "text": t}
    if iid == 7:
        f, h = v.get("failed"), v.get("hours_since_done")
        if not (_is_bool(f) and _is_num(h)):
            return bad()
        t = "距上次完成 %s 小時%s" % (h, "、有失敗標記" if f else "")
        if f or h > 36:
            return {"color": "red", "text": t}
        if h > 26:
            return {"color": "yellow", "text": t}
        return {"color": "green", "text": t}
    if iid == 8:
        if not (_is_num(v.get("staging_mb")) and _is_num(v.get("kept"))):
            return bad()
        return {"color": "info", "text": "保留 %d 份、約 %s MB" % (v["kept"], v["staging_mb"])}
    if iid == 9:
        h = v.get("hours_since_pull")
        if not _is_num(h):
            return _nodata("（讀不到拉取紀錄）")
        t = "距上次拉取 %s 小時" % h
        return {"color": "red" if h > 36 else "yellow" if h > 24 else "green", "text": t}
    if iid == 10:
        if not _is_bool(v.get("all_present")):
            return bad()
        return {"color": "info", "text": "固定標籤都在"} if v["all_present"] else {"color": "yellow", "text": "有固定標籤不見了"}
    if iid == 11:
        if not (_is_bool(v.get("active")) and _is_bool(v.get("enabled")) and _is_bool(v.get("fail_flag"))):
            return bad()
        if not v["active"] or v["fail_flag"]:
            return {"color": "red", "text": "服務不是 active 或有失敗旗標"}
        if not v["enabled"]:
            return {"color": "yellow", "text": "服務 active 但未啟用開機載入"}
        return {"color": "green", "text": "服務 active 且已啟用"}
    if iid == 13:
        if not (_is_bool(v.get("responsive")) and _is_num(v.get("failures_24h"))
                and _is_num(v.get("logger_triggers")) and _is_bool(v.get("waiting_increase"))):
            return bad()
        t = "24 小時失敗 %d 次、Error Logger 觸發 %d 次" % (v["failures_24h"], v["logger_triggers"])
        if not v["responsive"] or v["logger_triggers"] >= 3 or v["failures_24h"] >= 3:
            return {"color": "red", "text": t}
        if v["failures_24h"] >= 1 or v["logger_triggers"] >= 1 or v["waiting_increase"]:
            return {"color": "yellow", "text": t + ("；waiting 增加" if v["waiting_increase"] else "")}
        return {"color": "green", "text": t}
    if iid == 14:
        p, d = v.get("restore_passed"), v.get("days_since_restore")
        if not (_is_bool(p) and _is_num(d)):
            return bad()
        t = "還原驗證%s（%s 天前）" % ("通過" if p else "失敗", d)
        if not p:
            return {"color": "red", "text": t}
        if d > 14:
            return {"color": "yellow", "text": t}
        return {"color": "green", "text": t}
    return _nodata()


_ORDER = {"green": 0, "yellow": 1, "red": 2}


def evaluate(data, now_ms):
    """回傳 {items, stale, overall, complete, note}。整體燈號取最嚴；過期直接紅。"""
    rows = []
    for it in ITEMS:
        r = evaluate_item(it, data)
        rows.append({"id": it["id"], "name": it["name"], "color": r["color"], "text": r["text"]})
    gen = data.get("generated_at_ms") if isinstance(data, dict) else None
    stale = is_stale(gen, now_ms)
    worst, complete = 0, True
    for r in rows:
        if r["id"] == 12:
            continue
        if r["color"] == "nodata":
            complete = False
            worst = max(worst, _ORDER["yellow"])
        elif r["color"] in _ORDER:
            worst = max(worst, _ORDER[r["color"]])
        elif r["color"] not in ("info", "unknown"):
            complete = False
            worst = max(worst, _ORDER["yellow"])
    overall = ("green", "yellow", "red")[worst]
    if overall == "green" and not complete:
        overall = "yellow"
    if stale:
        overall = "red"
    return {"items": rows, "stale": stale, "overall": overall, "complete": complete, "note": NOTE}


# ───────────────────────────── 輸出過濾（只留白名單欄位）─────────────────────────────

_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,60}$")
_STATUS_ENUM = ("green", "yellow", "red", "grey")


def _clean(kind, v):
    """回傳 (ok, value)。"""
    if kind == "bool":
        return (True, v) if _is_bool(v) else (False, None)
    if kind == "num":
        return (True, v) if _is_num(v) else (False, None)
    if kind == "name":
        return (True, v) if isinstance(v, str) and _NAME_RE.match(v) else (False, None)
    if kind == "status":
        return (True, v) if v in _STATUS_ENUM else (False, None)
    if kind == "code":
        return (True, v) if v is None or (isinstance(v, int) and not isinstance(v, bool)) else (False, None)
    if kind == "numornone":
        return (True, v) if v is None or _is_num(v) else (False, None)
    return False, None


_FIELDS = {
    "containers": {"all_running": "bool", "restart_increase": "bool", "oom": "bool"},
    "vllm_queue": {"running": "num", "waiting": "num"},
    "qdrant": {"reachable": "bool", "exists": "bool", "status": "status", "points": "num", "baseline": "num"},
    "gpu": {"temp_c": "num", "util_pct": "num"},
    "resources": {"disk_pct": "num", "avail_gb": "num"},
    "backup": {"failed": "bool", "hours_since_done": "num"},
    "retention": {"staging_mb": "num", "kept": "num"},
    "mac_pull": {"hours_since_pull": "num"},
    "images": {"all_present": "bool"},
    "fw_service": {"active": "bool", "enabled": "bool", "fail_flag": "bool"},
    "n8n_exec": {"responsive": "bool", "failures_24h": "num", "logger_triggers": "num", "waiting_increase": "bool"},
    "tests": {"restore_passed": "bool", "days_since_restore": "num"},
}


def sanitize(data):
    """只保留白名單欄位與型別正確的值；其他（包含任何多餘欄位或文字）一律丟棄。"""
    out = {}
    if not isinstance(data, dict):
        return out
    gen = data.get("generated_at_ms")
    if isinstance(gen, int) and not isinstance(gen, bool):
        out["generated_at_ms"] = gen
    for key, fields in _FIELDS.items():
        v = data.get(key)
        if not isinstance(v, dict):
            continue
        clean = {}
        for f, kind in fields.items():
            if f in v:
                ok, val = _clean(kind, v[f])
                if ok:
                    clean[f] = val
        if key == "containers" and isinstance(v.get("names"), list):
            clean["names"] = [n for n in v["names"] if _clean("name", n)[0]]
        if clean:
            out[key] = clean
    h = data.get("health")
    if isinstance(h, dict) and isinstance(h.get("checks"), list):
        checks = []
        for c in h["checks"]:
            if isinstance(c, dict) and _clean("name", c.get("name"))[0]:
                okc, code = _clean("code", c.get("code"))
                okm, ms = _clean("numornone", c.get("ms"))
                if okc and okm:
                    checks.append({"name": c["name"], "code": code, "ms": ms})
        out["health"] = {"checks": checks}
    return out


def build_status(data, now_ms):
    """組出 status.json 的內容（已過濾）。"""
    clean = sanitize(data)
    clean["generated_at_ms"] = int(now_ms)
    ev = evaluate(clean, now_ms)
    return {
        "schema": SCHEMA_VERSION,
        "generated_at_ms": int(now_ms),
        "overall": ev["overall"],
        "complete": ev["complete"],
        "note": NOTE,
        "items": ev["items"],
        "data": clean,
    }


def render_summary(status, generated_text=None):
    """每日摘要（純文字，只有狀態、數量與時間）。"""
    names = {"green": "綠", "yellow": "黃", "red": "紅"}
    attention = [r for r in status["items"] if r["color"] in ("yellow", "red", "nodata")]
    lines = ["整體燈號：%s%s" % (names.get(status["overall"], "未知"),
                              "；需要留意：" + "、".join("第 %d 項%s" % (r["id"], r["name"]) for r in attention)
                              if attention else "；沒有需要留意的項目"),
             NOTE]
    sym = {"green": "綠", "yellow": "黃", "red": "紅", "nodata": "無資料", "info": "資訊", "unknown": "未確認"}
    for r in status["items"]:
        lines.append("%2d. [%s] %s：%s" % (r["id"], sym.get(r["color"], "?"), r["name"], r["text"]))
    if generated_text:
        lines.append("產生時間：" + generated_text)
    return "\n".join(lines) + "\n"


# ───────────────────────────── 檔案輸出 ─────────────────────────────

def write_atomic(path, text):
    """先寫暫存檔（同目錄、權限 600）再改名，避免讀到半份。"""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_SUMMARY_RE = re.compile(r"^daily-summary-(\d{8})\.txt$")


def prune_summaries(out_dir, today, keep=SUMMARY_KEEP):
    """只保留最近 keep 天的每日摘要（依檔名日期），只動符合檔名格式的檔案。"""
    removed = []
    for name in sorted(os.listdir(out_dir)):
        m = _SUMMARY_RE.match(name)
        if not m:
            continue
        try:
            d = datetime.datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if (today - d).days >= keep:
            os.unlink(os.path.join(out_dir, name))
            removed.append(name)
    return removed


# ───────────────────────────── 蒐集 ─────────────────────────────

def call_with_timeout(fn, timeout):
    """在 daemon 執行緒裡跑 fn；逾時或丟例外一律回 None（該項視為無資料）。"""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except Exception:
            box["v"] = None

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None
    return box.get("v")


class RealEnv:
    """真正的環境：指令一律用參數陣列（不經 shell）、固定逾時。只在部署後的狀態腳本使用。"""

    def now(self):
        return time.time()

    def run(self, argv, timeout):
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, shell=False)
        return r.returncode, r.stdout

    def http_get(self, url, timeout):
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read(2_000_000).decode("utf-8", "replace")
                return r.status, int((time.time() - t0) * 1000), body
        except urllib.error.HTTPError as e:
            return e.code, int((time.time() - t0) * 1000), ""
        except Exception:
            return None, None, ""

    def read_text(self, path):
        with open(os.path.expanduser(path), encoding="utf-8", errors="replace") as f:
            return f.read(2_000_000)

    def mtime(self, path):
        return os.stat(os.path.expanduser(path)).st_mtime

    def isfile(self, path):
        return os.path.isfile(os.path.expanduser(path))

    def listdir(self, path):
        return os.listdir(os.path.expanduser(path))

    def disk_pct(self, path):
        s = os.statvfs(path)
        used = (s.f_blocks - s.f_bfree) * s.f_frsize
        total = used + s.f_bavail * s.f_frsize
        return round(100.0 * used / total, 1) if total else None

    def tree_size_mb(self, path):
        total = 0
        for root, _, files in os.walk(os.path.expanduser(path)):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return round(total / 1048576, 1)


def parse_metrics(text):
    """從 Prometheus 文字取 vLLM 進行中／排隊請求數（只取數字）。"""
    vals = {}
    for line in text.splitlines():
        m = re.match(r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", line)
        if m:
            vals[m.group(1)] = int(float(m.group(2)))
    if "running" not in vals or "waiting" not in vals:
        raise ValueError("metrics 缺欄位")
    return vals


def parse_gpu(text):
    parts = [p.strip() for p in text.strip().splitlines()[0].split(",")]
    temp, util = float(parts[0]), float(parts[1])
    return {"temp_c": temp, "util_pct": util}


def parse_meminfo_avail_gb(text):
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.M)
    if not m:
        raise ValueError("沒有 MemAvailable")
    return round(int(m.group(1)) / 1048576, 1)


_PULL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) get \d{8}-\d{6}$")


def parse_last_pull(text):
    """回傳最後一次 `get <名稱>` 的本機時間戳記（epoch 秒）；沒有就丟 ValueError。"""
    last = None
    for line in text.splitlines():
        m = _PULL_RE.match(line.strip())
        if m:
            last = m.group(1)
    if last is None:
        raise ValueError("沒有 get 紀錄")
    return time.mktime(datetime.datetime.strptime(last, "%Y-%m-%d %H:%M:%S").timetuple())


def collect_containers(env, cfg, baseline=None):
    names = list(cfg["containers"])
    rc, out = env.run(["docker", "inspect"] + names, N8N_TIMEOUT)
    if rc != 0:
        raise RuntimeError("docker inspect 失敗")
    info = json.loads(out)
    by_name = {i["Name"].lstrip("/"): i for i in info}
    all_running = all(n in by_name and by_name[n]["State"]["Running"] is True for n in names)
    oom = any(bool(by_name[n]["State"].get("OOMKilled")) for n in names if n in by_name)
    counts = {n: int(by_name[n].get("RestartCount", 0)) for n in names if n in by_name}
    if isinstance(baseline, dict):
        increase = any(counts.get(n, 0) > int(baseline.get(n, 0)) for n in counts)
    elif all(c == 0 for c in counts.values()):
        increase = False
    else:
        increase = None  # 沒有基準又有重啟紀錄：無法判斷，視為無資料
    return {"all_running": all_running, "restart_increase": increase, "oom": oom,
            "names": [n for n in names if n in by_name], "restart_counts": counts}


def collect_health(env, cfg):
    checks = []
    for name, url in cfg["health_urls"]:
        code, ms, _ = env.http_get(url, COLLECT_TIMEOUT)
        checks.append({"name": name, "code": code, "ms": ms})
    return {"checks": checks}


def collect_vllm_queue(env, cfg):
    code, _, body = env.http_get(cfg["vllm_metrics_url"], COLLECT_TIMEOUT)
    if code != 200:
        raise RuntimeError("metrics 取不到")
    return parse_metrics(body)


def collect_qdrant(env, cfg):
    code, _, body = env.http_get("%s/collections/%s" % (cfg["qdrant_url"], cfg["qdrant_collection"]), COLLECT_TIMEOUT)
    base = cfg["qdrant_baseline"]
    if code is None:
        return {"reachable": False, "exists": False, "status": "grey", "points": 0, "baseline": base}
    if code == 404:
        return {"reachable": True, "exists": False, "status": "grey", "points": 0, "baseline": base}
    if code != 200:
        raise RuntimeError("Qdrant 回應異常")
    res = json.loads(body)["result"]
    st = res.get("status")
    return {"reachable": True, "exists": True, "status": st if st in _STATUS_ENUM else "grey",
            "points": int(res.get("points_count", 0)), "baseline": base}


def collect_gpu(env, cfg):
    rc, out = env.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu",
                       "--format=csv,noheader,nounits"], N8N_TIMEOUT)
    if rc != 0:
        raise RuntimeError("nvidia-smi 失敗")
    return parse_gpu(out)


def collect_resources(env, cfg):
    pct = env.disk_pct(cfg["disk_path"])
    if pct is None:
        raise RuntimeError("磁碟用量取不到")
    return {"disk_pct": pct, "avail_gb": parse_meminfo_avail_gb(env.read_text("/proc/meminfo"))}


def collect_backup(env, cfg):
    st = cfg["staging"]
    latest = env.read_text(st + "/LATEST").strip()
    if not re.match(r"^[0-9]{8}-[0-9]{6}$", latest):
        raise ValueError("LATEST 格式不合法")
    done = env.mtime("%s/%s/DONE" % (st, latest))
    return {"failed": env.isfile(st + "/FAILED"),
            "hours_since_done": round(max(0.0, (env.now() - done) / 3600), 1)}


def collect_retention(env, cfg):
    st = cfg["staging"]
    kept = [n for n in env.listdir(st) if re.match(r"^[0-9]{8}-[0-9]{6}$", n)]
    return {"kept": len(kept), "staging_mb": env.tree_size_mb(st)}


def collect_mac_pull(env, cfg):
    ts = parse_last_pull(env.read_text(cfg["serve_log"]))
    return {"hours_since_pull": round(max(0.0, (env.now() - ts) / 3600), 1)}


def collect_images(env, cfg):
    rc, out = env.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], N8N_TIMEOUT)
    if rc != 0:
        raise RuntimeError("docker images 失敗")
    have = set(out.split())
    return {"all_present": all(t in have for t in cfg["pinned_images"])}


_UNIT_WORDS = {"active", "inactive", "failed", "activating", "deactivating", "reloading",
               "enabled", "disabled", "static", "masked", "indirect", "generated", "alias", "linked"}


def collect_fw_service(env, cfg):
    r1, active = env.run(["systemctl", "is-active", cfg["fw_unit"]], N8N_TIMEOUT)
    r2, enabled = env.run(["systemctl", "is-enabled", cfg["fw_unit"]], N8N_TIMEOUT)
    a, e = active.strip(), enabled.strip()
    if a not in _UNIT_WORDS or e not in _UNIT_WORDS:
        raise ValueError("systemctl 輸出無法辨識")
    return {"active": a == "active", "enabled": e == "enabled", "fail_flag": env.isfile(cfg["fw_fail_flag"])}


# 容器內唯讀查詢，只輸出數量。部署前需在測試資料庫驗證欄位名稱（此版本沒有在正式服務上執行過）。
N8N_QUERY_JS = (
    "const {DatabaseSync}=require('node:sqlite');"
    "const db=new DatabaseSync(process.argv[1],{readOnly:true});"
    "const since=new Date(Date.now()-86400000).toISOString().replace('T',' ').replace('Z','');"
    "const n=(q,p)=>db.prepare(q).get(...p).c;"
    "console.log(JSON.stringify({"
    "failures_24h:n(\"SELECT COUNT(*) c FROM execution_entity WHERE status='error' AND startedAt>=?\",[since]),"
    "logger_triggers:n('SELECT COUNT(*) c FROM execution_entity WHERE workflowId=? AND startedAt>=?',[process.argv[2],since]),"
    "waiting:n(\"SELECT COUNT(*) c FROM execution_entity WHERE status='waiting'\",[])}));"
)


def query_n8n_counts(env, cfg):
    """容器內唯讀查詢，只回傳原始數量（失敗次數、Error Logger 觸發次數、waiting 數量）。"""
    rc, out = env.run(["docker", "exec", cfg["n8n_container"], "node", "-e", N8N_QUERY_JS,
                       cfg["n8n_db_path"], cfg["n8n_logger_workflow"]], N8N_TIMEOUT)
    if rc != 0:
        raise RuntimeError("n8n 統計查詢失敗")
    q = json.loads(out.strip().splitlines()[-1])
    return {"failures_24h": int(q["failures_24h"]), "logger_triggers": int(q["logger_triggers"]),
            "waiting": int(q["waiting"])}


def build_n8n_exec(counts, responsive, waiting_baseline=None):
    """用「目前」的基準重新算 waiting 是否增加；快取只存原始數量，不存這個判斷，所以基準一改馬上生效。"""
    waiting = counts["waiting"]
    increase = waiting > waiting_baseline if isinstance(waiting_baseline, int) else waiting > 0
    return {"responsive": responsive, "failures_24h": counts["failures_24h"],
            "logger_triggers": counts["logger_triggers"], "waiting_increase": increase, "waiting": waiting}


def collect_n8n_exec(env, cfg, responsive, waiting_baseline=None):
    return build_n8n_exec(query_n8n_counts(env, cfg), responsive, waiting_baseline)


def _cached_counts(cache):
    """取出快取裡的原始數量；也相容舊格式（舊版把整包結果放在 value 裡）。格式不對就當作沒有快取。"""
    src = cache.get("counts") if isinstance(cache.get("counts"), dict) else cache.get("value")
    if not isinstance(src, dict):
        return None
    out = {}
    for k in ("failures_24h", "logger_triggers", "waiting"):
        v = src.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return None
        out[k] = v
    return out


def collect_tests(env, cfg):
    d = cfg["restore_dir"]
    files = sorted(n for n in env.listdir(d) if re.match(r"^restore-test-[0-9]{8}-[0-9]{6}\.json$", n))
    if not files:
        raise ValueError("沒有還原驗證紀錄")
    rep = json.loads(env.read_text("%s/%s" % (d, files[-1])))
    when = datetime.datetime.strptime(rep["tested_at"], "%Y-%m-%dT%H:%M:%S").timestamp()
    if not isinstance(rep.get("passed"), bool):
        raise ValueError("passed 欄位不是布林")
    return {"restore_passed": rep["passed"], "days_since_restore": round(max(0.0, (env.now() - when) / 86400), 1)}


def collect_all(env, cfg=None, state=None, timeout=COLLECT_TIMEOUT):
    """蒐集全部項目；每項有逾時，失敗或逾時的項目不放進結果（評估時視為無資料）。回傳 (data, new_state)。"""
    cfg = cfg or DEFAULT_CONFIG
    state = dict(state or {})
    now = env.now()
    data = {"generated_at_ms": int(now * 1000)}
    jobs = [
        ("containers", lambda: collect_containers(env, cfg, state.get("restart_baseline"))),
        ("health", lambda: collect_health(env, cfg)),
        ("vllm_queue", lambda: collect_vllm_queue(env, cfg)),
        ("qdrant", lambda: collect_qdrant(env, cfg)),
        ("gpu", lambda: collect_gpu(env, cfg)),
        ("resources", lambda: collect_resources(env, cfg)),
        ("backup", lambda: collect_backup(env, cfg)),
        ("retention", lambda: collect_retention(env, cfg)),
        ("mac_pull", lambda: collect_mac_pull(env, cfg)),
        ("images", lambda: collect_images(env, cfg)),
        ("fw_service", lambda: collect_fw_service(env, cfg)),
        ("tests", lambda: collect_tests(env, cfg)),
    ]
    for key, fn in jobs:
        v = call_with_timeout(fn, timeout)
        if v is not None:
            data[key] = v
    if "containers" in data:
        state["restart_counts"] = data["containers"].get("restart_counts", {})
    health = data.get("health", {}).get("checks", [])
    responsive = any(c["name"] == "n8n" and c["code"] == 200 for c in health)
    counts = None
    cache = state.get("n8n_cache")
    if isinstance(cache, dict) and now - cache.get("at", 0) < N8N_CACHE_SECONDS:
        counts = _cached_counts(cache)
    if counts is None:
        counts = call_with_timeout(lambda: query_n8n_counts(env, cfg), N8N_TIMEOUT + 5)
        if counts is not None:
            state["n8n_cache"] = {"at": now, "counts": counts}
    if counts is not None:
        data["n8n_exec"] = build_n8n_exec(counts, responsive, state.get("waiting_baseline"))
    return data, state


# ───────────────────────────── 進入點 ─────────────────────────────

def default_out_dir():
    return os.path.expanduser(os.environ.get("CONSOLE_OUT_DIR", "~/trial/console"))


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def run(out_path, daily=False, out_dir=None, env=None, cfg=None):
    env = env or RealEnv()
    out_dir = out_dir or os.path.dirname(os.path.abspath(out_path))
    state_path = os.path.join(out_dir, "state.json")
    state = load_state(state_path)
    data, state = collect_all(env, cfg, state)
    now_ms = int(env.now() * 1000)
    status = build_status(data, now_ms)
    write_atomic(out_path, json.dumps(status, ensure_ascii=False, indent=1) + "\n")
    if daily:
        today = datetime.datetime.fromtimestamp(env.now()).date()
        text = render_summary(status, today.isoformat())
        write_atomic(os.path.join(out_dir, "daily-summary-%s.txt" % today.strftime("%Y%m%d")), text)
        write_atomic(os.path.join(out_dir, "daily-summary.txt"), text)
        prune_summaries(out_dir, today)
        state["restart_baseline"] = state.get("restart_counts", state.get("restart_baseline", {}))
        if isinstance(state.get("n8n_cache"), dict):
            c = _cached_counts(state["n8n_cache"])
            if c is not None:
                state["waiting_baseline"] = c["waiting"]
    write_atomic(state_path, json.dumps({k: v for k, v in state.items()}, ensure_ascii=False) + "\n")
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description="GX10 控制台唯讀狀態腳本")
    ap.add_argument("--out", help="status.json 輸出路徑（預設 <out-dir>/status.json）")
    ap.add_argument("--out-dir", help="輸出目錄（預設 ~/trial/console）")
    ap.add_argument("--daily", action="store_true", help="同時輸出每日摘要並保留 14 份")
    args = ap.parse_args(argv)
    out_dir = os.path.expanduser(args.out_dir) if args.out_dir else default_out_dir()
    out = os.path.expanduser(args.out) if args.out else os.path.join(out_dir, "status.json")
    run(out, daily=args.daily, out_dir=os.path.dirname(os.path.abspath(out)) if args.out else out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
