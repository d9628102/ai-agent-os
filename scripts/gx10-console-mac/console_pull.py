#!/usr/bin/env python3
"""Mac 端：從 GX10 拉取 status.json、驗證、處理過期與拉取失敗、產生靜態狀態頁、決定要不要通知。

設計原則（規格：console-spec-stage0-1-20261005.md 與其補充）：
1. 燈號的唯一來源是 GX10 產生的 status.json。這裡不重新計算任何項目的紅黃綠，只做：驗證格式、處理過期、
   處理拉取失敗、顯示、通知決策。（Mac 端唯一自己加的規則：過期、拉取失敗、資料格式異常、尚無資料。）
2. 格式驗證：欄位缺漏、未知燈號值、型別錯誤都不得顯示為綠；視為紅並標示「資料格式異常」。
3. 過期：資料產生時間距今超過 15 分鐘，或比現在晚超過 60 秒，整體燈號改為紅並顯示紅色橫幅。
4. 拉取失敗：保留上一份成功的資料並標示「拉取失敗」（整體燈號至少黃，不會是綠），與過期規則一起計算；
   從未成功拉取過顯示「尚無資料」紅。
5. 通知：只在整體燈號「進入紅色」時通知一次；持續紅色時每 6 小時最多再提醒一次；回到非紅色後重置；
   拉取連續失敗超過 15 分鐘也算紅色。通知函式與拉取函式都是注入的（預設用 ssh 與 osascript），測試不會執行它們。
6. 固定說明「整體燈號不含防火牆規則檢查（需管理權限）」一定顯示；第 12 項固定「未確認」，不判紅綠。
7. 頁面是單一靜態 HTML：不載入任何外部資源；所有資料都經 html.escape 當成純文字；唯一的內嵌小程式只用 textContent，
   且不含任何資料（用來在 Mac 排程停止時，讓舊頁面自己標示過期）。每次執行重新產生，原子寫入。
8. 輸出只含狀態、版本、數量與時間，不含問答原文、金鑰、網址簽章、密碼值（顯示前會再過濾一次）。

設定（環境變數）：
  GX10_CONSOLE_HOST  必填，沒有預設值，例如 user@host（GX10 上被限制成強制指令的那個帳號與主機）
  GX10_CONSOLE_KEY   選填，預設 $HOME/.ssh/gx10_console（專用金鑰檔路徑）
  GX10_CONSOLE_DIR   選填，預設 $HOME/GX10Console（輸出資料夾，權限 700，檔案 600）
  GX10_CONSOLE_SSH   選填，預設 /usr/bin/ssh（ssh 的絕對路徑；必須是絕對路徑，格式限制同上，不能含 ..）

Usage:
  GX10_CONSOLE_HOST=user@host python3 console_pull.py          # 拉取一次並更新 $GX10_CONSOLE_DIR/status.html
  python3 console_pull.py --preview-dir <資料夾>               # 只用假資料產生預覽頁，不連線、不通知
輸出檔（都在輸出資料夾）：status.html（狀態頁）、last_good.json（上一份成功且格式正確的資料）、
  state.json（通知與失敗時間的狀態）、last_run.txt（最近一次執行的一行結果；覆寫，不會無限成長）。
結束碼：0 完成（即使拉取失敗也會更新頁面並回 0）；2 設定錯誤（沒有設定主機等）。
"""
import argparse
import datetime
import html
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

NOTE = "整體燈號不含防火牆規則檢查（需管理權限）"
ITEM12_TEXT = "未確認（需管理權限）"
STALE_SECONDS = 15 * 60
FUTURE_SKEW_SECONDS = 60
REMIND_SECONDS = 6 * 3600
FAIL_RED_SECONDS = 15 * 60
SCHEMA_VERSION = 1
MAX_TEXT = 300
MAX_NAME = 100
PULL_TIMEOUT = 30
DEFAULT_SSH = "/usr/bin/ssh"      # 絕對路徑：launchd 的 PATH 很小，不依賴 PATH 找 ssh

OVERALL_COLORS = ("green", "yellow", "red")
ITEM_COLORS = ("green", "yellow", "red", "nodata", "info", "unknown")
COLOR_LABEL = {"green": "綠", "yellow": "黃", "red": "紅", "nodata": "無資料", "info": "資訊", "unknown": "未確認"}
CSS_CLASS = {"green": "c-green", "yellow": "c-yellow", "red": "c-red", "nodata": "c-nodata", "info": "c-info", "unknown": "c-unknown"}
ITEM_NAMES = {
    1: "容器狀態、啟動時間、重啟次數", 2: "n8n、vLLM、嵌入、Qdrant 健康檢查", 3: "vLLM 進行中／排隊請求",
    4: "Qdrant collection 點數與狀態", 5: "GPU 溫度、使用率", 6: "系統記憶體、磁碟",
    7: "夜間備份最新狀態與失敗標記", 8: "備份保留與磁碟用量", 9: "Mac 最近一次拉取時間",
    10: "映像固定標籤是否仍在", 11: "防火牆服務 active／啟用", 12: "防火牆規則是否實際載入",
    13: "n8n 24 小時執行統計（只取數量）", 14: "還原驗證與測試結果",
}

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@:-]*$")      # 不得以 - 開頭，避免被當成 ssh 選項
_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{10,}"), re.compile(r"AKIA[0-9A-Z]{12,}"), re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[a-z]-[A-Za-z0-9-]{10,}"), re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN[A-Z ]*-----"), re.compile(r"(?i)signature=\S+"), re.compile(r"(?i)bearer\s+\S+"),
    re.compile(r"https?://\S+"),
]
_SSH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")                # 覆蓋值必須是絕對路徑，只允許字母、數字與 . _ / -
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


# ───────────────────────────── 驗證 ─────────────────────────────

def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_status(obj):
    """回傳問題清單（空清單＝格式正確）。問題描述只含欄位名稱，不回顯內容。"""
    if not isinstance(obj, dict):
        return ["不是物件"]
    problems = []
    if not _is_int(obj.get("schema")) or obj.get("schema") != SCHEMA_VERSION:
        problems.append("schema")
    g = obj.get("generated_at_ms")
    if not _is_int(g) or g <= 0:
        problems.append("generated_at_ms")
    if obj.get("overall") not in OVERALL_COLORS:
        problems.append("overall")
    if not isinstance(obj.get("complete"), bool):
        problems.append("complete")
    items = obj.get("items")
    if not isinstance(items, list) or len(items) != 14:
        problems.append("items")
        return problems
    seen = set()
    for it in items:
        if not isinstance(it, dict):
            problems.append("item 型別")
            continue
        iid = it.get("id")
        if not _is_int(iid) or iid < 1 or iid > 14 or iid in seen:
            problems.append("item id")
            continue
        seen.add(iid)
        if not isinstance(it.get("name"), str) or len(it["name"]) > MAX_NAME:
            problems.append("item %d name" % iid)
        if it.get("color") not in ITEM_COLORS:
            problems.append("item %d color" % iid)
        if not isinstance(it.get("text"), str) or len(it["text"]) > MAX_TEXT * 3:
            problems.append("item %d text" % iid)
        if iid == 12 and it.get("color") != "unknown":
            problems.append("item 12 必須是未確認")
    if seen != set(range(1, 15)):
        problems.append("item 不齊全")
    return problems


def scrub(text):
    """顯示前再過濾一次：去掉控制字元、遮蔽看起來像金鑰或網址的內容、限制長度。"""
    if not isinstance(text, str):
        return ""
    text = _CTRL_RE.sub(" ", text)
    for p in _SECRET_PATTERNS:
        text = p.sub("[已隱藏]", text)
    return text[:MAX_TEXT]


def is_stale(generated_at_ms, now_ms):
    """資料過期：沒有／型別錯誤的時間、超過 15 分鐘沒更新、或比現在晚超過 60 秒（時鐘不可信）。"""
    if not _is_int(generated_at_ms):
        return True
    if now_ms - generated_at_ms > STALE_SECONDS * 1000:
        return True
    if generated_at_ms > now_ms + FUTURE_SKEW_SECONDS * 1000:
        return True
    return False


# ───────────────────────────── 檢視模型（不重算任何項目的燈號）─────────────────────────────

def make_view(current, last_good, now_ms, pull_failed=False, format_bad=False, fail_since_ms=None):
    """current：這次拉到且格式正確的 status（沒有就 None）；last_good：{"status":…, "fetched_ms":…} 或 None。

    整體燈號沿用 GX10 給的值，Mac 端只在下列情況改動：資料格式異常、過期、尚無資料、拉取連續失敗超過 15 分鐘（紅）；
    拉取失敗（其他情況）至少黃，不會是綠。
    """
    if current is not None:
        base, source, fetched = current, "fresh", now_ms
    elif last_good is not None:
        base, source, fetched = last_good["status"], "kept", last_good.get("fetched_ms")
    else:
        base, source, fetched = None, "none", None
    banners = []
    force_red = False
    if format_bad:
        banners.append(("red", "資料格式異常"))
        force_red = True
    elif pull_failed:
        banners.append(("yellow", "拉取失敗（顯示的是上一份成功的資料）" if base is not None else "拉取失敗"))
    if pull_failed and fail_since_ms is not None and now_ms - fail_since_ms > FAIL_RED_SECONDS * 1000:
        banners.append(("red", "拉取連續失敗超過 15 分鐘"))
        force_red = True
    stale = False
    if base is not None:
        stale = is_stale(base.get("generated_at_ms"), now_ms)
        if stale:
            banners.append(("red", "狀態資料過期（超過 15 分鐘沒有更新，或時間不可信）"))
            force_red = True
    else:
        banners.append(("red", "尚無資料"))
        force_red = True
    if force_red:
        overall = "red"
    else:
        overall = base["overall"]
        if pull_failed and overall == "green":
            overall = "yellow"
    rows = []
    for iid in range(1, 15):
        if iid == 12:
            rows.append({"id": 12, "name": ITEM_NAMES[12], "color": "unknown", "text": ITEM12_TEXT})
            continue
        src = None
        if base is not None:
            src = next((i for i in base["items"] if i.get("id") == iid), None)
        if src is None:
            rows.append({"id": iid, "name": ITEM_NAMES[iid], "color": "nodata", "text": "無資料"})
        else:
            rows.append({"id": iid, "name": scrub(src["name"]) or ITEM_NAMES[iid], "color": src["color"], "text": scrub(src["text"])})
    red_items = [r["id"] for r in rows if r["color"] == "red"] if base is not None else []
    return {"overall": overall, "banners": banners, "rows": rows, "source": source, "stale": stale,
            "generated_ms": base.get("generated_at_ms") if base is not None else None, "fetched_ms": fetched,
            "red_items": red_items}


# ───────────────────────────── 頁面 ─────────────────────────────

PAGE_SCRIPT = (
    "(function(){var b=document.body;var r=Number(b.getAttribute('data-rendered-ms'));var d=Date.now()-r;"
    "if(!(d<=900000&&d>=-60000)){var s=document.getElementById('stale-banner');s.hidden=false;"
    "s.textContent='頁面超過 15 分鐘沒有更新，或時間不可信（Mac 端排程可能已停止）';"
    "var o=document.getElementById('overall');o.textContent='紅';o.className='badge c-red';}})();"
)

PAGE_CSS = (
    "body{font-family:-apple-system,'PingFang TC',sans-serif;margin:24px;color:#222;background:#fafafa}"
    "h1{font-size:20px;margin:0 0 12px}.banner{padding:10px 14px;margin:8px 0;border-radius:6px;font-weight:600}"
    ".banner.red{background:#fde2e1;color:#8a1111}.banner.yellow{background:#fff3cd;color:#664d03}"
    ".banner.preview{background:#e3e8ff;color:#1c2b8a}.badge{display:inline-block;padding:4px 14px;border-radius:14px;font-weight:700}"
    ".c-green{background:#d1f0d8;color:#14532d}.c-yellow{background:#fff3cd;color:#664d03}.c-red{background:#fde2e1;color:#8a1111}"
    ".c-nodata{background:#e5e7eb;color:#374151}.c-info{background:#dbeafe;color:#1e3a8a}.c-unknown{background:#ede9fe;color:#4c1d95}"
    "table{border-collapse:collapse;width:100%;max-width:980px;margin-top:12px}td,th{border-bottom:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}"
    ".note{margin-top:14px;font-weight:600}.meta{color:#555;margin-top:8px;font-size:13px}"
)


def _fmt_time(ms):
    if not _is_int(ms):
        return "—"
    return datetime.datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def render_page(view, now_ms, preview=False):
    esc = html.escape
    out = []
    out.append('<!DOCTYPE html>\n<html lang="zh-Hant"><head><meta charset="utf-8">')
    out.append('<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'">')
    out.append('<meta http-equiv="refresh" content="60"><title>GX10 狀態</title><style>%s</style></head>' % PAGE_CSS)
    out.append('<body data-rendered-ms="%d"><h1>GX10 狀態</h1>' % int(now_ms))
    if preview:
        out.append('<div class="banner preview">預覽（假資料）：這一頁不是真實狀態，只用來檢視版面。</div>')
    for kind, text in view["banners"]:
        out.append('<div class="banner %s">%s</div>' % ("red" if kind == "red" else "yellow", esc(text)))
    if not preview:
        out.append('<div id="stale-banner" class="banner red" hidden></div>')
    cls = CSS_CLASS[view["overall"]]
    out.append('<p>整體燈號：<span id="overall" class="badge %s">%s</span></p>' % (cls, esc(COLOR_LABEL[view["overall"]])))
    out.append('<table><thead><tr><th>#</th><th>項目</th><th>燈號</th><th>說明</th></tr></thead><tbody>')
    for r in view["rows"]:
        out.append('<tr><td>%d</td><td>%s</td><td><span class="badge %s">%s</span></td><td>%s</td></tr>' % (
            r["id"], esc(r["name"]), CSS_CLASS[r["color"]], esc(COLOR_LABEL[r["color"]]), esc(r["text"])))
    out.append('</tbody></table>')
    out.append('<p class="note">%s</p>' % esc(NOTE))
    out.append('<p class="meta">資料產生時間：%s｜本頁產生時間：%s</p>' % (esc(_fmt_time(view["generated_ms"])), esc(_fmt_time(now_ms))))
    if not preview:
        out.append('<script>%s</script>' % PAGE_SCRIPT)
    out.append('</body></html>\n')
    return "\n".join(out)


# ───────────────────────────── 通知決策 ─────────────────────────────

def decide_notification(state, red, now_ms):
    """回傳 (要不要通知, 新狀態)。進入紅色通知一次；持續紅色每 6 小時最多再提醒一次；回到非紅色後重置。"""
    st = dict(state or {})
    if not red:
        st["red_since_ms"] = None
        st["last_notified_ms"] = None
        return False, st
    last = st.get("last_notified_ms")
    if not _is_int(last):
        st["red_since_ms"] = now_ms
        st["last_notified_ms"] = now_ms
        return True, st
    if now_ms - last >= REMIND_SECONDS * 1000:
        st["last_notified_ms"] = now_ms
        return True, st
    return False, st


def count_as_red(view, fail_since_ms, now_ms):
    """通知用的「紅色」：整體燈號為紅；但從未成功拉過資料時，要連續失敗超過 15 分鐘才算（避免剛設定好就誤報）。"""
    if view["overall"] != "red":
        return False
    if view["source"] == "none":
        return fail_since_ms is not None and now_ms - fail_since_ms > FAIL_RED_SECONDS * 1000
    return True


def notification_text(view):
    """通知文字只用固定詞彙與項目編號，不放任何來自 GX10 的文字。"""
    reasons = []
    for kind, text in view["banners"]:
        if kind == "red":
            reasons.append(text.split("（")[0])
    if view["red_items"]:
        reasons.append("紅燈項目：第 %s 項" % "、".join(str(i) for i in view["red_items"]))
    return "GX10 狀態：紅燈", "；".join(reasons) if reasons else "整體燈號為紅"


def applescript_quote(text):
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def osascript_argv(title, message):
    script = 'display notification "%s" with title "%s"' % (applescript_quote(message), applescript_quote(title))
    return ["/usr/bin/osascript", "-e", script]


# ───────────────────────────── 拉取（可注入）─────────────────────────────

def ssh_argv(host, key, ssh=DEFAULT_SSH):
    """固定的拉取指令：只用指定金鑰、不用 ssh-agent 以外的任何旗標（沒有轉發、沒有終端機參數），指令固定是 status。"""
    return [ssh, "-i", key, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, "status"]


def default_runner(argv, timeout):
    try:
        r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, shell=False)
    except (OSError, subprocess.TimeoutExpired):
        return 255, "", ""
    return r.returncode, r.stdout, r.stderr


def default_notifier(title, message):
    default_runner(osascript_argv(title, message), 10)


def pull_status(runner, host, key, timeout=PULL_TIMEOUT, ssh=DEFAULT_SSH):
    """回傳 (成功?, 文字)。結束碼非 0、輸出為空或過大都算失敗。"""
    rc, out, _ = runner(ssh_argv(host, key, ssh), timeout)
    if rc != 0 or not isinstance(out, str) or not out.strip() or len(out) > 1_000_000:
        return False, ""
    return True, out


# ───────────────────────────── 檔案 ─────────────────────────────

def ensure_dir(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def write_atomic(path, text):
    """先寫暫存檔（同目錄、權限 600）再改名。"""
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


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def load_last_good(path):
    v = _load_json(path)
    if v is None or not isinstance(v.get("status"), dict) or validate_status(v["status"]):
        return None            # 損毀或格式不對的舊資料不採用
    if not _is_int(v.get("fetched_ms")):
        return None
    return v


# ───────────────────────────── 執行一次 ─────────────────────────────

def run_once(host, key, out_dir, runner, notifier, now_ms, ssh=DEFAULT_SSH):
    ensure_dir(out_dir)
    state = _load_json(os.path.join(out_dir, "state.json")) or {}
    last_good = load_last_good(os.path.join(out_dir, "last_good.json"))
    ok, text = pull_status(runner, host, key, ssh=ssh)
    current, format_bad = None, False
    if ok:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if obj is not None and not validate_status(obj):
            current = obj
        else:
            format_bad = True
    pull_failed = current is None
    if pull_failed:
        if not _is_int(state.get("fail_since_ms")):
            state["fail_since_ms"] = now_ms
    else:
        state["fail_since_ms"] = None
        write_atomic(os.path.join(out_dir, "last_good.json"), json.dumps({"status": current, "fetched_ms": now_ms}, ensure_ascii=False) + "\n")
    view = make_view(current, last_good, now_ms, pull_failed=pull_failed, format_bad=format_bad, fail_since_ms=state.get("fail_since_ms"))
    write_atomic(os.path.join(out_dir, "status.html"), render_page(view, now_ms))
    red = count_as_red(view, state.get("fail_since_ms"), now_ms)
    notify, state = decide_notification(state, red, now_ms)
    if notify:
        title, msg = notification_text(view)
        notifier(title, msg)
    write_atomic(os.path.join(out_dir, "state.json"), json.dumps(state, ensure_ascii=False) + "\n")
    result = "format" if format_bad else ("fail" if pull_failed else "ok")
    write_atomic(os.path.join(out_dir, "last_run.txt"), "%s pull=%s overall=%s notified=%d\n" % (_fmt_time(now_ms), result, view["overall"], 1 if notify else 0))
    return view, notify


# ───────────────────────────── 預覽（只用假資料）─────────────────────────────

def fake_status(now_ms, overrides=None, generated_ms=None):
    """假的 status.json（所有內容都是範例，不連線任何機器）。"""
    colors = {1: "green", 2: "green", 3: "info", 4: "green", 5: "green", 6: "green", 7: "green", 8: "info",
              9: "green", 10: "info", 11: "green", 12: "unknown", 13: "green", 14: "green"}
    texts = {1: "容器都 running，重啟次數不變", 2: "四項健康檢查都是 200", 3: "進行中 0、排隊 0", 4: "點數 31（基準 31）、狀態 green",
             5: "溫度 45 度", 6: "磁碟 20%、可用記憶體 25 GB", 7: "距上次完成 10 小時", 8: "保留 2 份、約 12 MB", 9: "距上次拉取 6 小時",
             10: "固定標籤都在", 11: "服務 active 且已啟用", 12: ITEM12_TEXT, 13: "24 小時失敗 0 次、Error Logger 觸發 0 次", 14: "還原驗證通過（3 天前）"}
    for k, (c, t) in (overrides or {}).items():
        colors[k], texts[k] = c, t
    worst = "green"
    for iid, c in colors.items():
        if c == "red":
            worst = "red"
        elif c == "yellow" and worst != "red":
            worst = "yellow"
    return {"schema": 1, "generated_at_ms": now_ms if generated_ms is None else generated_ms, "overall": worst, "complete": True,
            "note": NOTE, "items": [{"id": i, "name": ITEM_NAMES[i], "color": colors[i], "text": texts[i]} for i in range(1, 15)]}


def preview_scenarios(now_ms):
    """回傳 [(檔名, 說明, 檢視模型)]。"""
    green = fake_status(now_ms)
    yellow = fake_status(now_ms, {6: ("yellow", "磁碟 83%、可用記憶體 25 GB"), 13: ("yellow", "24 小時失敗 1 次、Error Logger 觸發 0 次；waiting 增加")})
    red = fake_status(now_ms, {7: ("red", "距上次完成 40 小時"), 2: ("red", "n8n 無回應／逾時")})
    old = fake_status(now_ms, generated_ms=now_ms - 20 * 60 * 1000)
    return [
        ("1-all-green.html", "全綠", make_view(green, None, now_ms)),
        ("2-yellow.html", "含黃（磁碟、n8n 統計）", make_view(yellow, None, now_ms)),
        ("3-red.html", "含紅（備份、健康檢查）", make_view(red, None, now_ms)),
        ("4-pull-failed-stale.html", "拉取失敗，只剩 20 分鐘前的資料（過期）",
         make_view(None, {"status": old, "fetched_ms": now_ms - 20 * 60 * 1000}, now_ms, pull_failed=True, fail_since_ms=now_ms - 20 * 60 * 1000)),
        ("5-no-data.html", "從未成功拉取（尚無資料）", make_view(None, None, now_ms, pull_failed=True, fail_since_ms=now_ms)),
        ("6-bad-format.html", "資料格式異常", make_view(None, {"status": green, "fetched_ms": now_ms - 60000}, now_ms, pull_failed=True, format_bad=True)),
    ]


def write_preview(directory, now_ms):
    ensure_dir(directory)
    names = []
    for fname, desc, view in preview_scenarios(now_ms):
        write_atomic(os.path.join(directory, fname), render_page(view, now_ms, preview=True))
        names.append((fname, desc))
    index = ['<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8"><title>GX10 狀態預覽（假資料）</title></head><body>',
             '<h1>預覽（假資料）</h1><p>下列頁面全部是假資料，只用來檢視版面。</p><ul>']
    for fname, desc in names:
        index.append('<li><a href="%s">%s</a>：%s</li>' % (html.escape(fname), html.escape(fname), html.escape(desc)))
    index.append("</ul></body></html>\n")
    write_atomic(os.path.join(directory, "index.html"), "\n".join(index))
    return [n for n, _ in names]


# ───────────────────────────── 進入點 ─────────────────────────────

def main(argv=None, env=None, runner=None, notifier=None, now_ms=None):
    ap = argparse.ArgumentParser(description="GX10 控制台（Mac 端）")
    ap.add_argument("--preview-dir", help="只用假資料產生預覽頁到這個資料夾（不連線、不通知）")
    args = ap.parse_args(argv)
    env = os.environ if env is None else env
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if args.preview_dir:
        write_preview(os.path.expanduser(args.preview_dir), now)
        return 0
    host = (env.get("GX10_CONSOLE_HOST") or "").strip()
    if not host:
        sys.stderr.write("請設定 GX10_CONSOLE_HOST，例如 user@host（沒有預設值）\n")
        return 2
    if not _HOST_RE.match(host):
        sys.stderr.write("GX10_CONSOLE_HOST 格式不正確（只允許字母、數字與 . _ @ : -，且不能以 - 開頭）\n")
        return 2
    ssh = (env.get("GX10_CONSOLE_SSH") or "").strip() or DEFAULT_SSH
    if not _SSH_RE.match(ssh) or ".." in ssh.split("/"):
        sys.stderr.write("GX10_CONSOLE_SSH 格式不正確（必須是絕對路徑，只允許字母、數字與 . _ / -，不能含 ..）\n")
        return 2
    home = os.path.expanduser("~")
    key = env.get("GX10_CONSOLE_KEY") or os.path.join(home, ".ssh", "gx10_console")
    out_dir = env.get("GX10_CONSOLE_DIR") or os.path.join(home, "GX10Console")
    run_once(host, key, out_dir, runner or default_runner, notifier or default_notifier, now, ssh=ssh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
