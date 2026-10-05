#!/usr/bin/env python3
"""Mutation harness for tests/test_console_status.py 與 tests/test_console_forced_command.py。

在暫存目錄的副本上改 scripts/console_status.py 或 scripts/console_forced_command.py，真正的檔案完全不動。
每一種突變都必須讓對應的測試檔失敗（「抓到」）；沒抓到的會列出來，不隱藏。
Usage: python3 tests/mutate_console_status.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ST, FC = "console_status.py", "console_forced_command.py"


REPL = []        # 所有突變的替換文字，供開跑前的靜態自我檢查掃描


def sub(old, new, count=1):
    REPL.append(new)

    def f(t):
        assert old in t, "mutation anchor missing: %r" % old
        return t.replace(old, new, count)
    return f


def multi(*fs):
    def f(t):
        for g in fs:
            t = g(t)
        return t
    return f


# (標籤, 目標檔, 突變函式)
M = [
    # ── 移除必要輸入／缺資料變成正常 ──
    ("評估時跳過第 7 項（備份）", ST, sub("    for it in ITEMS:\n        r = evaluate_item(it, data)", "    for it in [i for i in ITEMS if i[\"id\"] != 7]:\n        r = evaluate_item(it, data)")),
    ("整體燈號忽略無資料項目", ST, sub('''        if r["color"] == "nodata":
            complete = False
            worst = max(worst, _ORDER["yellow"])''', '''        if r["color"] == "nodata":
            pass''')),
    ("缺資料（黃）改成綠", ST, sub('return {"color": "nodata", "text": "無資料" + extra}', 'return {"color": "green", "text": "無資料" + extra}')),
    ("缺資料但整體不降為黃", ST, sub('''    if overall == "green" and not complete:
        overall = "yellow"
''', "")),
    ("缺資料不算黃（worst 不更新）", ST, sub('''            complete = False
            worst = max(worst, _ORDER["yellow"])
        elif r["color"] in _ORDER:''', '''            complete = False
        elif r["color"] in _ORDER:''')),
    ("備份缺資料改成黃", ST, sub('return {"color": "red", "text": "無資料（備份無法確認，視為紅）"}', 'return {"color": "yellow", "text": "無資料"}')),
    ("備份格式錯誤改成黃", ST, sub('return {"color": "red", "text": "資料格式不正確（備份無法確認，視為紅）"}', 'return {"color": "yellow", "text": "資料格式不正確"}')),
    ("整體燈號只看前 6 項", ST, sub('''    for r in rows:
        if r["id"] == 12:
            continue
        if r["color"] == "nodata":''', '''    for r in rows[:6]:
        if r["id"] == 12:
            continue
        if r["color"] == "nodata":''')),
    ("整體燈號取最寬鬆而非最嚴", ST, sub("worst = max(worst, _ORDER[r[\"color\"]])", "worst = min(worst, _ORDER[r[\"color\"]])")),
    ("第 12 項計入整體（取消略過）", ST, sub('''        if r["id"] == 12:
            continue
        if r["color"] == "nodata":''', '''        if r["color"] == "nodata":''')),
    ("第 12 項改成判綠", ST, sub('return {"color": "unknown", "text": "未確認（需管理權限）"}', 'return {"color": "green", "text": "規則已載入"}')),
    ("第 12 項改成判紅", ST, sub('return {"color": "unknown", "text": "未確認（需管理權限）"}', 'return {"color": "red", "text": "規則未載入"}')),
    ("資訊項目 10 不見了改成 info", ST, sub('{"color": "yellow", "text": "有固定標籤不見了"}', '{"color": "info", "text": "有固定標籤不見了"}')),
    # ── 輸入檢查放寬 ──
    ("數值接受負數", ST, sub("and math.isfinite(v) and v >= 0)", "and math.isfinite(v) and v >= -1000)")),
    ("數值接受布林", ST, sub("return (isinstance(v, (int, float)) and not isinstance(v, bool)\n            and math.isfinite(v)", "return (isinstance(v, (int, float))\n            and math.isfinite(v)")),
    ("數值接受 nan／inf", ST, sub("            and math.isfinite(v) and v >= 0)", "            and v >= 0)")),
    ("布林接受整數 0／1", ST, sub("def _is_bool(v):\n    return isinstance(v, bool)", "def _is_bool(v):\n    return isinstance(v, (bool, int))")),
    ("溫度等欄位接受 0", ST, sub("return _is_num(v) and v > 0", "return _is_num(v) and v >= 0")),
    ("健康檢查 ms 接受字串", ST, sub('if ms is not None and not _is_num(ms):\n                return bad()', 'pass')),
    ("健康檢查 code 接受字串", ST, sub('if code is not None and (isinstance(code, bool) or not isinstance(code, int)):\n                return bad()', 'pass')),
    ("非 dict 的項目值被當成資料", ST, sub("    if not isinstance(v, dict):\n        if iid == 7:", "    if v is None:\n        if iid == 7:")),
    # ── 過期判斷 ──
    ("過期門檻改成 15 小時", ST, sub("STALE_SECONDS = 15 * 60", "STALE_SECONDS = 15 * 3600")),
    ("過期邊界 > 改成 >=", ST, sub("if now_ms - generated_at_ms > STALE_SECONDS * 1000:", "if now_ms - generated_at_ms >= STALE_SECONDS * 1000:")),
    ("拿掉未來時間戳記檢查", ST, sub("    if generated_at_ms > now_ms + FUTURE_SKEW_SECONDS * 1000:\n        return True\n", "")),
    ("時間戳記型別不檢查（字串當正常）", ST, sub("    if isinstance(generated_at_ms, bool) or not isinstance(generated_at_ms, (int, float)):\n        return True\n", "")),
    ("過期時整體不變紅", ST, sub('''    if stale:
        overall = "red"
''', "")),
    ("輸出時間戳記沿用資料內的值", ST, sub('    clean["generated_at_ms"] = int(now_ms)\n', "")),
    # ── 門檻與邊界 ──
    ("GPU 黃紅互換", ST, sub('color = "red" if t > 92 else "yellow" if t >= 85 else "green"', 'color = "yellow" if t > 92 else "red" if t >= 85 else "green"')),
    ("GPU 邊界 85 改 > 85", ST, sub('"yellow" if t >= 85', '"yellow" if t > 85')),
    ("GPU 邊界 92 改 >= 92", ST, sub('"red" if t > 92', '"red" if t >= 92')),
    ("磁碟黃紅互換", ST, sub('''        if d > 90 or a < 3:
            return {"color": "red", "text": t}
        if d >= 80 or a <= 10:
            return {"color": "yellow", "text": t}''', '''        if d >= 80 or a <= 10:
            return {"color": "red", "text": t}
        if d > 90 or a < 3:
            return {"color": "yellow", "text": t}''')),
    ("磁碟 80 改 > 80", ST, sub("if d >= 80 or a <= 10:", "if d > 80 or a <= 10:")),
    ("磁碟 90 改 >= 90", ST, sub("if d > 90 or a < 3:", "if d >= 90 or a < 3:")),
    ("記憶體 10 改 < 10", ST, sub("if d >= 80 or a <= 10:", "if d >= 80 or a < 10:")),
    ("記憶體 3 改 <= 3", ST, sub("if d > 90 or a < 3:", "if d > 90 or a <= 3:")),
    ("備份 26 改 >= 26", ST, sub("        if h > 26:\n", "        if h >= 26:\n")),
    ("備份 36 改 >= 36", ST, sub("if f or h > 36:", "if f or h >= 36:")),
    ("備份失敗標記不判紅", ST, sub("if f or h > 36:", "if h > 36:")),
    ("Mac 拉取 24 改 >= 24", ST, sub('"yellow" if h > 24', '"yellow" if h >= 24')),
    ("Mac 拉取 36 改 >= 36", ST, sub('"red" if h > 36 else "yellow" if h > 24', '"red" if h >= 36 else "yellow" if h > 24')),
    ("Mac 拉取黃紅互換", ST, sub('"red" if h > 36 else "yellow" if h > 24 else "green"', '"yellow" if h > 36 else "red" if h > 24 else "green"')),
    ("健康檢查 5 秒改 > 5000", ST, sub("elif ms >= 5000 and worst", "elif ms > 5000 and worst")),
    ("健康檢查 15 秒改 >= 15000", ST, sub("or ms > 15000:", "or ms >= 15000:")),
    ("健康檢查逾時（ms 為 None）不判紅", ST, sub('if code != 200 or ms is None or ms > 15000:', 'if code != 200 or (ms is not None and ms > 15000):')),
    ("n8n 失敗 3 次門檻改 4", ST, sub('v["failures_24h"] >= 3:', 'v["failures_24h"] >= 4:')),
    ("n8n 失敗不判黃", ST, sub('if v["failures_24h"] >= 1 or v["logger_triggers"] >= 1 or v["waiting_increase"]:', 'if v["logger_triggers"] >= 1 or v["waiting_increase"]:')),
    ("n8n 無回應不判紅", ST, sub('if not v["responsive"] or v["logger_triggers"] >= 3', 'if v["logger_triggers"] >= 3')),
    ("還原驗證 14 天改 >= 14", ST, sub("        if d > 14:\n", "        if d >= 14:\n")),
    ("還原驗證失敗不判紅", ST, sub('        if not p:\n            return {"color": "red", "text": t}\n', "")),
    ("向量庫點數不同不判黃", ST, sub('        if v["points"] != v["baseline"]:\n            return {"color": "yellow", "text": t + "；點數與基準不同"}\n', "")),
    ("向量庫狀態不是 green 不判紅", ST, sub('or v["status"] != "green":', ":")),
    ("容器 OOM 不判紅", ST, sub('if not v["all_running"] or v["oom"]:', 'if not v["all_running"]:')),
    ("容器重啟增加不判黃", ST, sub('        if v["restart_increase"]:\n            return {"color": "yellow", "text": "重啟次數增加　" + names}\n', "")),
    ("防火牆失敗旗標不判紅", ST, sub('if not v["active"] or v["fail_flag"]:', 'if not v["active"]:')),
    # ── 蒐集函式：例外、逾時、快取 ──
    ("蒐集函式例外不再被攔下", ST, sub('        except Exception:\n            box["v"] = None', '        except Exception:\n            raise')),
    ("蒐集逾時不生效（等到結束）", ST, sub("    t.join(timeout)", "    t.join()")),
    ("重啟基準缺少時當作沒增加", ST, sub("        increase = None  # 沒有基準又有重啟紀錄：無法判斷，視為無資料", "        increase = False")),
    ("n8n 快取改成永不使用", ST, sub("now - cache.get(\"at\", 0) < N8N_CACHE_SECONDS", "False")),
    ("n8n 快取改成永遠使用", ST, sub("now - cache.get(\"at\", 0) < N8N_CACHE_SECONDS", "True")),
    ("n8n 查詢改成可寫入模式", ST, sub("{readOnly:true}", "{readOnly:false}")),
    ("n8n 查詢逾時改成 1000 秒", ST, sub("N8N_TIMEOUT = 10", "N8N_TIMEOUT = 1000")),
    ("n8n responsive 一律當作 True", ST, sub('responsive = any(c["name"] == "n8n" and c["code"] == 200 for c in health)', "responsive = True")),
    ("n8n waiting 無基準時不當作增加", ST, sub("increase = waiting > waiting_baseline if isinstance(waiting_baseline, int) else waiting > 0", "increase = waiting > waiting_baseline if isinstance(waiting_baseline, int) else False")),
    ("備份時間讀不到時當作 0 小時", ST, sub('    done = env.mtime("%s/%s/DONE" % (st, latest))', '    try:\n        done = env.mtime("%s/%s/DONE" % (st, latest))\n    except OSError:\n        done = env.now()')),
    ("LATEST 格式不檢查", ST, sub('    if not re.match(r"^[0-9]{8}-[0-9]{6}$", latest):\n        raise ValueError("LATEST 格式不合法")\n', "")),
    # ── 固定說明 ──
    ("固定說明文字被改掉", ST, sub('NOTE = "整體燈號不含防火牆規則檢查（需管理權限）"', 'NOTE = "整體燈號"')),
    ("評估結果不帶固定說明", ST, sub('"overall": overall, "complete": complete, "note": NOTE}', '"overall": overall, "complete": complete, "note": ""}')),
    ("status.json 不帶固定說明", ST, sub('        "note": NOTE,\n        "items"', '        "items"')),
    ("每日摘要不帶固定說明", ST, sub("                              if attention else \"；沒有需要留意的項目\"),\n             NOTE]", "                              if attention else \"；沒有需要留意的項目\")]")),
    # ── 輸出過濾 ──
    ("輸出不經白名單過濾", ST, sub("    clean = sanitize(data)\n    clean[\"generated_at_ms\"]", "    clean = dict(data)\n    clean[\"generated_at_ms\"]")),
    ("名稱白名單允許空白與任意字元", ST, sub('_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,60}$")', '_NAME_RE = re.compile(r"^.{1,60}$")')),
    ("狀態字串不限制列舉", ST, sub('return (True, v) if v in _STATUS_ENUM else (False, None)', 'return (True, v)')),
    ("多餘欄位不丟棄（containers 附帶文字）", ST, sub('        if key == "containers" and isinstance(v.get("names"), list):', '        clean.update({k: x for k, x in v.items() if k not in fields})\n        if key == "containers" and isinstance(v.get("names"), list):')),
    # ── 寫檔與 shell ──
    ("原子寫入改成直接寫目標檔", ST, sub("        os.replace(tmp, path)", "        open(path, 'w').write(text)")),
    ("輸出檔權限放寬成 644", ST, sub("os.chmod(tmp, 0o600)", "os.chmod(tmp, 0o644)")),
    ("失敗時不清掉暫存檔", ST, sub("        try:\n            os.unlink(tmp)\n        except OSError:\n            pass\n        raise", "        raise")),
    ("每日摘要保留 14 份改成 15 份", ST, sub("SUMMARY_KEEP = 14", "SUMMARY_KEEP = 15")),
    ("清理誤刪不符合格式的檔案", ST, sub("        m = _SUMMARY_RE.match(name)\n        if not m:\n            continue", "        m = _SUMMARY_RE.match(name)\n        if not m:\n            os.unlink(os.path.join(out_dir, name))\n            continue")),
    ("程式裡寫死 IP 位址", ST, sub('"vllm_metrics_url": "http://127.0.0.1:8000/metrics"', '"vllm_metrics_url": "http://10.9.8.7:8000/metrics"')),
    ("程式裡寫死金鑰", ST, sub('SCHEMA_VERSION = 1', 'SCHEMA_VERSION = 1\nTOKEN = "abcd1234efgh"')),
    # ── forced command ──
    ("強制指令：比對前去掉空白", FC, multi(sub("if original_command in FILES:", "if original_command.strip() in FILES:"), sub("FILES[original_command]", "FILES[original_command.strip()]"))),
    ("強制指令：比對前轉小寫", FC, multi(sub("if original_command in FILES:", "if original_command.lower() in FILES:"), sub("FILES[original_command]", "FILES[original_command.lower()]"))),
    ("強制指令：只比對開頭（startswith）", FC, multi(sub("if original_command in FILES:", "if any(original_command.startswith(k) for k in FILES):"), sub("FILES[original_command]", "FILES[[k for k in FILES if original_command.startswith(k)][0]]"))),
    ("強制指令：只取第一個字（split）", FC, multi(sub("if original_command in FILES:", "if original_command.split(' ')[0] in FILES:"), sub("FILES[original_command]", "FILES[original_command.split(' ')[0]]"))),
    ("強制指令：換行前的部分當指令", FC, multi(sub("if original_command in FILES:", "if original_command.splitlines()[0:1] and original_command.splitlines()[0] in FILES:"), sub("FILES[original_command]", "FILES[original_command.splitlines()[0]]"))),
    # 下面三個突變「不執行任何東西」：只把收到的輸入（或固定字串）寫進暫存複本輸出目錄的標記檔，
    # 讓「不修改資料檔」之類的既有測試能偵測到多出來的檔案。
    ("強制指令：拒絕時照輸入內容處理（模擬：只寫標記檔）", FC, sub('    err.write(b"not allowed\\n")', '    open(os.path.join(out_dir, "INPUT_PROCESSED.marker"), "a").write(repr(original_command)[:80])\n    err.write(b"not allowed\\n")')),
    ("強制指令：拒絕時把輸入轉交其他程序（模擬：只寫標記檔）", FC, sub('    err.write(b"not allowed\\n")', '    open(os.path.join(out_dir, "INPUT_FORWARDED.marker"), "a").write(repr(original_command)[:80])\n    err.write(b"not allowed\\n")')),
    ("強制指令：拒絕時呼叫外部程式（模擬：只寫標記檔）", FC, sub('    err.write(b"not allowed\\n")', '    open(os.path.join(out_dir, "EXTERNAL_CALL.marker"), "a").write("called")\n    err.write(b"not allowed\\n")')),
    ("強制指令：多開放 state 檔", FC, sub('"summary": "daily-summary.txt"}', '"summary": "daily-summary.txt", "state": "state.json"}')),
    ("強制指令：status 與 summary 對調", FC, sub('FILES = {"status": "status.json", "summary": "daily-summary.txt"}', 'FILES = {"status": "daily-summary.txt", "summary": "status.json"}')),
    ("強制指令：不限制讀取大小", FC, sub("body = f.read(MAX_BYTES)", "body = f.read()")),
    ("強制指令：log 記完整輸入", FC, sub("    return ascii(text[:limit])", "    return ascii(text)")),
    ("強制指令：log 不跳脫輸入", FC, sub("    return ascii(text[:limit])", "    return text[:limit]")),
    ("強制指令：log 記檔案內容", FC, sub("        out.write(body)\n        _log(original_command, log_path)", "        out.write(body)\n        _log(original_command + ' ' + body.decode('utf-8', 'replace'), log_path)")),
    ("強制指令：拒絕時結束碼改 0", FC, sub('    _log("拒絕 %s" % _safe(original_command), log_path)\n    return 2', '    _log("拒絕 %s" % _safe(original_command), log_path)\n    return 0')),
    ("強制指令：檔案不存在時結束碼改 0", FC, sub("            return 3\n        out.write(body)", "            return 0\n        out.write(body)")),
    ("強制指令：拒絕時不寫 log", FC, sub('    _log("拒絕 %s" % _safe(original_command), log_path)\n', "")),
    ("強制指令：非字串輸入不處理", FC, sub('    if not isinstance(original_command, str):\n        original_command = ""\n', "")),
    ("強制指令：log 寫失敗就丟例外", FC, sub("    except OSError:\n        pass\n\n\ndef serve", "    except OSError:\n        raise\n\n\ndef serve")),
    ("強制指令：混入真實 IP", FC, sub("MAX_BYTES = 1_000_000", "MAX_BYTES = 1_000_000\nHOST = '203.0.113.9'")),
]

TEST_FOR = {ST: "test_console_status.py", FC: "test_console_forced_command.py"}

# 會把輸入交給外殼或執行任意程式的寫法：任何突變的替換文字都不得含有，沒有例外
FORBIDDEN = re.compile(r"(?i)os\.system|\.system\s*\(|subprocess|popen|\beval\b|\bexec|shell|sh\s+-c")


def check_replacements():
    """執行任何突變之前的靜態自我檢查：掃描所有突變的替換文字，含危險字樣就中止，不執行任何測試。"""
    bad = [t[:60] for t in REPL if FORBIDDEN.search(t)]
    if bad:
        print("靜態自我檢查失敗：下列突變的替換文字含有會執行輸入的寫法，已中止，沒有執行任何測試：")
        for b in bad:
            print("  -", repr(b))
        sys.exit(2)
    print("靜態自我檢查通過：%d 個突變的替換文字都沒有執行輸入的寫法" % len(REPL))


def main():
    check_replacements()
    if "--check-only" in sys.argv:
        return
    ok, survivors = 0, []
    for label, fname, fn in M:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "scripts").mkdir()
            (td / "tests").mkdir()
            for f in (ST, FC):
                shutil.copy(ROOT / "scripts" / f, td / "scripts" / f)
            for t in TEST_FOR.values():
                shutil.copy(ROOT / "tests" / t, td / "tests" / t)
            target = td / "scripts" / fname
            target.write_text(fn(target.read_text(encoding="utf-8")), encoding="utf-8")
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                str(td / "tests" / TEST_FOR[fname])], capture_output=True, text=True, timeout=300)
            killed = r.returncode != 0
            ok += killed
            if not killed:
                survivors.append(label)
            print("%s  %s" % ("抓到" if killed else "沒抓到", label), flush=True)
    print("\n抓到 %d／總共 %d" % (ok, len(M)))
    if survivors:
        print("沒抓到：")
        for s in survivors:
            print("  -", s)
    sys.exit(0 if not survivors else 1)


if __name__ == "__main__":
    main()
