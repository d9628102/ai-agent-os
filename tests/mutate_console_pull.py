#!/usr/bin/env python3
"""Mutation harness for tests/test_console_pull.py（scripts/gx10-console-mac/console_pull.py 的測試）。

在暫存目錄的副本上改 console_pull.py，真正的檔案完全不動。每一種突變都必須讓測試失敗（「抓到」）；沒抓到的會列出來，不隱藏。
突變只做「文字替換」，替換文字裡不得有會執行輸入的寫法（見 FORBIDDEN）；開跑前會先做靜態自我檢查，含這類字樣就中止，不執行任何測試。
Usage:
  python3 tests/mutate_console_pull.py               # 跑全部突變
  python3 tests/mutate_console_pull.py --check-only  # 只做靜態自我檢查，並確認每個突變的替換位置都存在（不執行測試）
  python3 tests/mutate_console_pull.py --self-test   # 證明防呆有效（防呆見 tests/mutation_guard.py）
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_guard as guard
from mutation_guard import CAUGHT, SURVIVED, INVALID, RUNTIME
SRC_REL = Path("scripts") / "gx10-console-mac" / "console_pull.py"
TEST_REL = Path("tests") / "test_console_pull.py"
REPL = []        # 所有突變的替換文字，供開跑前的靜態自我檢查掃描


def sub(old, new, count=1):
    REPL.append(new)

    def f(t):
        assert old in t, "mutation anchor missing: %r" % old[:80]
        return t.replace(old, new, count)
    return f


M = [
    # ── 格式驗證放寬 ──
    ("驗證放寬：未知項目燈號放行", sub('        if it.get("color") not in ITEM_COLORS:\n            problems.append("item %d color" % iid)', '        pass')),
    ("驗證放寬：未知整體燈號放行", sub('    if obj.get("overall") not in OVERALL_COLORS:\n        problems.append("overall")', '    pass')),
    ("驗證放寬：時間欄位型別不檢查", sub('    if not _is_int(g) or g <= 0:', '    if g is None:')),
    ("驗證放寬：布林當整數", sub("    return isinstance(v, int) and not isinstance(v, bool)", "    return isinstance(v, int)")),
    ("驗證放寬：項目數量不檢查", sub('    if not isinstance(items, list) or len(items) != 14:', '    if not isinstance(items, list):')),
    ("驗證放寬：項目編號重複放行", sub('if not _is_int(iid) or iid < 1 or iid > 14 or iid in seen:', 'if not _is_int(iid) or iid < 1 or iid > 14:')),
    ("驗證放寬：第 12 項不檢查", sub('        if iid == 12 and it.get("color") != "unknown":\n            problems.append("item 12 必須是未確認")', '        pass')),
    ("驗證放寬：schema 不檢查", sub('    if not _is_int(obj.get("schema")) or obj.get("schema") != SCHEMA_VERSION:', '    if False:')),
    ("驗證放寬：缺 complete 放行", sub('    if not isinstance(obj.get("complete"), bool):\n        problems.append("complete")', '    pass')),
    ("驗證放寬：項目 text 型別不檢查", sub('        if not isinstance(it.get("text"), str) or len(it["text"]) > MAX_TEXT * 3:\n            problems.append("item %d text" % iid)', '        pass')),
    ("驗證放寬：項目 name 型別不檢查", sub('        if not isinstance(it.get("name"), str) or len(it["name"]) > MAX_NAME:\n            problems.append("item %d name" % iid)', '        pass')),
    ("驗證永遠回報沒有問題", sub('    return problems\n\n\ndef scrub', '    return []\n\n\ndef scrub')),
    # ── 過期 ──
    ("過期邊界 > 改 >=", sub("if now_ms - generated_at_ms > STALE_SECONDS * 1000:", "if now_ms - generated_at_ms >= STALE_SECONDS * 1000:")),
    ("拿掉未來時間檢查", sub("    if generated_at_ms > now_ms + FUTURE_SKEW_SECONDS * 1000:\n        return True\n", "")),
    ("未來時間邊界 > 改 >=", sub("if generated_at_ms > now_ms + FUTURE_SKEW_SECONDS * 1000:", "if generated_at_ms >= now_ms + FUTURE_SKEW_SECONDS * 1000:")),
    ("過期門檻改成 150 分鐘", sub("STALE_SECONDS = 15 * 60", "STALE_SECONDS = 150 * 60")),
    ("過期判斷不檢查時間型別", sub("    if not _is_int(generated_at_ms):\n        return True\n", "")),
    ("過期時整體不變紅", sub('            banners.append(("red", "狀態資料過期（超過 15 分鐘沒有更新，或時間不可信）"))\n            force_red = True', '            banners.append(("red", "狀態資料過期（超過 15 分鐘沒有更新，或時間不可信）"))')),
    # ── 檢視模型 ──
    ("拉取失敗仍顯示綠", sub('        if pull_failed and overall == "green":\n            overall = "yellow"', '        pass')),
    ("資料格式異常不變紅", sub('    if format_bad:\n        banners.append(("red", "資料格式異常"))\n        force_red = True', '    if format_bad:\n        banners.append(("red", "資料格式異常"))')),
    ("尚無資料不變紅", sub('        banners.append(("red", "尚無資料"))\n        force_red = True', '        banners.append(("red", "尚無資料"))')),
    ("連續失敗超過 15 分鐘不變紅", sub('        banners.append(("red", "拉取連續失敗超過 15 分鐘"))\n        force_red = True', '        banners.append(("red", "拉取連續失敗超過 15 分鐘"))')),
    ("連續失敗邊界 > 改 >=", sub("if pull_failed and fail_since_ms is not None and now_ms - fail_since_ms > FAIL_RED_SECONDS * 1000:", "if pull_failed and fail_since_ms is not None and now_ms - fail_since_ms >= FAIL_RED_SECONDS * 1000:")),
    ("Mac 端重算燈號（有紅燈項目就降級）", sub('        overall = base["overall"]\n', '        overall = base["overall"]\n        if any(r["color"] == "red" for r in base["items"]) and overall == "green":\n            overall = "yellow"\n')),
    ("Mac 端重算燈號（取項目最壞值）", sub('        overall = base["overall"]\n', '        overall = "red" if any(r["color"] == "red" for r in base["items"]) else base["overall"]\n')),
    ("第 12 項採用來源值", sub('        if iid == 12:\n            rows.append({"id": 12, "name": ITEM_NAMES[12], "color": "unknown", "text": ITEM12_TEXT})\n            continue\n', '')),
    ("第 12 項文字被改掉", sub('ITEM12_TEXT = "未確認（需管理權限）"', 'ITEM12_TEXT = "已確認"')),
    ("項目燈號全部顯示綠", sub('"color": src["color"], "text": scrub(src["text"])})', '"color": "green", "text": scrub(src["text"])})')),
    ("缺項目顯示綠", sub('rows.append({"id": iid, "name": ITEM_NAMES[iid], "color": "nodata", "text": "無資料"})', 'rows.append({"id": iid, "name": ITEM_NAMES[iid], "color": "green", "text": "無資料"})')),
    ("顯示前不過濾文字", sub('"name": scrub(src["name"]) or ITEM_NAMES[iid], "color": src["color"], "text": scrub(src["text"])', '"name": src["name"], "color": src["color"], "text": src["text"]')),
    ("過濾不遮蔽網址", sub('    re.compile(r"https?://\\S+"),\n', '')),
    # ── 通知 ──
    ("通知重複（持續紅色每次都通知）", sub('    if now_ms - last >= REMIND_SECONDS * 1000:\n        st["last_notified_ms"] = now_ms\n        return True, st\n    return False, st', '    st["last_notified_ms"] = now_ms\n    return True, st')),
    ("通知不重置（回復後狀態留著）", sub('    if not red:\n        st["red_since_ms"] = None\n        st["last_notified_ms"] = None\n        return False, st', '    if not red:\n        return False, st')),
    ("提醒間隔改 1 小時", sub("REMIND_SECONDS = 6 * 3600", "REMIND_SECONDS = 1 * 3600")),
    ("提醒邊界 >= 改 >", sub("    if now_ms - last >= REMIND_SECONDS * 1000:", "    if now_ms - last > REMIND_SECONDS * 1000:")),
    ("提醒永遠不再發", sub("    if now_ms - last >= REMIND_SECONDS * 1000:", "    if now_ms - last >= REMIND_SECONDS * 1000000000:")),
    ("黃燈也算紅色通知", sub('    if view["overall"] != "red":\n        return False\n', '    if view["overall"] == "green":\n        return False\n')),
    ("從未成功立刻算紅色通知", sub('        return fail_since_ms is not None and now_ms - fail_since_ms > FAIL_RED_SECONDS * 1000', '        return True')),
    ("從未成功的 15 分鐘邊界 > 改 >=", sub('        return fail_since_ms is not None and now_ms - fail_since_ms > FAIL_RED_SECONDS * 1000', '        return fail_since_ms is not None and now_ms - fail_since_ms >= FAIL_RED_SECONDS * 1000')),
    ("通知文字放入 GX10 的文字", sub('        reasons.append("紅燈項目：第 %s 項" % "、".join(str(i) for i in view["red_items"]))', '        reasons.append("紅燈：" + "、".join(r["text"] for r in view["rows"] if r["id"] in view["red_items"]))')),
    ("通知字串不跳脫", sub("def applescript_quote(text):\n    return ", "def applescript_quote(text):\n    return text\n    return ")),
    ("通知改用預設函式（不走注入）", sub("        notifier(title, msg)", "        default_notifier(title, msg)")),
    ("通知狀態不存檔", sub('    write_atomic(os.path.join(out_dir, "state.json"), json.dumps(state, ensure_ascii=False) + "\\n")', '    pass')),
    ("失敗開始時間不記錄", sub('        if not _is_int(state.get("fail_since_ms")):\n            state["fail_since_ms"] = now_ms', '        pass')),
    ("失敗開始時間每次覆寫", sub('        if not _is_int(state.get("fail_since_ms")):', '        if True:')),
    ("成功後不清失敗時間", sub('        state["fail_since_ms"] = None\n        write_atomic', '        write_atomic')),
    # ── 頁面 ──
    ("頁面不跳脫 HTML", sub("    esc = html.escape\n", "    esc = lambda s, quote=True: str(s)\n")),
    ("頁面移除固定說明", sub("    out.append('<p class=\"note\">%s</p>' % esc(NOTE))\n", "")),
    ("固定說明被改掉", sub('NOTE = "整體燈號不含防火牆規則檢查（需管理權限）"', 'NOTE = "整體燈號"')),
    ("內嵌小程式改用 innerHTML", sub("s.textContent='頁面超過", "s.innerHTML='頁面超過")),
    ("內嵌小程式過期門檻放寬", sub("d<=900000", "d<=9000000")),
    ("內嵌小程式未來門檻放寬", sub("d>=-60000", "d>=-6000000")),
    ("移除內容安全政策", sub("    out.append('<meta http-equiv=\"Content-Security-Policy\" content=\"default-src \\'none\\'; style-src \\'unsafe-inline\\'; script-src \\'unsafe-inline\\'\">')\n", "")),
    ("頁面加入外部樣式表", sub("<title>GX10 狀態</title>", "<title>GX10 狀態</title><link rel=\"stylesheet\" href=\"x.css\">")),
    ("資料被放進內嵌小程式", sub("out.append('<script>%s</script>' % PAGE_SCRIPT)", "out.append('<script>%s</script>' % (PAGE_SCRIPT + 'var d=' + json.dumps(view['rows'][0]['text']) + ';'))")),
    ("頁面移除產生時間屬性", sub('<body data-rendered-ms="%d">', '<body data-x="%d">')),
    ("預覽頁不標示假資料", sub("        out.append('<div class=\"banner preview\">預覽（假資料）：這一頁不是真實狀態，只用來檢視版面。</div>')", "        pass")),
    ("預覽模式仍要求主機位址", sub("        write_preview(os.path.expanduser(args.preview_dir), now)\n        return 0", "        write_preview(os.path.expanduser(args.preview_dir), now)")),
    # ── 設定與拉取 ──
    ("主機位址加預設值", sub('host = (env.get("GX10_CONSOLE_HOST") or "").strip()', 'host = (env.get("GX10_CONSOLE_HOST") or "user@host").strip()')),
    ("主機位址不檢查格式", sub("    if not _HOST_RE.match(host):", "    if False:")),
    ("主機位址允許以 - 開頭", sub('r"^[A-Za-z0-9][A-Za-z0-9._@:-]*$"', 'r"^[A-Za-z0-9._@:-]*$"')),
    ("金鑰預設改用備份金鑰", sub('os.path.join(home, ".ssh", "gx10_console")', 'os.path.join(home, ".ssh", "gx10_backup")')),
    ("輸出資料夾預設改名", sub('os.path.join(home, "GX10Console")', 'os.path.join(home, "GX10Out")')),
    ("拉取指令多帶轉發旗標", sub('"-o", "ConnectTimeout=10", host, "status"]', '"-o", "ConnectTimeout=10", "-A", host, "status"]')),
    ("拉取指令改成其他動作", sub('"ConnectTimeout=10", host, "status"]', '"ConnectTimeout=10", host, "summary"]')),
    ("拉取：結束碼非 0 仍算成功", sub("    if rc != 0 or not isinstance(out, str)", "    if not isinstance(out, str)")),
    ("拉取：空輸出算成功", sub(" or not out.strip()", "")),
    ("拉取改用預設函式（不走注入）", sub("    ok, text = pull_status(runner, host, key, ssh=ssh)", "    ok, text = pull_status(default_runner, host, key, ssh=ssh)")),
    ("上一份資料不驗證", sub('    if v is None or not isinstance(v.get("status"), dict) or validate_status(v["status"]):', '    if v is None or not isinstance(v.get("status"), dict):')),
    # ── 檔案 ──
    ("輸出檔權限放寬成 644", sub("os.chmod(tmp, 0o600)", "os.chmod(tmp, 0o644)")),
    ("輸出資料夾權限放寬成 755", sub("os.makedirs(path, mode=0o700, exist_ok=True)\n    os.chmod(path, 0o700)", "os.makedirs(path, mode=0o755, exist_ok=True)\n    os.chmod(path, 0o755)")),
    ("原子寫入改成直接寫目標檔", sub("        os.replace(tmp, path)", "        open(path, 'w').write(text)")),
    ("失敗時不清掉暫存檔", sub("        try:\n            os.unlink(tmp)\n        except OSError:\n            pass\n        raise", "        raise")),
    ("程式裡寫死 IP 位址", sub('PULL_TIMEOUT = 30', 'PULL_TIMEOUT = 30\nDEFAULT_HOST = "203.0.113.9"')),
    ("程式裡寫死金鑰", sub('PULL_TIMEOUT = 30', 'PULL_TIMEOUT = 30\nTOKEN = "abcd1234efgh"')),
    # ── ssh 的絕對路徑 ──
    ("拉取改回依賴 PATH 找 ssh", sub('DEFAULT_SSH = "/usr/bin/ssh"', 'DEFAULT_SSH = "ssh"')),
    ("拉取路徑被換成其他位置", sub('DEFAULT_SSH = "/usr/bin/ssh"', 'DEFAULT_SSH = "/opt/x/ssh"')),
    ("run_once 的預設 ssh 改成依賴 PATH", sub("now_ms, ssh=DEFAULT_SSH):", 'now_ms, ssh="ssh"):')),
    ("pull_status 的預設 ssh 改成依賴 PATH", sub("timeout=PULL_TIMEOUT, ssh=DEFAULT_SSH):", 'timeout=PULL_TIMEOUT, ssh="ssh"):')),
    ("ssh_argv 的預設 ssh 改成依賴 PATH", sub("def ssh_argv(host, key, ssh=DEFAULT_SSH):", 'def ssh_argv(host, key, ssh="ssh"):')),
    ("ssh 覆蓋值不檢查格式", sub('    if not _SSH_RE.match(ssh) or ".." in ssh.split("/"):', '    if False:')),
    ("ssh 覆蓋值允許相對路徑", sub('_SSH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")', '_SSH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")')),
    ("ssh 覆蓋值允許 .. 路徑", sub(' or ".." in ssh.split("/"):', ':')),
    ("ssh 覆蓋值允許空白與特殊字元", sub('_SSH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")', '_SSH_RE = re.compile(r"^/.+$")')),
    ("ssh 覆蓋值沒有傳進拉取指令", sub('pull_status(runner, host, key, ssh=ssh)', 'pull_status(runner, host, key)')),
    ("ssh 覆蓋環境變數被忽略", sub('ssh = (env.get("GX10_CONSOLE_SSH") or "").strip() or DEFAULT_SSH', 'ssh = DEFAULT_SSH')),
    ("ssh 覆蓋值沒有傳進 run_once", sub("notifier or default_notifier, now, ssh=ssh)", "notifier or default_notifier, now)")),
]

# 非程式檔的突變（範例檔與說明）
PLIST = "com.psf.gx10-console.plist.example"
MF = [
    ("plist 的 PATH 被放寬", PLIST, sub("/usr/bin:/bin", "/usr/bin:/bin:/Users/YOUR_USER/bin")),
    ("plist 拿掉 PATH 設定", PLIST, sub("    <key>PATH</key><string>/usr/bin:/bin</string>\n", "")),
    ("plist 程式路徑改成依賴 PATH", PLIST, sub("<string>/usr/bin/python3</string>", "<string>python3</string>")),
    ("plist 改成每 30 秒執行", PLIST, sub("<integer>300</integer>", "<integer>30</integer>")),
    ("plist 混入真實位址樣式", PLIST, sub("user@host", "tester@192.168.1.10")),
    ("README 拿掉 ssh 路徑覆蓋說明", "README.md", sub("GX10_CONSOLE_SSH", "GX10_CONSOLE_X", 99)),
    ("README 拿掉 PATH 說明", "README.md", sub("PATH", "路徑", 99)),
    ("README 混入內網位址", "README.md", sub("# Mac 端 GX10 狀態頁", "# Mac 端 GX10 狀態頁 192.168.1.10")),
]

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


SRC_NAME = SRC_REL.name
PKG = SRC_REL.parent


def all_rels():
    return guard.list_files(ROOT, PKG) + [TEST_REL]


def baseline(runner=None):
    """未突變的複本必須全部通過。"""
    return guard.run_baseline(ROOT, all_rels(), [TEST_REL], runner)


def evaluate_mutation(fname, fn, drop=None, runner=None):
    return guard.evaluate(ROOT, all_rels(), TEST_REL, PKG / fname, fn, drop=drop, runner=runner)


def all_mutations():
    return [(label, SRC_NAME, fn) for label, fn in M] + MF


def self_check():
    """證明兩道保護有效（不執行任何突變程式碼）。回傳問題清單。"""
    return guard.self_test() + guard.check_harness(ROOT, all_rels(), TEST_REL, SRC_REL, PKG / "README.md")


def main():
    check_replacements()
    if "--self-test" in sys.argv:
        problems = self_check()
        print("保護自我檢查：%s" % ("全部正常" if not problems else "有問題"))
        for q in problems:
            print("  -", q)
        sys.exit(0 if not problems else 5)
    if "--check-only" in sys.argv:
        src = (ROOT / SRC_REL).read_text(encoding="utf-8")
        for label, fname, fn in all_mutations():          # 只在記憶體裡套用，確認替換位置都存在，不執行任何東西
            fn((ROOT / PKG / fname).read_text(encoding="utf-8"))
        print("每個突變的替換位置都存在：%d 個" % len(all_mutations()))
        return
    ok, detail = baseline()
    if not ok:
        print("中止：" + detail)
        sys.exit(3)
    print("未突變的複本全部通過，開始跑突變")
    results = []
    for label, fname, fn in all_mutations():
        cat, detail = evaluate_mutation(fname, fn)
        results.append((label, cat, detail))
        print("%s  %s" % ({CAUGHT: "抓到", RUNTIME: "執行期例外", SURVIVED: "沒抓到", INVALID: "無效結果"}[cat], label), flush=True)
    sys.exit(guard.summarize(results))


if __name__ == "__main__":
    main()
