"""
測試重點（控制台 Mac 端，scripts/gx10-console-mac/console_pull.py）：
- 燈號只來自 GX10 的 status.json：Mac 端不重算（整體燈號沿用 GX10 的值），只做格式驗證、過期、拉取失敗、尚無資料
- 格式驗證：缺欄位、未知燈號值、錯誤型別、布林冒充數字、重複或缺少的項目，都不得顯示為綠（視為紅並標示「資料格式異常」）
- 過期：剛好 15 分鐘不算、多 1 毫秒算；比現在晚超過 60 秒算（時鐘不可信）；拉取失敗時整體燈號至少黃、連續失敗超過 15 分鐘為紅
- 通知狀態機：進入紅色通知一次、持續紅色 6 小時內不重複、滿 6 小時再提醒、回復後重置、連續拉取失敗超過 15 分鐘算紅
- 固定說明一定存在；第 12 項固定「未確認」
- 頁面純文字：用 HTML 解析器驗證，特殊字元與標籤只會變成文字；沒有外部資源；內嵌小程式固定且不含資料、不用 innerHTML
- 必須設定主機位址（沒有預設值、不能以 - 開頭）；拉取指令固定；輸出權限 700／600；原子寫入
- 輸出不含問答原文、金鑰樣式、網址；通知文字不含任何來自 GX10 的文字
- 所有外部呼叫（ssh、osascript）都走注入的假函式；預設的真實函式在測試中一律被禁止
"""
import ast
import copy
import json
import os
import re
import sys
from html.parser import HTMLParser

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "gx10-console-mac"))

import pytest

import console_pull as P

NOW = 1_791_190_000_000          # 毫秒
MIN = 60_000
CANARY = "CANARY_QA_TEXT_31de"
PAGE_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "gx10-console-mac", "console_pull.py")


@pytest.fixture(autouse=True)
def forbid_real_calls(monkeypatch):
    """預設的真實函式（ssh、osascript）在測試中一律禁止；測試只能用注入的假函式。"""
    def boom(*a, **k):
        raise AssertionError("測試中不得呼叫真實的 ssh／osascript")
    monkeypatch.setattr(P, "default_runner", boom)
    monkeypatch.setattr(P, "default_notifier", boom)


def good(now=NOW, **kw):
    return P.fake_status(now, **kw)


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags, self.attrs, self.text, self.scripts = [], [], [], []
        self._in_script = False
        self.overall_text, self.overall_class = None, None
        self._cap = None

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.append((tag, dict(attrs)))
        if tag == "script":
            self._in_script = True
            self.scripts.append("")
        d = dict(attrs)
        if d.get("id") == "overall":
            self._cap = "overall"
            self.overall_class = d.get("class")
            self.overall_text = ""

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        if tag == "span" and self._cap == "overall":
            self._cap = None

    def handle_data(self, data):
        if self._in_script:
            self.scripts[-1] += data
        else:
            self.text.append(data)
            if self._cap == "overall":
                self.overall_text += data


def parse(html_text):
    p = Page()
    p.feed(html_text)
    p.close()
    return p


def page_of(view, now=NOW, preview=False):
    return parse(P.render_page(view, now, preview=preview))


class FakeRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append((argv, timeout))
        rc, out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return rc, out, ""


class FakeNotifier:
    def __init__(self):
        self.calls = []

    def __call__(self, title, message):
        self.calls.append((title, message))


def ok_out(status):
    return (0, json.dumps(status, ensure_ascii=False))


HOST = "user@host"
KEY = "/keys/k"


def run(tmp_path, runner, notifier, now):
    return P.run_once(HOST, KEY, str(tmp_path / "out"), runner, notifier, now)


# ── 格式驗證 ──
def test_good_status_is_valid():
    assert P.validate_status(good()) == []


def _mut(fn):
    d = good()
    fn(d)
    return d


BAD = {
    "不是物件": [], "None": None, "字串": "x", "空物件": {},
    "缺 schema": _mut(lambda d: d.pop("schema")), "schema=2": _mut(lambda d: d.update(schema=2)),
    "schema=布林": _mut(lambda d: d.update(schema=True)),
    "缺時間": _mut(lambda d: d.pop("generated_at_ms")), "時間字串": _mut(lambda d: d.update(generated_at_ms="1791190000000")),
    "時間浮點": _mut(lambda d: d.update(generated_at_ms=1.5e12)), "時間 0": _mut(lambda d: d.update(generated_at_ms=0)),
    "時間負數": _mut(lambda d: d.update(generated_at_ms=-5)), "時間布林": _mut(lambda d: d.update(generated_at_ms=True)),
    "缺 overall": _mut(lambda d: d.pop("overall")), "overall 未知": _mut(lambda d: d.update(overall="purple")),
    "overall 空字串": _mut(lambda d: d.update(overall="")), "overall 數字": _mut(lambda d: d.update(overall=0)),
    "overall None": _mut(lambda d: d.update(overall=None)), "overall 大寫": _mut(lambda d: d.update(overall="GREEN")),
    "缺 complete": _mut(lambda d: d.pop("complete")), "complete 數字": _mut(lambda d: d.update(complete=1)),
    "缺 items": _mut(lambda d: d.pop("items")), "items 不是列表": _mut(lambda d: d.update(items={})),
    "items 少一個": _mut(lambda d: d["items"].pop()), "items 多一個": _mut(lambda d: d["items"].append(dict(d["items"][0]))),
    "item 不是物件": _mut(lambda d: d["items"].__setitem__(0, "x")),
    "item id 重複": _mut(lambda d: d["items"][1].update(id=1)), "item id 超出": _mut(lambda d: d["items"][0].update(id=15)),
    "item id 字串": _mut(lambda d: d["items"][0].update(id="1")), "item id 布林": _mut(lambda d: d["items"][0].update(id=True)),
    "item 缺 name": _mut(lambda d: d["items"][0].pop("name")), "item name 數字": _mut(lambda d: d["items"][0].update(name=5)),
    "item name 過長": _mut(lambda d: d["items"][0].update(name="x" * 101)),
    "item 缺 color": _mut(lambda d: d["items"][0].pop("color")), "item 未知燈號": _mut(lambda d: d["items"][0].update(color="purple")),
    "item 燈號 None": _mut(lambda d: d["items"][0].update(color=None)), "item 燈號大寫": _mut(lambda d: d["items"][0].update(color="GREEN")),
    "item 缺 text": _mut(lambda d: d["items"][0].pop("text")), "item text 數字": _mut(lambda d: d["items"][0].update(text=0)),
    "item text 過長": _mut(lambda d: d["items"][0].update(text="x" * 1000)),
    "第 12 項判綠": _mut(lambda d: d["items"][11].update(color="green")),
    "第 12 項判紅": _mut(lambda d: d["items"][11].update(color="red")),
}


@pytest.mark.parametrize("name", list(BAD))
def test_validation_rejects_malformed_status(name):
    assert P.validate_status(BAD[name]) != []


def test_validation_messages_do_not_echo_values():
    d = good()
    d["items"][0]["color"] = CANARY
    assert CANARY not in " ".join(P.validate_status(d))


def test_unknown_color_never_displays_as_green(tmp_path):
    bad = good()
    bad["items"][3]["color"] = "purple"
    n = FakeNotifier()
    view, _ = run(tmp_path, FakeRunner([ok_out(bad)]), n, NOW)
    assert view["overall"] == "red" and ("red", "資料格式異常") in view["banners"]
    pg = parse((tmp_path / "out" / "status.html").read_text(encoding="utf-8"))
    assert pg.overall_text == "紅" and "c-red" in pg.overall_class and "資料格式異常" in "".join(pg.text)


def test_bad_json_and_extra_garbage_are_format_errors(tmp_path):
    for out in ("not json", "{", "[]", "null", "{\"x\": 1}", ""):
        view, _ = run(tmp_path, FakeRunner([(0, out)]), FakeNotifier(), NOW)
        assert view["overall"] == "red"


# ── 過期 ──
def test_stale_boundaries():
    assert not P.is_stale(NOW, NOW)
    assert not P.is_stale(NOW - 15 * MIN, NOW)
    assert P.is_stale(NOW - 15 * MIN - 1, NOW)
    assert not P.is_stale(NOW + 60_000, NOW)
    assert P.is_stale(NOW + 60_001, NOW)
    for bad in (None, "x", True, 1.5e12, [], float("nan")):
        assert P.is_stale(bad, NOW)


def test_stale_fresh_pull_is_red_with_banner():
    v = P.make_view(good(NOW - 15 * MIN - 1), None, NOW)
    assert v["overall"] == "red" and v["stale"]
    assert any("過期" in t for k, t in v["banners"])
    v2 = P.make_view(good(NOW - 15 * MIN), None, NOW)
    assert v2["overall"] == "green" and not v2["stale"]


def test_future_timestamp_is_red():
    assert P.make_view(good(NOW + 61_000), None, NOW)["overall"] == "red"
    assert P.make_view(good(NOW + 60_000), None, NOW)["overall"] == "green"


# ── 檢視模型：不重算燈號 ──
@pytest.mark.parametrize("overall", ["green", "yellow", "red"])
def test_overall_comes_from_gx10_not_recomputed(overall):
    st = good()
    st["overall"] = overall
    assert P.make_view(st, None, NOW)["overall"] == overall


def test_mac_does_not_recompute_from_items():
    st = good()
    st["overall"] = "green"
    st["items"][6]["color"] = "red"          # GX10 說整體是綠、項目 7 是紅：Mac 端照 GX10 的整體燈號顯示，不自己重算
    v = P.make_view(st, None, NOW)
    assert v["overall"] == "green"
    assert next(r for r in v["rows"] if r["id"] == 7)["color"] == "red" and v["red_items"] == [7]


def test_item_colors_are_copied_not_changed():
    st = P.fake_status(NOW, {6: ("yellow", "x"), 13: ("red", "y")})
    v = P.make_view(st, None, NOW)
    got = {r["id"]: r["color"] for r in v["rows"]}
    want = {i["id"]: i["color"] for i in st["items"]}
    want[12] = "unknown"
    assert got == want


def test_item12_always_unconfirmed_and_not_counted():
    st = good()
    st["items"][11]["color"], st["items"][11]["text"] = "green", "規則已載入"     # 就算有人塞進來也不採用
    v = P.make_view(st, None, NOW)
    r12 = next(r for r in v["rows"] if r["id"] == 12)
    assert r12["color"] == "unknown" and r12["text"] == "未確認（需管理權限）"
    assert v["overall"] == "green" and 12 not in v["red_items"]
    st["items"][11]["color"] = "red"
    assert P.make_view(st, None, NOW)["red_items"] == []


# ── 拉取失敗與尚無資料 ──
@pytest.mark.parametrize("last,expect", [("green", "yellow"), ("yellow", "yellow"), ("red", "red")])
def test_pull_failure_keeps_last_data_but_never_green(last, expect):
    st = good(NOW - 5 * MIN)
    st["overall"] = last
    v = P.make_view(None, {"status": st, "fetched_ms": NOW - 5 * MIN}, NOW, pull_failed=True, fail_since_ms=NOW - MIN)
    assert v["overall"] == expect and v["source"] == "kept"
    assert any(k == "yellow" and "拉取失敗" in t for k, t in v["banners"]) or expect == "red"
    assert any("拉取失敗" in t for k, t in v["banners"])


def test_pull_failure_with_old_data_is_red_and_both_banners():
    old = good(NOW - 20 * MIN)
    v = P.make_view(None, {"status": old, "fetched_ms": NOW - 20 * MIN}, NOW, pull_failed=True, fail_since_ms=NOW - 20 * MIN)
    assert v["overall"] == "red"
    texts = " ".join(t for _, t in v["banners"])
    assert "拉取失敗" in texts and "過期" in texts


def test_consecutive_failure_over_15_minutes_is_red_even_with_fresh_looking_data():
    st = good(NOW)
    v = P.make_view(None, {"status": st, "fetched_ms": NOW}, NOW, pull_failed=True, fail_since_ms=NOW - 15 * MIN - 1)
    assert v["overall"] == "red" and any("連續失敗" in t for _, t in v["banners"])
    v2 = P.make_view(None, {"status": st, "fetched_ms": NOW}, NOW, pull_failed=True, fail_since_ms=NOW - 15 * MIN)
    assert v2["overall"] == "yellow"


def test_never_succeeded_shows_no_data_red():
    v = P.make_view(None, None, NOW, pull_failed=True, fail_since_ms=NOW)
    assert v["overall"] == "red" and v["source"] == "none"
    assert ("red", "尚無資料") in v["banners"]
    assert all(r["color"] == "nodata" for r in v["rows"] if r["id"] != 12)
    assert next(r for r in v["rows"] if r["id"] == 12)["color"] == "unknown"


def test_format_bad_is_always_red_even_with_green_last_data():
    v = P.make_view(None, {"status": good(), "fetched_ms": NOW}, NOW, pull_failed=True, format_bad=True)
    assert v["overall"] == "red" and ("red", "資料格式異常") in v["banners"]


# ── 通知狀態機 ──
H = 3600_000


def test_notification_state_machine():
    st = {}
    n, st = P.decide_notification(st, False, NOW)
    assert n is False
    n, st = P.decide_notification(st, True, NOW)
    assert n is True and st["last_notified_ms"] == NOW                       # 進入紅色
    n, st = P.decide_notification(st, True, NOW + 5 * MIN)
    assert n is False                                                       # 持續紅色，不重複
    n, st = P.decide_notification(st, True, NOW + 6 * H - 1)
    assert n is False
    n, st = P.decide_notification(st, True, NOW + 6 * H)
    assert n is True and st["last_notified_ms"] == NOW + 6 * H              # 滿 6 小時再提醒一次
    n, st = P.decide_notification(st, True, NOW + 6 * H + MIN)
    assert n is False
    n, st = P.decide_notification(st, False, NOW + 7 * H)
    assert n is False and st["last_notified_ms"] is None and st["red_since_ms"] is None      # 回復後重置
    n, st = P.decide_notification(st, True, NOW + 7 * H + MIN)
    assert n is True                                                        # 再次進入紅色，再通知


def test_notification_clock_going_backwards_does_not_notify():
    _, st = P.decide_notification({}, True, NOW)
    n, _ = P.decide_notification(st, True, NOW - H)
    assert n is False


def test_notification_state_keeps_other_fields():
    n, st = P.decide_notification({"fail_since_ms": 123}, True, NOW)
    assert st["fail_since_ms"] == 123
    n, st = P.decide_notification(st, False, NOW + 1)
    assert st["fail_since_ms"] == 123


def test_yellow_never_notifies():
    for c in ("green", "yellow"):
        st = good()
        st["overall"] = c
        v = P.make_view(st, None, NOW)
        assert P.count_as_red(v, None, NOW) is False


def test_count_as_red_for_never_succeeded_waits_15_minutes():
    v = P.make_view(None, None, NOW, pull_failed=True, fail_since_ms=NOW)
    assert P.count_as_red(v, NOW, NOW) is False
    assert P.count_as_red(v, NOW - 15 * MIN, NOW) is False
    assert P.count_as_red(v, NOW - 15 * MIN - 1, NOW) is True
    assert P.count_as_red(v, None, NOW) is False


def test_end_to_end_notifications_over_time(tmp_path):
    n = FakeNotifier()
    red = good()
    red["items"][6].update(color="red", text="距上次完成 40 小時")
    red["overall"] = "red"
    gr = good()

    def at(t, status):
        st = copy.deepcopy(status)
        st["generated_at_ms"] = t
        return run(tmp_path, FakeRunner([ok_out(st)]), n, t)

    at(NOW, gr)
    assert n.calls == []
    at(NOW + 5 * MIN, red)
    assert len(n.calls) == 1                                  # 進入紅色
    at(NOW + 10 * MIN, red)
    at(NOW + 5 * H, red)
    assert len(n.calls) == 1                                  # 持續紅色 6 小時內不重複
    at(NOW + 5 * MIN + 6 * H, red)
    assert len(n.calls) == 2                                  # 6 小時後提醒
    at(NOW + 7 * H, gr)
    at(NOW + 7 * H + 5 * MIN, red)
    assert len(n.calls) == 3                                  # 回復後重置，再次紅色又通知
    assert "第 7 項" in n.calls[0][1]


def test_end_to_end_consecutive_pull_failures_notify_once_after_15_minutes(tmp_path):
    n = FakeNotifier()
    r = FakeRunner([(255, "")])
    for k in range(0, 15, 5):                       # 0、5、10 分鐘：從未成功，還沒滿 15 分鐘
        run(tmp_path, r, n, NOW + k * MIN)
    assert n.calls == []
    run(tmp_path, r, n, NOW + 15 * MIN)
    assert n.calls == []                            # 剛好 15 分鐘不算
    run(tmp_path, r, n, NOW + 15 * MIN + 1000)
    assert len(n.calls) == 1
    run(tmp_path, r, n, NOW + 20 * MIN)
    assert len(n.calls) == 1                        # 持續失敗不重複
    # 成功一次後恢復，失敗時間重置
    run(tmp_path, FakeRunner([ok_out(good(NOW + 25 * MIN))]), n, NOW + 25 * MIN)
    st = json.loads((tmp_path / "out" / "state.json").read_text())
    assert st["fail_since_ms"] is None and st["last_notified_ms"] is None


def test_pull_failure_with_good_last_data_goes_red_after_it_goes_stale(tmp_path):
    n = FakeNotifier()
    run(tmp_path, FakeRunner([ok_out(good(NOW))]), n, NOW)
    fail = FakeRunner([(255, "")])
    v, _ = run(tmp_path, fail, n, NOW + 5 * MIN)
    assert v["overall"] == "yellow" and n.calls == []
    v, _ = run(tmp_path, fail, n, NOW + 10 * MIN)
    assert v["overall"] == "yellow"
    v, _ = run(tmp_path, fail, n, NOW + 16 * MIN)
    assert v["overall"] == "red" and len(n.calls) == 1


# ── 通知文字 ──
def test_notification_text_has_no_gx10_text():
    st = good()
    st["items"][6].update(color="red", text=CANARY)
    st["items"][6]["name"] = CANARY
    st["overall"] = "red"
    title, msg = P.notification_text(P.make_view(st, None, NOW))
    assert CANARY not in title + msg and "第 7 項" in msg


def test_osascript_argv_is_a_list_and_escapes():
    argv = P.osascript_argv('標題"x', 'a"b\\c\nd')
    assert isinstance(argv, list) and argv[0] == "/usr/bin/osascript" and argv[1] == "-e"
    assert '\\"' in argv[2] and "\n" not in argv[2] and "a\\\"b\\\\c d" in argv[2]
    assert P.applescript_quote('"\\') == '\\"\\\\'


# ── 頁面 ──
ALLOWED_TAGS = {"html", "head", "meta", "title", "style", "body", "h1", "div", "p", "span", "table", "thead", "tbody", "tr", "th", "td", "script"}


@pytest.mark.parametrize("which", ["green", "red", "none", "failed"])
def test_fixed_note_always_present_once(which):
    if which == "none":
        v = P.make_view(None, None, NOW, pull_failed=True, fail_since_ms=NOW)
    elif which == "failed":
        v = P.make_view(None, {"status": good(NOW - 30 * MIN), "fetched_ms": NOW}, NOW, pull_failed=True)
    elif which == "red":
        st = good()
        st["overall"] = "red"
        v = P.make_view(st, None, NOW)
    else:
        v = P.make_view(good(), None, NOW)
    pg = page_of(v)
    assert sum(1 for t in pg.text if t.strip() == "整體燈號不含防火牆規則檢查（需管理權限）") == 1


def test_page_item12_row_is_fixed_unconfirmed():
    pg = page_of(P.make_view(good(), None, NOW))
    joined = "\n".join(pg.text)
    assert "防火牆規則是否實際載入" in joined and "未確認（需管理權限）" in joined


def test_page_overall_badge_matches_view():
    for overall, label in (("green", "綠"), ("yellow", "黃"), ("red", "紅")):
        st = good()
        st["overall"] = overall
        pg = page_of(P.make_view(st, None, NOW))
        assert pg.overall_text == label


def test_special_characters_become_text_not_markup():
    evil = "<script>alert(1)</script><img src=x onerror=alert(1)>&amp;\"'<a href=//x>x</a>"
    st = good()
    st["items"][0].update(name=evil, text=evil)
    st["items"][4]["text"] = "</td></tr></table><h1>HACK</h1>"
    pg = page_of(P.make_view(st, None, NOW))
    assert set(pg.tags) <= ALLOWED_TAGS, set(pg.tags) - ALLOWED_TAGS
    assert pg.tags.count("script") == 1 and pg.scripts[0] == P.PAGE_SCRIPT       # 唯一的小程式是固定的，資料不會進到裡面
    assert pg.tags.count("h1") == 1 and pg.tags.count("table") == 1
    for tag, attrs in pg.attrs:
        assert not any(k.startswith("on") or k in ("src", "href", "srcset", "action", "style") for k in attrs), (tag, attrs)
    assert "<script>alert(1)</script>" in "".join(pg.text)                          # 原字串以文字顯示
    assert "&amp;" in "".join(pg.text)


def test_no_external_resources_and_csp_present():
    pg = page_of(P.make_view(good(), None, NOW))
    assert not {"a", "link", "img", "iframe", "object", "embed", "form", "base", "video", "audio", "source"} & set(pg.tags)
    csp = [a for t, a in pg.attrs if t == "meta" and a.get("http-equiv") == "Content-Security-Policy"]
    assert csp and "default-src 'none'" in csp[0]["content"]
    for tag, attrs in pg.attrs:
        assert not any(k in ("src", "href") for k in attrs)
    html_text = P.render_page(P.make_view(good(), None, NOW), NOW)
    assert not re.search(r"https?://|//[a-z]", html_text)


def test_page_script_is_fixed_and_uses_text_only():
    s = P.PAGE_SCRIPT
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval", "Function(", "setTimeout", "fetch", "XMLHttp", "import(", "location"):
        assert bad not in s, bad
    assert "textContent" in s and "if(!(d<=900000&&d>=-60000))" in s      # 15 分鐘與未來 60 秒的門檻要和 Python 端一致
    assert P.STALE_SECONDS * 1000 == 900000 and P.FUTURE_SKEW_SECONDS * 1000 == 60000
    html_text = P.render_page(P.make_view(good(), None, NOW), NOW)
    assert html_text.count(s) == 1


def test_page_rendered_ms_attribute_is_an_integer_only():
    pg = page_of(P.make_view(good(), None, NOW), now=NOW)
    body = next(a for t, a in pg.attrs if t == "body")
    assert body["data-rendered-ms"] == str(NOW)


def test_escape_is_really_applied():
    st = good()
    st["items"][0]["text"] = "<b>x</b>"
    raw = P.render_page(P.make_view(st, None, NOW), NOW)
    assert "<b>x</b>" not in raw and "&lt;b&gt;x&lt;/b&gt;" in raw


# ── 設定 ──
def test_host_is_required_no_default(tmp_path, capsys):
    r = FakeRunner([(0, "x")])
    for env in ({}, {"GX10_CONSOLE_HOST": ""}, {"GX10_CONSOLE_HOST": "   "}):
        out = tmp_path / "o"
        rc = P.main([], env=dict(env, GX10_CONSOLE_DIR=str(out)), runner=r, notifier=FakeNotifier(), now_ms=NOW)
        assert rc == 2 and r.calls == [] and not out.exists()
    assert "GX10_CONSOLE_HOST" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "-p22", "a b", "a;b", "a$(x)", "a\nb", "user@host -v", "`x`", "a|b", "a'b"])
def test_invalid_host_is_rejected(host, tmp_path):
    r = FakeRunner([(0, "x")])
    rc = P.main([], env={"GX10_CONSOLE_HOST": host, "GX10_CONSOLE_DIR": str(tmp_path / "o")}, runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert rc == 2 and r.calls == []


def test_main_uses_defaults_and_injected_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    r = FakeRunner([ok_out(good())])
    rc = P.main([], env={"GX10_CONSOLE_HOST": "user@host"}, runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert rc == 0
    argv, timeout = r.calls[0]
    assert argv[0] == "/usr/bin/ssh"
    assert argv[argv.index("-i") + 1] == str(tmp_path / ".ssh" / "gx10_console")
    assert (tmp_path / "GX10Console" / "status.html").exists()


def test_main_env_overrides(tmp_path):
    r = FakeRunner([ok_out(good())])
    P.main([], env={"GX10_CONSOLE_HOST": "h", "GX10_CONSOLE_KEY": "/k/x", "GX10_CONSOLE_DIR": str(tmp_path / "d")}, runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert r.calls[0][0][2] == "/k/x" and (tmp_path / "d" / "status.html").exists()


def test_ssh_command_is_fixed():
    assert P.ssh_argv("user@host", "/k") == ["/usr/bin/ssh", "-i", "/k", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                                              "-o", "ConnectTimeout=10", "user@host", "status"]
    argv = P.ssh_argv("h", "/k")
    assert not {"-A", "-L", "-R", "-D", "-t", "-tt", "-T", "-X", "-o ForwardAgent=yes"} & set(argv)
    assert argv[-1] == "status" and argv[-2] == "h"


@pytest.mark.parametrize("rc,out", [(255, ""), (1, "{}"), (0, ""), (0, "   \n"), (0, "x" * 1_000_001)])
def test_pull_failure_cases(rc, out):
    ok, text = P.pull_status(FakeRunner([(rc, out)]), "h", "/k")
    assert ok is False and text == ""


def test_pull_success_and_runner_receives_list_and_timeout():
    r = FakeRunner([(0, '{"a": 1}')])
    ok, text = P.pull_status(r, "h", "/k")
    assert ok and text == '{"a": 1}'
    argv, timeout = r.calls[0]
    assert isinstance(argv, list) and 0 < timeout <= 60 and argv[0] == "/usr/bin/ssh"      # 沒指定時用預設的絕對路徑


# ── 輸出檔 ──
def test_output_permissions_700_and_600(tmp_path):
    old = os.umask(0)
    try:
        run(tmp_path, FakeRunner([ok_out(good())]), FakeNotifier(), NOW)
    finally:
        os.umask(old)
    out = tmp_path / "out"
    assert (out.stat().st_mode & 0o777) == 0o700
    files = sorted(p.name for p in out.iterdir())
    assert files == ["last_good.json", "last_run.txt", "state.json", "status.html"]
    for p in out.iterdir():
        assert (p.stat().st_mode & 0o777) == 0o600, p.name


def test_existing_wide_directory_is_tightened(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    os.chmod(out, 0o755)
    run(tmp_path, FakeRunner([ok_out(good())]), FakeNotifier(), NOW)
    assert (out.stat().st_mode & 0o777) == 0o700


def test_write_atomic_replaces_and_failure_keeps_old(tmp_path, monkeypatch):
    p = tmp_path / "f.txt"
    P.write_atomic(str(p), "one")
    P.write_atomic(str(p), "two")
    assert p.read_text() == "two" and [x.name for x in tmp_path.iterdir()] == ["f.txt"]

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(P.os, "replace", boom)
    with pytest.raises(OSError):
        P.write_atomic(str(p), "three")
    assert p.read_text() == "two" and [x.name for x in tmp_path.iterdir()] == ["f.txt"]


def test_last_run_file_is_overwritten_not_appended(tmp_path):
    for k in range(3):
        run(tmp_path, FakeRunner([(255, "")]), FakeNotifier(), NOW + k * MIN)
    lines = (tmp_path / "out" / "last_run.txt").read_text().splitlines()
    assert len(lines) == 1 and "pull=fail" in lines[0]


def test_last_good_is_kept_across_runs_and_used_on_failure(tmp_path):
    n = FakeNotifier()
    run(tmp_path, FakeRunner([ok_out(good(NOW))]), n, NOW)
    v, _ = run(tmp_path, FakeRunner([(255, "")]), n, NOW + 5 * MIN)
    assert v["source"] == "kept" and v["generated_ms"] == NOW
    pg = parse((tmp_path / "out" / "status.html").read_text(encoding="utf-8"))
    assert "拉取失敗" in "".join(pg.text)


def test_corrupt_or_invalid_last_good_is_ignored(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for content in ("not json", "[]", json.dumps({"status": {"x": 1}, "fetched_ms": NOW}), json.dumps({"status": good(), "fetched_ms": "x"})):
        (out / "last_good.json").write_text(content)
        v, _ = run(tmp_path, FakeRunner([(255, "")]), FakeNotifier(), NOW)
        assert v["source"] == "none"


def test_invalid_pull_does_not_overwrite_last_good(tmp_path):
    n = FakeNotifier()
    run(tmp_path, FakeRunner([ok_out(good(NOW))]), n, NOW)
    before = (tmp_path / "out" / "last_good.json").read_text()
    bad = good(NOW + MIN)
    bad["items"][0]["color"] = "purple"
    run(tmp_path, FakeRunner([ok_out(bad)]), n, NOW + MIN)
    assert (tmp_path / "out" / "last_good.json").read_text() == before


# ── 輸出內容 ──
def test_output_has_no_qa_text_secrets_or_urls(tmp_path):
    st = good()
    st["data"] = {"q": CANARY}
    st["extra"] = CANARY
    st["items"][0]["text"] = ("sk-" + "A" * 24 + " AKIA" + "B" * 16 + " ghp_" + "C" * 30 + " https://example.invalid/x?signature=abc "
                              "Bearer abcdefghijklmnop -----BEGIN " + "PRIVATE" + " KEY----- 正常文字")
    run(tmp_path, FakeRunner([ok_out(st)]), FakeNotifier(), NOW)
    out = tmp_path / "out"
    blob = "".join(p.read_text(encoding="utf-8") for p in out.iterdir() if p.name in ("status.html", "state.json", "last_run.txt"))
    assert CANARY not in blob
    for pat in (r"sk-[A-Za-z0-9]{10,}", r"AKIA[0-9A-Z]{12,}", r"ghp_[A-Za-z0-9]{20,}", r"https?://", r"(?i)signature=", r"(?i)bearer\s+\S{10,}", r"BEGIN [A-Z ]*PRIVATE"):
        assert not re.search(pat, blob), pat
    assert "正常文字" in blob and "[已隱藏]" in blob


def test_scrub():
    assert P.scrub(5) == "" and P.scrub(None) == ""
    assert P.scrub("a\x00b\x1bc") == "a b c"
    assert len(P.scrub("x" * 1000)) == P.MAX_TEXT
    assert P.scrub("see https://x.invalid/a ok") == "see [已隱藏] ok"


# ── 預覽 ──
def test_preview_pages_are_marked_fake_and_have_fixed_note(tmp_path):
    names = P.write_preview(str(tmp_path / "pv"), NOW)
    assert len(names) >= 4
    for n in names:
        pg = parse((tmp_path / "pv" / n).read_text(encoding="utf-8"))
        txt = "".join(pg.text)
        assert "預覽（假資料）" in txt and "整體燈號不含防火牆規則檢查（需管理權限）" in txt
        assert "script" not in pg.tags
    modes = {(tmp_path / "pv" / n).stat().st_mode & 0o777 for n in names}
    assert modes == {0o600} and ((tmp_path / "pv").stat().st_mode & 0o777) == 0o700


def test_preview_scenarios_cover_green_yellow_red_and_failures():
    sc = {n: v for n, _, v in P.preview_scenarios(NOW)}
    assert sc["1-all-green.html"]["overall"] == "green"
    assert sc["2-yellow.html"]["overall"] == "yellow"
    assert sc["3-red.html"]["overall"] == "red"
    assert sc["4-pull-failed-stale.html"]["overall"] == "red" and sc["4-pull-failed-stale.html"]["source"] == "kept"
    assert sc["5-no-data.html"]["source"] == "none"
    assert ("red", "資料格式異常") in sc["6-bad-format.html"]["banners"]


def test_preview_mode_needs_no_host_and_calls_nothing(tmp_path):
    r = FakeRunner([(0, "x")])
    n = FakeNotifier()
    rc = P.main(["--preview-dir", str(tmp_path / "pv")], env={}, runner=r, notifier=n, now_ms=NOW)
    assert rc == 0 and r.calls == [] and n.calls == [] and (tmp_path / "pv" / "index.html").exists()


# ── 原始碼 ──
def test_source_has_no_dangerous_calls_or_shell_mode_and_no_real_addresses():
    src = open(PAGE_SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("system", "popen", "check_output", "getoutput", "call", "Popen"), node.attr
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec", "__import__")
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src)
    assert "psf01" not in src and "/home/" not in src and "/Users/" not in src
    assert not re.search(r"(?i)\b(password|passwd|secret|api_key|token)\s*=\s*[\"']", src)


def test_host_variable_has_no_default_in_source():
    src = open(PAGE_SRC, encoding="utf-8").read()
    assert 'env.get("GX10_CONSOLE_HOST") or ""' in src
    assert not re.search(r"GX10_CONSOLE_HOST[\"']\s*,\s*[\"'][^\"']+[\"']", src)


def test_default_runner_is_never_used_when_injected(tmp_path):
    # forbid_real_calls 已把預設函式換成會丟錯的函式；注入的假函式能正常跑完，代表流程沒有偷用預設值
    run(tmp_path, FakeRunner([ok_out(good())]), FakeNotifier(), NOW)


# ── 範例檔與說明 ──
DIR = os.path.dirname(PAGE_SRC)


def test_plist_example_is_valid_and_has_only_placeholders():
    import plistlib
    with open(os.path.join(DIR, "com.psf.gx10-console.plist.example"), "rb") as f:
        d = plistlib.load(f)
    assert d["StartInterval"] == 300 and d["Label"] == "com.psf.gx10-console"
    assert d["ProgramArguments"][:2] == ["/usr/bin/python3", "-B"]                       # 絕對路徑，不依賴 PATH
    assert d["EnvironmentVariables"]["PATH"] == "/usr/bin:/bin"
    assert d["EnvironmentVariables"]["GX10_CONSOLE_HOST"] == "user@host"
    assert "YOUR_USER" in " ".join(d["ProgramArguments"])
    raw = open(os.path.join(DIR, "com.psf.gx10-console.plist.example"), encoding="utf-8").read()
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", raw) and "psf01" not in raw and "/Users/" not in raw.replace("/Users/YOUR_USER", "")


def test_readme_documents_settings_and_limits_without_real_values():
    t = open(os.path.join(DIR, "README.md"), encoding="utf-8").read()
    for must in ("GX10_CONSOLE_HOST", "GX10_CONSOLE_KEY", "GX10_CONSOLE_DIR", "GX10_CONSOLE_SSH", "/usr/bin/ssh", "PATH", "/usr/bin:/bin", "沒有預設值", "from=", "沒有密碼短語", "通知權限", "回復", "整體燈號不含防火牆規則檢查（需管理權限）"):
        assert must in t, must
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", t) and "psf01" not in t


def test_directory_contains_only_expected_files():
    assert sorted(f for f in os.listdir(DIR) if not f.startswith("__")) == ["README.md", "com.psf.gx10-console.plist.example", "console_pull.py"]


# ── ssh 的絕對路徑 ──
def test_default_ssh_is_an_absolute_path_and_not_path_dependent():
    assert P.DEFAULT_SSH == "/usr/bin/ssh" and P.DEFAULT_SSH.startswith("/")
    assert P.ssh_argv("h", "/k")[0] == "/usr/bin/ssh"
    assert P.ssh_argv("h", "/k", "/opt/x/ssh")[0] == "/opt/x/ssh"


def test_run_once_and_pull_status_default_to_absolute_ssh(tmp_path):
    r = FakeRunner([(255, "")])
    run(tmp_path, r, FakeNotifier(), NOW)
    assert r.calls[0][0][0] == "/usr/bin/ssh"
    r2 = FakeRunner([(255, "")])
    P.pull_status(r2, "h", "/k")
    assert r2.calls[0][0][0] == "/usr/bin/ssh"


def test_ssh_override_is_used_when_valid(tmp_path):
    r = FakeRunner([ok_out(good())])
    rc = P.main([], env={"GX10_CONSOLE_HOST": "h", "GX10_CONSOLE_SSH": "/opt/homebrew/bin/ssh", "GX10_CONSOLE_DIR": str(tmp_path / "o")},
                runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert rc == 0 and r.calls[0][0][0] == "/opt/homebrew/bin/ssh"


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_ssh_override_means_default(tmp_path, empty):
    r = FakeRunner([ok_out(good())])
    P.main([], env={"GX10_CONSOLE_HOST": "h", "GX10_CONSOLE_SSH": empty, "GX10_CONSOLE_DIR": str(tmp_path / "o")}, runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert r.calls[0][0][0] == "/usr/bin/ssh"


@pytest.mark.parametrize("bad", ["ssh", "relative/ssh", "./ssh", "~/bin/ssh", "/usr/bin/../bin/ssh", "/usr/bin/..", "/usr/bin/ssh;id", "/usr/bin/ssh x",
                                 "/usr/bin/ssh\nx", "/usr/bin/s$(x)", "-/usr/bin/ssh", "-oProxyCommand=x", "/usr/bin/ssh|cat", "/usr/bin/ss`h`", "/usr/bin/ssh'x"])
def test_invalid_ssh_override_is_rejected(bad, tmp_path, capsys):
    r = FakeRunner([(0, "x")])
    out = tmp_path / "o"
    rc = P.main([], env={"GX10_CONSOLE_HOST": "h", "GX10_CONSOLE_SSH": bad, "GX10_CONSOLE_DIR": str(out)}, runner=r, notifier=FakeNotifier(), now_ms=NOW)
    assert rc == 2 and r.calls == [] and not out.exists()
    assert "GX10_CONSOLE_SSH" in capsys.readouterr().err
