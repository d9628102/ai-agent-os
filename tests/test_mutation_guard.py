"""
測試重點（三個變異檢查腳本共用的防呆，tests/mutation_guard.py 與 mutate_console_status.py、mutate_console_pull.py、mutate_gx10_backup_mac.py）：
- 結果分類：只有「測試斷言失敗」才算抓到；匯入失敗、找不到檔案、語法錯誤（收集錯誤）、非斷言例外、沒有收集到測試、設定錯誤、逾時，都是無效結果
- 複本少一個檔案、突變沒有改變內容：回報無效結果，而且不得執行測試
- 未突變的複本沒通過：中止（結束碼 3），不跑任何突變；有無效結果：整體失敗（結束碼 4）；有沒抓到：結束碼 1
- 防呆本身被破壞時，自我檢查會抓到（驗證這個檢查不是空的）
- 三個腳本都經由共用模組執行測試，不自己直接呼叫外部程式
本測試只用無害的小型假專案與假的執行函式，不執行任何突變程式碼。
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import mutation_guard as G
import mutate_console_pull as HP
import mutate_console_status as HS
import mutate_gx10_backup_mac as HB

HARNESSES = {"status": HS, "pull": HP, "backup": HB}


def test_classification_of_synthetic_projects():
    assert G.self_test() == []


@pytest.mark.parametrize("name", list(HARNESSES))
def test_each_harness_guards_work(name):
    assert HARNESSES[name].self_check() == []


@pytest.mark.parametrize("name", list(HARNESSES))
def test_each_harness_reports_invalid_for_incomplete_copy_without_running_tests(name):
    H = HARNESSES[name]
    rels = H.all_rels()
    boom = G._boom
    if name == "status":
        fname = HS.ST
    elif name == "pull":
        fname = HP.SRC_NAME
    else:
        fname = HB.SH
    victims = [r for r in rels if not r.name.startswith("test_") and r.name != fname]
    assert victims
    for victim in victims:
        cat, detail = H.evaluate_mutation(fname, lambda t: t + "\n", drop=victim, runner=boom)
        assert cat == G.INVALID and "複本不完整" in detail, (victim, cat, detail)
    cat, detail = H.evaluate_mutation(fname, lambda t: t, runner=boom)
    assert cat == G.INVALID and "沒有改變" in detail


@pytest.mark.parametrize("name", list(HARNESSES))
def test_baseline_failure_aborts_with_code_3_and_runs_no_mutation(name, monkeypatch, capsys):
    H = HARNESSES[name]
    monkeypatch.setattr(H, "baseline", lambda runner=None: (False, "未突變的複本沒有全部通過（測試）"))
    monkeypatch.setattr(H, "evaluate_mutation", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不應該跑突變")))
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:
        H.main()
    assert e.value.code == 3 and "中止" in capsys.readouterr().out


@pytest.mark.parametrize("name", list(HARNESSES))
@pytest.mark.parametrize("outcome,code", [(G.CAUGHT, 0), (G.RUNTIME, 0), (G.SURVIVED, 1), (G.INVALID, 4)])
def test_exit_codes_for_all_caught_survivor_and_invalid(name, outcome, code, monkeypatch):
    H = HARNESSES[name]
    monkeypatch.setattr(H, "baseline", lambda runner=None: (True, ""))
    monkeypatch.setattr(H, "evaluate_mutation", lambda *a, **k: (outcome, "x"))
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:
        H.main()
    assert e.value.code == code


def test_one_invalid_result_among_caught_makes_the_whole_run_fail(monkeypatch):
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return (G.INVALID, "複本不完整") if calls["n"] == 3 else (G.CAUGHT, "x")
    monkeypatch.setattr(HS, "baseline", lambda runner=None: (True, ""))
    monkeypatch.setattr(HS, "evaluate_mutation", fake)
    monkeypatch.setattr(sys, "argv", ["x"])
    with pytest.raises(SystemExit) as e:
        HS.main()
    assert e.value.code == 4


def test_summary_counts(capsys):
    code = G.summarize([("a", G.CAUGHT, ""), ("b", G.SURVIVED, ""), ("c", G.INVALID, "壞掉"), ("d", G.RUNTIME, "TypeError")])
    out = capsys.readouterr().out
    assert code == 4 and "總數 4；斷言抓到 1；執行期例外 1；沒抓到 1；無效結果 1" in out and "壞掉" in out
    assert "  - d ：" in out and "TypeError" in out            # 執行期例外逐個列出


# ── 新類別：執行期例外 ──
def _classify(src, timeout=60):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "test_x.py").write_text(src, encoding="utf-8")
        return G.run_pytest(td, "test_x.py", timeout=timeout)


def test_runtime_exception_is_its_own_category():
    cat, detail = _classify("def test_a():\n    {}['k']\n")
    assert cat == G.RUNTIME and "KeyError" in detail
    assert cat not in (G.CAUGHT, G.INVALID, G.SURVIVED, G.OK)


def test_runtime_exception_is_not_counted_as_assertion_caught(capsys):
    G.summarize([("x", G.RUNTIME, "KeyError")])
    out = capsys.readouterr().out
    assert "斷言抓到 0" in out and "執行期例外 1" in out


def test_runtime_exception_is_not_counted_as_invalid_and_does_not_fail_the_run(capsys):
    assert G.summarize([("x", G.RUNTIME, "KeyError"), ("y", G.CAUGHT, "")]) == 0
    assert "無效結果 0" in capsys.readouterr().out


def test_runtime_exception_does_not_hide_survivors_or_invalid_results():
    assert G.summarize([("x", G.RUNTIME, ""), ("y", G.SURVIVED, "")]) == 1
    assert G.summarize([("x", G.RUNTIME, ""), ("y", G.INVALID, "")]) == 4
    assert G.summarize([("x", G.RUNTIME, ""), ("y", G.SURVIVED, ""), ("z", G.INVALID, "")]) == 4


def test_mixed_assertion_and_runtime_failures_count_as_assertion_caught():
    cat, _ = _classify("def test_a():\n    assert 1 == 2\n\ndef test_b():\n    {}['k']\n")
    assert cat == G.CAUGHT


@pytest.mark.parametrize("src", [
    "import module_that_does_not_exist_xyz\n\ndef test_a():\n    assert True\n",          # 匯入失敗（收集階段）
    "def test_a(:\n    pass\n",                                                              # 語法錯誤
    "x = 1\n",                                                                                # 沒有收集到測試
    "import pytest\n\n@pytest.fixture\ndef f():\n    raise RuntimeError('x')\n\ndef test_a(f):\n    assert True\n",     # 設定錯誤
    "def test_a():\n    import module_that_does_not_exist_xyz\n",                          # 測試本體內的匯入失敗
])
def test_invalid_situations_stay_invalid(src):
    assert _classify(src)[0] == G.INVALID


def test_timeout_stays_invalid():
    assert _classify("import time\n\ndef test_a():\n    time.sleep(30)\n", timeout=3)[0] == G.INVALID


def test_evaluate_passes_runtime_category_through_without_changing_it():
    rels = HP.all_rels()
    cat, _ = G.evaluate(HP.ROOT, rels, HP.TEST_REL, HP.SRC_REL, lambda t: t + "\n# m\n", runner=lambda td, t, timeout: (G.RUNTIME, "x"))
    assert cat == G.RUNTIME


# ── 防呆被破壞時，自我檢查要抓得到 ──
def test_self_check_detects_a_guard_that_ignores_incomplete_copies(monkeypatch):
    monkeypatch.setattr(G, "verify_copy", lambda *a, **k: [])
    assert HP.self_check() != []


def test_self_check_detects_a_guard_that_skips_the_baseline(monkeypatch):
    monkeypatch.setattr(G, "run_baseline", lambda *a, **k: (True, ""))
    assert any("沒通過時沒有中止" in p for p in HP.self_check())


def test_self_check_detects_a_guard_that_counts_any_failure_as_caught(monkeypatch):
    monkeypatch.setattr(G, "ASSERT_PREFIXES", ("",))      # 任何失敗訊息都當成斷言失敗
    assert G.self_test() != []


def test_self_check_detects_a_guard_that_allows_no_op_mutations(monkeypatch):
    real = G.evaluate

    def lax(root, rels, test_rel, target_rel, fn, drop=None, runner=None, timeout=300):
        return real(root, rels, test_rel, target_rel, lambda t: fn(t) + "\n", drop=drop, runner=runner, timeout=timeout)
    monkeypatch.setattr(G, "evaluate", lax)
    assert HB.self_check() != []


# ── 腳本本身 ──
@pytest.mark.parametrize("name", list(HARNESSES))
def test_harnesses_do_not_run_external_programs_directly(name):
    path = HARNESSES[name].__file__
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add((n.module or "").split(".")[0])
    assert "subprocess" not in imported and "mutation_guard" in imported
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            assert n.attr not in ("system", "popen", "Popen", "check_output"), n.attr


@pytest.mark.parametrize("name", list(HARNESSES))
def test_harnesses_have_static_replacement_check(name):
    H = HARNESSES[name]
    assert H.FORBIDDEN.search("os." + "sys" + "tem(x)") and H.FORBIDDEN.search("sub" + "process") and H.FORBIDDEN.search("sh" + "ell")
    assert H.REPL and not [t for t in H.REPL if H.FORBIDDEN.search(t)]


def test_guard_module_runs_only_pytest_in_a_child_process():
    src = open(G.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert [c for c in calls if c.startswith("subprocess.")] == ["subprocess.run"]
    for n in ast.walk(tree):
        if isinstance(n, ast.keyword) and n.arg == "shell":
            raise AssertionError("不得使用外殼模式")
