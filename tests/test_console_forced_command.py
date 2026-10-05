"""
測試重點（控制台強制指令，scripts/console_forced_command.py）：這支放在 authorized_keys，
是「拿到 gx10_console 金鑰的人能做什麼」的唯一邊界，所以測它的邊界：
- 只有完全相同的 `status`、`summary` 被接受；輸出是已產生的檔案內容
- 空輸入、None、超長、大小寫變化、前後空白、換行、分號、&&、管線、反引號、$()、路徑跳脫、參數、null 位元組、
  全形字元、其他指令（shell、cat、docker、get）一律拒絕（結束碼 2），且不輸出任何檔案內容
- 注入的指令不會被執行（標記檔不會出現）；本檔不引用 subprocess、os.system 或 docker，也不以外殼方式執行任何東西
- 就算輸出目錄外面有機密檔案，任何輸入都拿不到；不寫入輸出目錄內的資料檔（只寫 serve.log）
- log 每次只有一行，被拒絕的輸入截短並跳脫（不會有換行或控制字元，不會洩漏長內容）
- 檔案不存在時回結束碼 3，不拋例外
"""
import ast
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import console_forced_command as F

OUTSIDE_MARKER = "OUTSIDE_SECRET_5f2c"
STATUS_BODY = '{"overall": "green"}\n'
SUMMARY_BODY = "整體燈號：綠\n"


@pytest.fixture
def env(tmp_path):
    d = tmp_path / "console"
    d.mkdir()
    (d / "status.json").write_text(STATUS_BODY)
    (d / "daily-summary.txt").write_text(SUMMARY_BODY)
    (d / "state.json").write_text("STATE_SHOULD_NOT_BE_SERVED")
    (tmp_path / "secret.txt").write_text(OUTSIDE_MARKER)
    (tmp_path / "authorized_keys").write_text(OUTSIDE_MARKER)
    return dict(out_dir=str(d), tmp=tmp_path)


def call(env, cmd):
    out, err = io.BytesIO(), io.BytesIO()
    rc = F.serve(cmd, out, err, out_dir=env["out_dir"])
    return rc, out.getvalue(), err.getvalue()


def test_status_and_summary_exact_only(env):
    assert call(env, "status") == (0, STATUS_BODY.encode(), b"")
    assert call(env, "summary") == (0, SUMMARY_BODY.encode(), b"")


REJECTED = [
    "", None, " ", "status ", " status", "status\n", "\nstatus", "status\r\n", "\tstatus", "summary ", " summary",
    "STATUS", "Status", "sTatus", "SUMMARY", "Summary", "statuss", "stat", "summ", "status status", "status summary",
    "status;ls", "status; cat /etc/passwd", "status&&id", "status || id", "status|cat", "status`id`", "`status`",
    "$(status)", "status$(id)", "status\nid", "status\x00", "status\x00id", "status > /tmp/x", "status#",
    "../status", "./status", "/status", "status/", "status/..", "status.json", "../../etc/passwd", "/etc/passwd",
    "state", "state.json", "cat status", "cat status.json", "get status", "latest", "failed", "ls", "sh", "bash -c status",
    "docker ps", "docker exec n8n sh", "sudo status", "status --help", "-status", "ｓｔａｔｕｓ", "status​", "ꜱtatus",
    "a" * 5000, "status" + "a" * 100000, "status\n" * 100, "\x1b[31mstatus", "%73tatus", "status%0a", 123, b"status", ["status"],
]


@pytest.mark.parametrize("cmd", REJECTED, ids=lambda c: repr(c)[:30])
def test_everything_else_is_rejected_without_output(env, cmd):
    rc, out, err = call(env, cmd)
    assert rc == 2 and out == b""
    assert err == b"not allowed\n"


def test_rejected_input_is_never_executed(env):
    marker = env["tmp"] / "marker"
    for cmd in ("status; touch %s" % marker, "status && touch %s" % marker, "`touch %s`" % marker,
                "$(touch %s)" % marker, "status\ntouch %s" % marker, "touch %s" % marker):
        assert call(env, cmd)[0] == 2
    assert not marker.exists()


def test_no_path_escape_and_no_other_files_served(env):
    for cmd in ("../secret.txt", "../authorized_keys", "state", "state.json", "serve.log", "status/../state.json"):
        rc, out, err = call(env, cmd)
        assert rc == 2 and OUTSIDE_MARKER.encode() not in out + err and b"STATE_SHOULD_NOT" not in out + err


def test_accepted_output_never_contains_other_files(env):
    for cmd in ("status", "summary"):
        rc, out, _ = call(env, cmd)
        assert OUTSIDE_MARKER.encode() not in out and b"STATE_SHOULD_NOT" not in out


def test_missing_file_returns_3_without_exception(env):
    os.unlink(os.path.join(env["out_dir"], "status.json"))
    rc, out, err = call(env, "status")
    assert rc == 3 and out == b"" and err == b"not available\n"
    assert call(env, "summary")[0] == 0


def test_unreadable_directory_returns_3(env):
    rc, out, err = call(dict(env, out_dir=os.path.join(env["out_dir"], "nope")), "status")
    assert rc == 3 and out == b""


def test_large_file_is_capped(env):
    big = os.path.join(env["out_dir"], "status.json")
    with open(big, "wb") as f:
        f.write(b"x" * (F.MAX_BYTES + 1000))
    rc, out, _ = call(env, "status")
    assert rc == 0 and len(out) == F.MAX_BYTES


def test_does_not_modify_data_files(env):
    d = env["out_dir"]
    before = {n: open(os.path.join(d, n), "rb").read() for n in os.listdir(d)}
    for cmd in ("status", "summary", "x", "", "status;rm -rf /"):
        call(env, cmd)
    after = {n: open(os.path.join(d, n), "rb").read() for n in os.listdir(d) if n != "serve.log"}
    assert after == before
    assert set(os.listdir(d)) == set(before) | {"serve.log"}


def test_log_is_one_line_per_call_and_sanitized(env):
    long_bad = "x" * 5000 + "SECRET_TAIL_9f"
    for cmd in ("status", "summary", "bad\nINJECTED_LINE", "\x1b[2J" + "y" * 100, long_bad, "", None):
        call(env, cmd)
    lines = open(os.path.join(env["out_dir"], "serve.log"), encoding="utf-8").read().splitlines()
    assert len(lines) == 7
    assert "INJECTED_LINE" not in [l.split(" ", 2)[-1] for l in lines]      # 換行被跳脫，沒有變成獨立的 log 行
    assert all("\x1b" not in l for l in lines)
    assert not any("SECRET_TAIL_9f" in l for l in lines)                   # 超長輸入被截短
    assert max(len(l) for l in lines) < 120
    assert lines[0].endswith(" status") and lines[1].endswith(" summary")
    assert "拒絕" in lines[2]


def test_log_does_not_contain_file_content(env):
    call(env, "status")
    log = open(os.path.join(env["out_dir"], "serve.log"), encoding="utf-8").read()
    assert "green" not in log and "整體燈號" not in log


def test_log_write_failure_does_not_break_response(env):
    out, err = io.BytesIO(), io.BytesIO()
    rc = F.serve("status", out, err, out_dir=env["out_dir"], log_path="/nonexistent-dir/x/serve.log")
    assert rc == 0 and out.getvalue() == STATUS_BODY.encode()


def test_source_has_no_subprocess_shell_or_docker():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "console_forced_command.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imported.add((n.module or "").split(".")[0])
    assert imported <= {"datetime", "os", "sys"}
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not {c for c in calls if c.startswith(("os.system", "os.popen", "os.exec", "os.spawn", "subprocess", "eval", "exec", "__import__"))}
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "shell"]


def test_command_only_compared_never_used_as_a_path_or_argument():
    """被接受的只有兩個固定字串；檔案名稱來自固定對照表，不是輸入。"""
    assert F.FILES == {"status": "status.json", "summary": "daily-summary.txt"}


def test_main_reads_env_var_only(env, monkeypatch, capsysbinary):
    monkeypatch.setenv("CONSOLE_OUT_DIR", env["out_dir"])
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "status")
    assert F.main() == 0
    assert capsysbinary.readouterr().out == STATUS_BODY.encode()
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "STATUS")
    assert F.main() == 2
    monkeypatch.delenv("SSH_ORIGINAL_COMMAND")
    assert F.main() == 2      # 沒有帶指令（例如互動登入）也拒絕


def test_no_secrets_or_real_addresses_in_source():
    import re
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "console_forced_command.py")
    src = open(path, encoding="utf-8").read()
    assert not re.search(r"(?i)\b(password|passwd|secret|api_key|token)\s*=\s*[\"']", src)
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src)
    assert "psf01" not in src


# ── serve.log 的權限一律 600 ──
def _mode(path):
    return os.stat(path).st_mode & 0o777


def test_new_log_file_is_created_with_mode_600_even_with_open_umask(env):
    log = os.path.join(env["out_dir"], "serve.log")
    old = os.umask(0)
    try:
        call(env, "status")
    finally:
        os.umask(old)
    assert _mode(log) == 0o600


def test_existing_wide_log_is_tightened_before_writing_and_content_kept(env):
    log = os.path.join(env["out_dir"], "serve.log")
    with open(log, "w", encoding="utf-8") as f:
        f.write("2026-01-01 00:00:00 舊的一行\n")
    os.chmod(log, 0o664)
    call(env, "status")
    assert _mode(log) == 0o600
    lines = open(log, encoding="utf-8").read().splitlines()
    assert lines[0].endswith("舊的一行") and len(lines) == 2 and lines[1].endswith(" status")


def test_log_mode_stays_600_after_many_calls(env):
    log = os.path.join(env["out_dir"], "serve.log")
    for cmd in ("status", "summary", "x", ""):
        call(env, cmd)
        assert _mode(log) == 0o600


def test_log_is_opened_with_mode_600_directly_not_relying_on_a_later_chmod(env, monkeypatch):
    modes = []
    real_open = os.open

    def spy(path, flags, mode=0o777, **kw):
        modes.append(mode)
        return real_open(path, flags, mode, **kw)
    monkeypatch.setattr(F.os, "open", spy)
    call(env, "status")
    assert modes == [0o600]
