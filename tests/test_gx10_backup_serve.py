"""
測試重點（Mac 拉取備份用的強制指令，scripts/gx10_backup_serve.py）：這支放在 authorized_keys，
是「拿到 Mac 專用金鑰的人能做什麼」的唯一邊界，所以測它的邊界：
- latest／failed／get 三個動作各自正確；get 只給有 DONE 的備份，輸出是可解開的 tar
- 名稱嚴格格式：.. 、斜線、絕對路徑、萬用字元、多餘空白參數、空字串、其他指令（shell、cat、rsync）一律拒絕
- 就算 STAGING 外面有機密檔案，任何輸入都拿不到
- 不寫入任何東西（STAGING 內容在呼叫前後不變）
- log 只有動作與結果，不含備份內容（金絲雀）
"""
import io
import os
import sys
import tarfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import gx10_backup_serve as S

SECRET = "OUTSIDE_SECRET_5f2c"
CANARY = "CANARY_QA_TEXT_31de"


@pytest.fixture
def env(tmp_path):
    st = tmp_path / "staging"
    good = st / "20260929-171000"
    good.mkdir(parents=True)
    (good / "DONE").write_text("x")
    (good / "data.txt").write_text(CANARY)
    (good / "sub").mkdir()
    (good / "sub" / "b.txt").write_text("b")
    bad = st / "20260930-023000"
    bad.mkdir()
    (bad / "data.txt").write_text("partial")          # 沒有 DONE
    (st / "LATEST").write_text("20260929-171000\n")
    (tmp_path / "secret.txt").write_text(SECRET)      # STAGING 外面的機密
    (tmp_path / "authorized_keys").write_text(SECRET)
    return dict(staging=str(st), log_path=str(tmp_path / "serve.log"), tmp=tmp_path)


def call(env, cmd):
    out, err = io.BytesIO(), io.BytesIO()
    rc = S.serve(cmd, out, err, staging=env["staging"], log_path=env["log_path"])
    return rc, out.getvalue(), err.getvalue()


def snapshot(path):
    return sorted((os.path.join(r, f), open(os.path.join(r, f), "rb").read()) for r, _, fs in os.walk(path) for f in fs)


def test_latest_prints_the_name(env):
    rc, out, _ = call(env, "latest")
    assert rc == 0 and out.decode().strip() == "20260929-171000"


def test_failed_absent_returns_3_and_present_prints_it(env):
    assert call(env, "failed")[0] == 3
    open(os.path.join(env["staging"], "FAILED"), "w").write("2026-09-29 步驟=qdrant")
    rc, out, _ = call(env, "failed")
    assert rc == 0 and "步驟=qdrant".encode() in out


def test_get_streams_a_valid_tar_of_the_finished_backup(env):
    rc, out, _ = call(env, "get 20260929-171000")
    assert rc == 0
    with tarfile.open(fileobj=io.BytesIO(out)) as t:
        names = t.getnames()
        assert "20260929-171000/DONE" in names and "20260929-171000/sub/b.txt" in names
        assert t.extractfile("20260929-171000/data.txt").read().decode() == CANARY


def test_get_refuses_backups_without_done(env):
    rc, out, _ = call(env, "get 20260930-023000")
    assert rc == 4 and out == b""


def test_get_missing_backup_returns_3(env):
    assert call(env, "get 20200101-000000")[0] == 3


@pytest.mark.parametrize("cmd", [
    "", "   ", "ls", "cat /etc/passwd", "bash", "sh -c id", "rsync --server --sender -logDtpr . /",
    "latest extra", "failed x", "get", "get   ", "get 20260929-171000 extra",
    "get ../secret.txt", "get ../../authorized_keys", "get /etc/passwd", "get /home/psf01/backup-staging/20260929-171000",
    "get 20260929-171000/../..", "get 20260929-171000/", "get 20260929-17100", "get 2026092-1710000", "get *", "get 2026*",
    "get 20260929-171000;id", "get $(id)", "get `id`", "GET 20260929-171000", "Latest",
    "get 20260929-171000 && cat secret", "latest;cat /etc/passwd",
])
def test_everything_else_is_refused_and_leaks_nothing(env, cmd):
    rc, out, _ = call(env, cmd)
    assert rc in (2, 3)
    assert SECRET.encode() not in out and b"root:" not in out


def test_nothing_outside_staging_is_ever_readable(env):
    for cmd in ("get ../secret.txt", "get ../authorized_keys", "get ....-......"):
        assert SECRET.encode() not in call(env, cmd)[1]


def test_serving_never_modifies_staging(env):
    before = snapshot(env["staging"])
    for cmd in ("latest", "failed", "get 20260929-171000", "get 20260930-023000", "bogus"):
        call(env, cmd)
    assert snapshot(env["staging"]) == before


def test_log_has_actions_but_no_content(env):
    call(env, "get 20260929-171000")
    call(env, "get ../secret.txt")
    text = open(env["log_path"], encoding="utf-8").read()
    assert "get 20260929-171000" in text and "格式不合法" in text
    assert CANARY not in text and SECRET not in text
