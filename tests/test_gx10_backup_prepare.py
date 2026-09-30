"""
測試重點（GX10 備份整理，scripts/gx10_backup_prepare.py；還原測試腳本 gx10_backup_verify.py 只測純邏輯部分）：

全部使用自建的假資料（tmp_path 裡的假 n8n 資料庫、假 trial 資料夾、本機假的 Qdrant HTTP 伺服器），
不碰正式服務與資料：

- 備份包內容：Qdrant 快照（集合名稱現查）、n8n 資料庫（線上備份且 integrity_check）、config、trial 與 eval 壓縮檔
- trial 壓縮檔排除 n8n-prune-backup 與 __pycache__；其餘都在
- MANIFEST／SHA256SUMS 與實際檔案一致；記下 Qdrant 筆數與內容雜湊、n8n 筆數與流程名稱、config 雜湊
- 先做在 .tmp-，成功才改名並放 DONE、更新 LATEST；沒有 DONE 的不算
- 任何一步失敗：結束碼 1、不留半成品、寫 FAILED（只有步驟與例外種類，不含內容）；下次成功會清掉 FAILED
- n8n 資料庫 integrity_check 不是 ok 就失敗
- 暫存只留最新 KEEP 份成功的，遺留的 .tmp- 也清掉
- 權限：資料夾 700、檔案 600
- log 只有筆數與大小，不含問答內容（金絲雀）
- 還原測試的檔案核對：能抓到雜湊不符、缺檔、清單不一致
"""

import http.server
import json
import os
import sqlite3
import sys
import tarfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import gx10_backup_prepare as B
import gx10_backup_verify as V

CANARY = "CANARY_QA_TEXT_77aa 智財歸屬是誰"


class FakeQdrant(http.server.BaseHTTPRequestHandler):
    collections = {"psf_eim_kb": [1, 2, 3], "psf_test_kb": [1]}
    deleted = []
    fail_snapshot = False

    def _send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("content-type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        p = self.path
        if p == "/collections":
            self._send({"result": {"collections": [{"name": n} for n in self.collections]}})
        elif "/snapshots/" in p:
            self._send(b"SNAPSHOT-BYTES-" + p.split("/")[2].encode(), "application/octet-stream")

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        p = self.path
        if p.endswith("/points/scroll"):
            name = p.split("/")[2]
            self._send({"result": {"points": [{"id": i, "payload": {"t": CANARY}, "vector": [0.1, 0.2]} for i in self.collections[name]],
                                   "next_page_offset": None}})
        elif p.endswith("/snapshots"):
            if self.fail_snapshot:
                self.send_response(500)
                self.end_headers()
                return
            self._send({"result": {"name": "snap-1.snapshot"}})

    def do_DELETE(self):
        FakeQdrant.deleted.append(self.path)
        self._send({"result": True})

    def log_message(self, *a):
        pass


@pytest.fixture
def qdrant():
    FakeQdrant.deleted = []
    FakeQdrant.fail_snapshot = False
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeQdrant)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def env(tmp_path, qdrant):
    n8n = tmp_path / "n8n"
    n8n.mkdir()
    db = sqlite3.connect(n8n / "database.sqlite")
    db.executescript("""create table workflow_entity(id text, name text);
                        create table credentials_entity(id text, name text);
                        create table execution_entity(id integer primary key, data text);""")
    db.executemany("insert into workflow_entity values (?,?)", [("a", "My workflow"), ("b", "PSF EIM QA Approval")])
    db.execute("insert into credentials_entity values ('c','Langfuse Basic Auth')")
    db.execute("insert into execution_entity(data) values (?)", (CANARY,))
    db.commit()
    db.close()
    (n8n / "config").write_text('{"encryptionKey":"k"}')
    trial = tmp_path / "trial"
    (trial / "n8n-prune-backup").mkdir(parents=True)
    (trial / "n8n-prune-backup" / "old.sqlite").write_text("old db")
    (trial / "__pycache__").mkdir()
    (trial / "__pycache__" / "x.pyc").write_text("x")
    (trial / "usage-manual.md").write_text("手冊")
    ev = tmp_path / "eval" / "task2-deeptech"
    ev.mkdir(parents=True)
    (ev / "doc.md").write_text("語料")
    return dict(staging=str(tmp_path / "staging"), qdrant_url=qdrant, n8n_dir=str(n8n), trial_dir=str(trial),
                eval_dirs=[str(ev)], owner_uid=os.getuid(), with_docker=False, log_path=str(tmp_path / "b.log"), tmp=tmp_path)


def go(env, **kw):
    args = {k: v for k, v in env.items() if k != "tmp"}
    args.update(kw)
    return B.run(**args)


def latest(env):
    return os.path.join(env["staging"], open(os.path.join(env["staging"], "LATEST")).read().strip())


def test_backup_contains_everything_and_marks_done(env):
    assert go(env) == 0
    d = latest(env)
    assert os.path.exists(os.path.join(d, "DONE"))
    for rel in ("qdrant/psf_eim_kb.snapshot", "qdrant/psf_test_kb.snapshot", "n8n/database.sqlite", "n8n/config",
                "trial.tar.gz", "eval-task2-deeptech.tar.gz", "MANIFEST.json", "SHA256SUMS"):
        assert os.path.exists(os.path.join(d, rel)), rel


def test_snapshots_are_deleted_from_qdrant_after_download(env):
    go(env)
    assert sorted(FakeQdrant.deleted) == sorted(f"/collections/{n}/snapshots/snap-1.snapshot" for n in FakeQdrant.collections)


def test_manifest_matches_actual_files_and_records_verification_facts(env):
    go(env)
    d = latest(env)
    m = json.load(open(os.path.join(d, "MANIFEST.json")))
    assert V.check_files(d, m) == []
    assert m["qdrant"]["psf_eim_kb"]["points"] == 3 and len(m["qdrant"]["psf_eim_kb"]["fingerprint"]) == 64
    assert m["n8n"]["counts"] == {"workflow_entity": 2, "credentials_entity": 1, "execution_entity": 1}
    assert m["n8n"]["workflow_names"] == ["My workflow", "PSF EIM QA Approval"]
    assert m["n8n"]["config_sha256"] == B.sha256_file(os.path.join(env["n8n_dir"], "config"))


def test_n8n_database_backup_is_a_consistent_copy(env):
    go(env)
    c = sqlite3.connect(os.path.join(latest(env), "n8n", "database.sqlite"))
    assert c.execute("pragma integrity_check").fetchone()[0] == "ok"
    assert c.execute("select count(*) from workflow_entity").fetchone()[0] == 2


def test_trial_tar_excludes_prune_backups_and_pycache_but_keeps_the_rest(env):
    go(env)
    with tarfile.open(os.path.join(latest(env), "trial.tar.gz")) as t:
        names = t.getnames()
    assert any(n.endswith("usage-manual.md") for n in names)
    assert not any("n8n-prune-backup" in n or "__pycache__" in n for n in names)


def test_eval_tar_only_contains_the_listed_subdirectory(env):
    go(env)
    with tarfile.open(os.path.join(latest(env), "eval-task2-deeptech.tar.gz")) as t:
        assert any(n.endswith("doc.md") for n in t.getnames())


def test_permissions_are_private(env):
    go(env)
    for root, dirs, files in os.walk(env["staging"]):
        for n in dirs:
            assert oct(os.stat(os.path.join(root, n)).st_mode & 0o777) == "0o700", os.path.join(root, n)
        for n in files:
            assert oct(os.stat(os.path.join(root, n)).st_mode & 0o777) == "0o600", os.path.join(root, n)
    assert oct(os.stat(env["staging"]).st_mode & 0o777) == "0o700"


def test_failure_leaves_no_partial_backup_and_writes_failed_marker(env):
    FakeQdrant.fail_snapshot = True
    assert go(env) == 1
    names = os.listdir(env["staging"])
    assert names == ["FAILED"]
    text = open(os.path.join(env["staging"], "FAILED")).read()
    assert "步驟=qdrant" in text and "CANARY" not in text


def test_failure_after_a_good_backup_keeps_the_good_one_and_success_clears_failed(env):
    assert go(env) == 0
    good = latest(env)
    FakeQdrant.fail_snapshot = True
    assert go(env, now=__import__("datetime").datetime(2030, 1, 1)) == 1
    assert os.path.exists(os.path.join(good, "DONE")) and open(os.path.join(env["staging"], "LATEST")).read().strip() == os.path.basename(good)
    FakeQdrant.fail_snapshot = False
    assert go(env, now=__import__("datetime").datetime(2030, 1, 2)) == 0
    assert not os.path.exists(os.path.join(env["staging"], "FAILED"))


def test_n8n_integrity_failure_aborts(env, monkeypatch):
    real = sqlite3.connect

    class Corrupt(sqlite3.Connection):
        def execute(self, sql, *a):
            if "integrity_check" in sql:
                return real(":memory:").execute("select '*** corrupt ***'")
            return super().execute(sql, *a)

    def connect(path, *a, **k):
        if str(path).endswith("database.sqlite") and "staging" in str(path):
            k["factory"] = Corrupt
        return real(path, *a, **k)

    monkeypatch.setattr(B.sqlite3, "connect", connect)
    assert go(env) == 1
    assert "步驟=n8n-integrity" in open(os.path.join(env["staging"], "FAILED")).read()


def test_only_the_newest_backups_are_kept_and_stale_tmp_is_removed(env):
    import datetime
    os.makedirs(env["staging"])
    os.makedirs(os.path.join(env["staging"], ".tmp-stale"))
    for day in (1, 2, 3, 4):
        assert go(env, now=datetime.datetime(2030, 1, day), keep=2) == 0
    kept = sorted(d for d in os.listdir(env["staging"]) if d[0].isdigit())
    assert kept == ["20300103-000000", "20300104-000000"]
    assert not any(d.startswith(".tmp-") for d in os.listdir(env["staging"]))


def test_log_has_no_question_or_answer_text(env):
    go(env)
    text = open(env["log_path"], encoding="utf-8").read()
    assert "備份完成" in text and "CANARY" not in text and "智財" not in text


def test_verify_file_check_detects_tampering_and_missing_files(env):
    go(env)
    d = latest(env)
    m = json.load(open(os.path.join(d, "MANIFEST.json")))
    with open(os.path.join(d, "trial.tar.gz"), "ab") as f:
        f.write(b"x")
    assert any("雜湊不符 trial.tar.gz" in p for p in V.check_files(d, m))
    os.remove(os.path.join(d, "n8n", "config"))
    assert any("缺檔 n8n/config" in p for p in V.check_files(d, m))


def test_verify_file_check_detects_file_list_mismatch(env):
    go(env)
    d = latest(env)
    m = json.load(open(os.path.join(d, "MANIFEST.json")))
    m["files"].pop("n8n/config")
    assert any("清單不同" in p for p in V.check_files(d, m))


def test_collection_fingerprint_changes_when_content_changes(qdrant):
    a = B.collection_fingerprint(qdrant, "psf_eim_kb")
    FakeQdrant.collections["psf_eim_kb"] = [1, 2]
    try:
        b = B.collection_fingerprint(qdrant, "psf_eim_kb")
    finally:
        FakeQdrant.collections["psf_eim_kb"] = [1, 2, 3]
    assert a[1] == 3 and b[1] == 2 and a[0] != b[0]


def test_backed_up_config_is_byte_identical_to_the_source(env):
    go(env)
    src = os.path.join(env["n8n_dir"], "config")
    assert open(os.path.join(latest(env), "n8n", "config"), "rb").read() == open(src, "rb").read()
