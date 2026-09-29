"""
測試重點（n8n 本機執行紀錄清理，scripts/n8n_prune_executions.py）：

所有測試都用「自建的假資料庫」（tmp_path 裡的 sqlite，表結構是 n8n 真實結構的子集），
絕不碰正式 n8n 資料庫：autouse 的防護會在任何 sqlite3.connect 指向 n8n 資料卷路徑時直接失敗。

- 只刪「表單 workflow」超過天數的紀錄；審核 workflow（含很舊的）與其他 workflow 不動；
  新的表單紀錄不動；附屬表（metadata／annotations／data）跟著刪、沒有孤兒
- 日期邊界：--before 是「該日 00:00 UTC 之前」，剛好在邊界上的不刪
- dry-run 不刪、不備份；沒東西可刪不備份
- 超過筆數上限：結束碼 2，什麼都不刪、不備份
- 鎖不到（別的連線占著寫入鎖）：結束碼 4，資料完全不變
- 中途某批失敗：當批整批回滾（含附屬表），前面已提交的批次保留
- 備份：刪除前的內容、權限 600、輪替只留最新 N 份；完整性檢查不是 ok → 結束碼 3，不刪任何東西
- log 只有筆數與編號，不含問答內容（金絲雀）
- 找不到資料庫：結束碼 1，不會建立空檔
"""

import datetime
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import n8n_prune_executions as P

FORM = P.FORM_WORKFLOW_ID
APPROVAL = "4pYorQT9wbQIk41n"
OTHER = "SomeOtherWorkflow1"
CANARY_Q = "CANARY_QUESTION_5d2b 智財歸屬是誰"
CANARY_A = "CANARY_ANSWER_5d2b 這是回答"
NOW = datetime.datetime(2026, 10, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)
CUTOFF_30D = P.compute_cutoff(days=30, now=NOW)          # 2026-09-30 12:00:00.000

SCHEMA = """
create table execution_entity (id integer primary key autoincrement, workflowId varchar(36) not null,
                               startedAt datetime, status varchar not null);
create table execution_data (executionId int primary key not null, workflowData text not null, data text not null,
                             foreign key(executionId) references execution_entity(id) on delete cascade);
create table execution_metadata (id integer primary key autoincrement, executionId integer not null,
                                 key varchar(255) not null, value text not null);
create table execution_annotations (id integer primary key autoincrement, executionId integer not null,
                                    vote varchar(6), note text);
"""


@pytest.fixture(autouse=True)
def never_touch_the_production_database(monkeypatch):
    real_connect = sqlite3.connect

    def guarded(database, *a, **k):
        assert "/var/lib/docker" not in str(database) and "n8n_data" not in str(database), database
        return real_connect(database, *a, **k)

    monkeypatch.setattr(sqlite3, "connect", guarded)


def add(conn, wf, started, status="waiting", with_children=True):
    cur = conn.execute("insert into execution_entity(workflowId, startedAt, status) values (?,?,?)", (wf, started, status))
    eid = cur.lastrowid
    conn.execute("insert into execution_data values (?,?,?)", (eid, "{}", f"[\"{CANARY_Q}\",\"{CANARY_A}\"]"))
    if with_children:
        conn.execute("insert into execution_metadata(executionId,key,value) values (?,?,?)", (eid, "k", CANARY_Q))
        conn.execute("insert into execution_annotations(executionId,vote,note) values (?,?,?)", (eid, "up", CANARY_A))
    return eid


@pytest.fixture
def env(tmp_path):
    db = str(tmp_path / "database.sqlite")
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    ids = {
        "old_form_1": add(conn, FORM, "2026-09-17 01:53:08.564"),
        "old_form_2": add(conn, FORM, "2026-09-23 03:38:43.352"),
        "old_approval": add(conn, APPROVAL, "2026-09-01 00:00:00.000", status="success"),
        "old_other": add(conn, OTHER, "2026-08-01 00:00:00.000"),
        "new_form": add(conn, FORM, "2026-10-29 08:00:00.000"),
        "boundary_form": add(conn, FORM, CUTOFF_30D),                 # 剛好等於門檻：不刪（嚴格小於）
        "just_old_form": add(conn, FORM, "2026-09-30 11:59:59.999"),
    }
    conn.commit()
    conn.close()
    return SimpleEnv(db, str(tmp_path / "bak"), str(tmp_path / "prune.log"), ids)


class SimpleEnv:
    def __init__(self, db, bak, log, ids):
        self.db, self.bak, self.log, self.ids = db, bak, log, ids

    def run(self, cutoff=CUTOFF_30D, **kw):
        kw.setdefault("sleep", lambda s: None)
        return P.prune(self.db, cutoff, backup_dir=self.bak, log_path=self.log, **kw)

    def rows(self, table="execution_entity", col="id"):
        c = sqlite3.connect(self.db)
        try:
            return sorted(r[0] for r in c.execute(f"select {col} from {table}"))
        finally:
            c.close()

    def backups(self):
        return sorted(os.listdir(self.bak)) if os.path.isdir(self.bak) else []


def test_deletes_only_old_form_records(env):
    assert env.run() == 0
    i = env.ids
    assert env.rows() == sorted([i["old_approval"], i["old_other"], i["new_form"], i["boundary_form"]])


def test_children_removed_with_deleted_records_and_kept_for_the_rest(env):
    env.run()
    kept = env.rows()
    assert env.rows("execution_data", "executionId") == kept
    assert env.rows("execution_metadata", "executionId") == kept
    assert env.rows("execution_annotations", "executionId") == kept


def test_approval_records_untouched_even_when_very_old(env):
    env.run(cutoff="2030-01-01 00:00:00.000", max_delete=1000)
    assert set(env.rows()) == {env.ids["old_approval"], env.ids["old_other"]}


def test_before_date_boundary_is_strictly_before_midnight_utc(env):
    conn = sqlite3.connect(env.db)
    a = add(conn, FORM, "2026-09-23 23:59:59.999")
    b = add(conn, FORM, "2026-09-24 00:00:00.000")
    conn.commit(); conn.close()
    assert P.compute_cutoff(before="2026-09-24") == "2026-09-24 00:00:00.000"
    env.run(cutoff=P.compute_cutoff(before="2026-09-24"))
    rows = env.rows()
    assert a not in rows and b in rows


def test_compute_cutoff_days_and_invalid_date():
    assert P.compute_cutoff(days=30, now=NOW) == "2026-09-30 12:00:00.000"
    with pytest.raises(ValueError):
        P.compute_cutoff(before="2026-13-40")


def test_dry_run_deletes_nothing_and_makes_no_backup(env):
    before = env.rows()
    assert env.run(dry_run=True) == 0
    assert env.rows() == before and env.backups() == []


def test_nothing_to_delete_makes_no_backup(env):
    assert env.run(cutoff="2020-01-01 00:00:00.000") == 0
    assert env.backups() == []


def test_over_limit_aborts_without_deleting_or_backing_up(env):
    before = env.rows()
    assert env.run(max_delete=2) == 2
    assert env.rows() == before and env.backups() == []


def test_exactly_at_limit_is_allowed(env):
    assert env.run(max_delete=3) == 0            # 剛好 3 筆要刪
    assert len(env.rows()) == 4


def test_lock_timeout_rolls_back_and_changes_nothing(env):
    before = env.rows()
    blocker = sqlite3.connect(env.db, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")           # 模擬 n8n 占著寫入鎖
    try:
        assert env.run(busy_ms=200) == 4
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert env.rows() == before
    assert env.rows("execution_data", "executionId") == before


def test_failing_batch_rolls_back_whole_batch_and_keeps_committed_ones(env):
    conn = sqlite3.connect(env.db)
    target = env.ids["just_old_form"]              # 排序最後一筆要刪的
    conn.execute(f"create trigger boom before delete on execution_entity when old.id = {target} "
                 "begin select raise(abort, 'boom'); end")
    conn.commit(); conn.close()
    assert env.run(batch=2) == 4                   # 待刪 3 筆：批 1 = 前 2 筆成功，批 2 = 最後 1 筆失敗
    i = env.ids
    rows = env.rows()
    assert i["old_form_1"] not in rows and i["old_form_2"] not in rows       # 已提交的批次保留
    assert target in rows                                                    # 失敗那批整批回滾
    assert target in env.rows("execution_data", "executionId")               # 附屬表也回滾
    assert target in env.rows("execution_metadata", "executionId")
    assert target in env.rows("execution_annotations", "executionId")


def test_backup_taken_before_deleting_with_private_permissions(env):
    before = env.rows()
    env.run()
    (name,) = env.backups()
    path = os.path.join(env.bak, name)
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    b = sqlite3.connect(path)
    assert sorted(r[0] for r in b.execute("select id from execution_entity")) == before
    assert b.execute("pragma integrity_check").fetchone()[0] == "ok"
    b.close()


def test_backup_integrity_not_ok_aborts_without_deleting(env, monkeypatch):
    before = env.rows()
    monkeypatch.setattr(P, "backup_and_check", lambda conn, d, now=None: ("/x/bak.sqlite", "*** in database main ***", 7))
    assert env.run() == 3
    assert env.rows() == before
    assert env.rows("execution_data", "executionId") == before


def test_backup_is_created_before_any_delete(env, monkeypatch):
    order = []
    real = P.backup_and_check
    monkeypatch.setattr(P, "backup_and_check", lambda *a, **k: order.append("backup") or real(*a, **k))
    real_sleep = lambda s: order.append("batch")
    env.run(sleep=real_sleep)
    assert order[0] == "backup" and "batch" in order


def test_rotation_keeps_only_the_newest_backups(tmp_path):
    d = tmp_path / "bak"
    d.mkdir()
    for n in range(1, 11):
        (d / f"database-2026090{n % 10}-00000{n}.sqlite").write_text("x")
    (d / "other.txt").write_text("keep")
    removed = P.rotate_backups(str(d), keep=7)
    left = sorted(os.listdir(d))
    assert len(removed) == 3 and "other.txt" in left and len([f for f in left if f.startswith("database-")]) == 7


def test_log_has_counts_and_ids_but_no_question_or_answer_text(env, capsys):
    env.run()
    text = open(env.log, encoding="utf-8").read() + capsys.readouterr().out
    assert "待刪 3 筆" in text and "刪除 3 筆" in text and "id " in text
    assert "CANARY" not in text and "智財" not in text and "回答" not in text.replace("回答內容", "")


def test_log_after_abort_also_has_no_content(env, capsys):
    env.run(max_delete=1)
    text = open(env.log, encoding="utf-8").read() + capsys.readouterr().out
    assert "超過上限" in text and "CANARY" not in text


def test_missing_database_returns_1_and_creates_nothing(tmp_path):
    db = str(tmp_path / "nope.sqlite")
    assert P.prune(db, CUTOFF_30D, backup_dir=str(tmp_path / "bak"), log_path=str(tmp_path / "l.log")) == 1
    assert not os.path.exists(db) and not os.path.exists(tmp_path / "bak")


def test_preexisting_orphan_data_is_reported_with_code_5(env):
    conn = sqlite3.connect(env.db)
    conn.execute("insert into execution_data values (9999, '{}', '[]')")
    conn.commit(); conn.close()
    assert env.run() == 5


def test_running_twice_is_harmless(env):
    assert env.run() == 0
    after_first = env.rows()
    assert env.run() == 0
    assert env.rows() == after_first and len(env.backups()) == 1


def test_main_requires_exactly_one_of_days_or_before(tmp_path, capsys):
    with pytest.raises(SystemExit):
        P.main(["--db", str(tmp_path / "x.sqlite")])
    with pytest.raises(SystemExit):
        P.main(["--days", "30", "--before", "2026-09-24", "--db", str(tmp_path / "x.sqlite")])


def test_main_dry_run_end_to_end_on_a_fake_database(env):
    before = env.rows()
    assert P.main(["--days", "30", "--dry-run", "--db", env.db, "--backup-dir", env.bak, "--log", env.log]) == 0
    assert env.rows() == before


def test_production_paths_are_the_documented_ones():
    assert P.DB == "/var/lib/docker/volumes/n8n_data/_data/database.sqlite"
    assert P.FORM_WORKFLOW_ID == "XctHEK3cQdOp6WBd"
    assert (P.BATCH, P.MAX_DELETE, P.KEEP_BACKUPS, P.BUSY_MS) == (50, 500, 7, 30000)
