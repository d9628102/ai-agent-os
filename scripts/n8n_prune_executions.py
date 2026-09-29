#!/usr/bin/env python3
"""n8n 本機執行紀錄清理（方案 B）。規格：/home/psf01/trial/n8n-cleanup-spec.md

只刪「表單問答 workflow」超過保留天數的執行紀錄（含 execution_data 內的問答全文）。
n8n 自己的清理會跳過 waiting 狀態，表單紀錄永遠是 waiting，所以要靠這支。

安全設計：
- 有東西要刪才備份；備份用 SQLite 線上備份 API（WAL 安全），備份後做 integrity_check，
  不是 ok 就中止、不刪任何東西
- 每批 50 筆、每批一個短交易（BEGIN IMMEDIATE，等鎖最多 30 秒，等不到就放棄這批並停止）
- 先刪附屬表（metadata、annotations、data）再刪 execution_entity，任何錯誤整批 ROLLBACK
- 單次上限 500 筆，超過就中止（避免條件寫錯一次刪光）
- 只刪 FORM_WORKFLOW_ID（審核 workflow 與其他 workflow 的紀錄不動）；不做 VACUUM
- log 只寫筆數、編號、日期範圍，不寫任何執行內容（問題、回答）

用法（要能讀寫 n8n 資料庫，通常需要 root）：
  n8n_prune_executions.py --dry-run [--days 30 | --before YYYY-MM-DD]
  n8n_prune_executions.py --days 30            （每日排程）
  n8n_prune_executions.py --before 2026-09-24  （一次性；日期為 UTC，該日 00:00 之前）

結束碼：0 完成或沒東西可刪；1 找不到資料庫；2 待刪筆數超過上限；3 備份完整性檢查
不是 ok；4 刪除中途出錯（當批已回滾）；5 刪完後發現孤兒 execution_data。
"""
import argparse
import datetime
import os
import sqlite3
import sys
import time

DB = "/var/lib/docker/volumes/n8n_data/_data/database.sqlite"
FORM_WORKFLOW_ID = "XctHEK3cQdOp6WBd"
BACKUP_DIR = "/home/psf01/trial/n8n-prune-backup"
LOG = "/home/psf01/trial/n8n-prune/prune.log"
BATCH, MAX_DELETE, KEEP_BACKUPS, BUSY_MS = 50, 500, 7, 30000


def _log(msg, log_path):
    line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def stats(conn):
    n_e = conn.execute("select count(*) from execution_entity").fetchone()[0]
    n_f = conn.execute("select count(*) from execution_entity where workflowId=?", (FORM_WORKFLOW_ID,)).fetchone()[0]
    n_d = conn.execute("select count(*) from execution_data").fetchone()[0]
    return f"execution_entity={n_e} (表單 {n_f}) execution_data={n_d}"


def size_line(db):
    parts = []
    for suffix in ("", "-wal"):
        p = db + suffix
        parts.append(f"{os.path.basename(p)}={os.path.getsize(p) / 1e6:.2f}MB" if os.path.exists(p) else f"{suffix or 'db'}=無")
    return " ".join(parts)


def compute_cutoff(days=None, before=None, now=None):
    """回傳 (UTC 字串)：開始時間早於這個字串的紀錄要刪。n8n 的 startedAt 是 UTC。"""
    if before:
        datetime.date.fromisoformat(before)   # 格式錯誤直接拋 ValueError
        return before + " 00:00:00.000"
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (now - datetime.timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S.000")


def backup_and_check(conn, backup_dir, now=None):
    """線上備份＋對備份檔做 integrity_check。回傳 (備份路徑, integrity 結果, 備份內的 execution_entity 筆數)。"""
    os.makedirs(backup_dir, mode=0o700, exist_ok=True)
    now = now or datetime.datetime.now()
    bak = os.path.join(backup_dir, "database-" + now.strftime("%Y%m%d-%H%M%S") + ".sqlite")
    dst = sqlite3.connect(bak)
    conn.backup(dst)
    dst.close()
    os.chmod(bak, 0o600)
    chk = sqlite3.connect(bak)
    integ = chk.execute("pragma integrity_check").fetchone()[0]
    n_bak = chk.execute("select count(*) from execution_entity").fetchone()[0]
    chk.close()
    return bak, integ, n_bak


def rotate_backups(backup_dir, keep=KEEP_BACKUPS):
    backups = sorted(f for f in os.listdir(backup_dir) if f.startswith("database-") and f.endswith(".sqlite"))
    removed = backups[:-keep] if keep > 0 else backups
    for old in removed:
        os.remove(os.path.join(backup_dir, old))
    return removed


def prune(db, cutoff, dry_run=False, backup_dir=BACKUP_DIR, log_path=None, max_delete=MAX_DELETE,
          batch=BATCH, busy_ms=BUSY_MS, truncate_checkpoint=False, sleep=time.sleep, keep_backups=KEEP_BACKUPS):
    log = lambda m: _log(m, log_path)
    if not os.path.exists(db):
        log(f"找不到資料庫 {db}，不做事")
        return 1
    conn = sqlite3.connect(db, timeout=busy_ms / 1000)
    conn.execute(f"PRAGMA busy_timeout={busy_ms}")
    try:
        ids = [r[0] for r in conn.execute(
            "select id from execution_entity where workflowId=? and startedAt < ? order by id", (FORM_WORKFLOW_ID, cutoff))]
        mm = conn.execute("select min(startedAt), max(startedAt) from execution_entity where id in (%s)" %
                          ",".join("?" * len(ids)), ids).fetchone() if ids else (None, None)
        log(f"開始 dry_run={dry_run} cutoff(UTC)<{cutoff} 待刪 {len(ids)} 筆 範圍 {mm[0]} ~ {mm[1]} | 前 {stats(conn)} | {size_line(db)}")
        if dry_run or not ids:
            return 0
        if len(ids) > max_delete:
            log(f"待刪 {len(ids)} 筆超過上限 {max_delete}，中止，未刪任何東西")
            return 2

        bak, integ, n_bak = backup_and_check(conn, backup_dir)
        log(f"備份 {bak} integrity_check={integ} execution_entity={n_bak}")
        if integ != "ok":
            log("備份完整性檢查不是 ok，中止，未刪任何東西")
            return 3
        for old in rotate_backups(backup_dir, keep_backups):
            log(f"輪替刪除舊備份 {old}")

        conn.isolation_level = None
        deleted = 0
        try:
            for i in range(0, len(ids), batch):
                part = ids[i:i + batch]
                q = ",".join("?" * len(part))
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute(f"delete from execution_metadata where executionId in ({q})", part)
                    conn.execute(f"delete from execution_annotations where executionId in ({q})", part)
                    conn.execute(f"delete from execution_data where executionId in ({q})", part)
                    cur = conn.execute(f"delete from execution_entity where id in ({q}) and workflowId=?",
                                       part + [FORM_WORKFLOW_ID])
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                deleted += cur.rowcount
                log(f"批次 {i // batch + 1}：刪除 {cur.rowcount} 筆（id {part[0]}~{part[-1]}）")
                sleep(0.5)
        except sqlite3.Error as e:
            log(f"刪除中斷（已整批回滾當批）：{type(e).__name__}: {e}；已刪 {deleted} 筆")
            return 4
        orphans = conn.execute(
            "select count(*) from execution_data where executionId not in (select id from execution_entity)").fetchone()[0]
        if truncate_checkpoint or datetime.date.today().weekday() == 6:
            mode = "TRUNCATE" if truncate_checkpoint else "PASSIVE"
            log(f"checkpoint({mode}) -> {conn.execute(f'PRAGMA wal_checkpoint({mode})').fetchone()}")
        st = os.stat(db)
        conn.close()   # 最後一個連線關閉時 SQLite 可能刪除 -wal／-shm；n8n 停機時由我們重建，擁有者要跟主檔一致
        if os.geteuid() == 0:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(db + suffix):
                    os.chown(db + suffix, st.st_uid, st.st_gid)
        conn = sqlite3.connect(db, timeout=busy_ms / 1000)
        log(f"完成 刪除 {deleted} 筆 孤兒 execution_data={orphans} | 後 {stats(conn)} | {size_line(db)} | 擁有者 uid={st.st_uid}")
        return 0 if orphans == 0 else 5
    finally:
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--days", type=int)
    g.add_argument("--before", help="UTC 日期 YYYY-MM-DD，刪除開始時間早於這天 00:00 的紀錄")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--backup-dir", default=BACKUP_DIR)
    ap.add_argument("--log", default=LOG)
    ap.add_argument("--max-delete", type=int, default=MAX_DELETE)
    ap.add_argument("--truncate-checkpoint", action="store_true", help="只在 n8n 已停止時使用")
    a = ap.parse_args(argv)
    return prune(a.db, compute_cutoff(a.days, a.before), dry_run=a.dry_run, backup_dir=a.backup_dir,
                 log_path=a.log, max_delete=a.max_delete, truncate_checkpoint=a.truncate_checkpoint)


if __name__ == "__main__":
    sys.exit(main())
