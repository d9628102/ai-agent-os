#!/usr/bin/env python3
"""備份還原測試：證明備份真的能還原。全部在拋棄式容器與暫存資料夾進行，不碰正式服務。

用法：gx10_backup_verify.py [備份資料夾]      （不給就用 STAGING/LATEST 指的那一份）
檢查項目（每項都要通過；任何一項失敗結束碼 1，這份備份視為無效）
  1 檔案核對：SHA256SUMS 與實際檔案一致、MANIFEST 列的檔案都在
  2 Qdrant：起一個空的 Qdrant 容器（127.0.0.1:6335），匯入每個快照，筆數與「全部點的內容雜湊」
            與備份當下記錄的一致
  3 n8n：用備份的資料庫與 config 起一個新的 n8n 容器（127.0.0.1:5680），確認：config 雜湊與備份時一致、
         能正常啟動、列出的流程名稱與備份時一致、憑證筆數一致、資料庫完整性 ok、沒有加密金鑰不符的錯誤
  4 壓縮檔：trial.tar.gz 與 eval 壓縮檔能完整讀取（含 CRC），trial 內有預期的檔案
容器與暫存資料夾一律在結束時清除。報告（只有通過／失敗與筆數，不含內容）存成
/home/psf01/trial/backup/restore-test-<時間>.json。
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gx10_backup_prepare as prep

STAGING = prep.STAGING
REPORT_DIR = "/home/psf01/trial/backup"
QDRANT_TEST, N8N_TEST = "qdrant-restore-test", "n8n-restore-test"
QDRANT_VOLUME = "qdrant-restore-test-vol"
QDRANT_PORT, N8N_PORT = 6335, 5680


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def wait_http(url, seconds):
    end = time.time() + seconds
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return True
        except Exception:
            time.sleep(2)
    return False


def check_files(bdir, manifest):
    problems = []
    sums = {}
    with open(os.path.join(bdir, "SHA256SUMS")) as f:
        for line in f:
            sha, rel = line.rstrip("\n").split("  ", 1)
            sums[rel] = sha
    if set(sums) != set(manifest["files"]):
        problems.append("SHA256SUMS 與 MANIFEST 的檔案清單不同")
    for rel, meta in manifest["files"].items():
        p = os.path.join(bdir, rel)
        if not os.path.exists(p):
            problems.append(f"缺檔 {rel}")
        elif prep.sha256_file(p) != meta["sha256"] or sums.get(rel) != meta["sha256"]:
            problems.append(f"雜湊不符 {rel}")
    return problems


def check_qdrant(bdir, manifest, tmp):
    problems, detail = [], {}
    sh(["docker", "rm", "-f", QDRANT_TEST])
    sh(["docker", "volume", "rm", "-f", QDRANT_VOLUME])
    # 用 docker 具名資料卷而不是資料夾：Qdrant 在容器裡以 root 寫檔，資料夾會留下我們刪不掉的檔案
    r = sh(["docker", "run", "-d", "--name", QDRANT_TEST, "-p", f"127.0.0.1:{QDRANT_PORT}:6333", "-v", f"{QDRANT_VOLUME}:/qdrant/storage", "qdrant/qdrant"])
    if r.returncode != 0:
        return ["無法啟動測試用 Qdrant 容器"], detail
    url = f"http://127.0.0.1:{QDRANT_PORT}"
    if not wait_http(url + "/healthz", 60):
        return ["測試用 Qdrant 沒有在 60 秒內啟動"], detail
    for name, meta in manifest["qdrant"].items():
        snap = os.path.join(bdir, "qdrant", f"{name}.snapshot")
        up = sh(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", "POST",
                 f"{url}/collections/{name}/snapshots/upload?priority=snapshot", "-F", f"snapshot=@{snap}"])
        if up.stdout.strip() != "200":
            problems.append(f"集合 {name} 匯入快照失敗（HTTP {up.stdout.strip()}）")
            continue
        fp, n = prep.collection_fingerprint(url, name)
        detail[name] = {"points": n}
        if n != meta["points"]:
            problems.append(f"集合 {name} 筆數不符（備份 {meta['points']}、還原 {n}）")
        elif fp != meta["fingerprint"]:
            problems.append(f"集合 {name} 內容雜湊不符")
    return problems, detail


def check_n8n(bdir, manifest, tmp):
    problems, detail = [], {}
    ndir = os.path.join(tmp, "n8n-data")
    os.makedirs(ndir)
    for fn in ("database.sqlite", "config"):
        shutil.copy2(os.path.join(bdir, "n8n", fn), os.path.join(ndir, fn))
        os.chmod(os.path.join(ndir, fn), 0o600)
    if os.geteuid() == 0:
        for fn in ("", "database.sqlite", "config"):
            os.chown(os.path.join(ndir, fn) if fn else ndir, 1000, 1000)
    if prep.sha256_file(os.path.join(ndir, "config")) != manifest["n8n"]["config_sha256"]:
        problems.append("config 雜湊與備份時不符")
    sh(["docker", "rm", "-f", N8N_TEST])
    r = sh(["docker", "run", "-d", "--name", N8N_TEST, "-p", f"127.0.0.1:{N8N_PORT}:5678", "-v", f"{ndir}:/home/node/.n8n",
            "-e", "N8N_SECURE_COOKIE=false", "-e", "N8N_DIAGNOSTICS_ENABLED=false", "n8nio/n8n"])
    if r.returncode != 0:
        return problems + ["無法啟動測試用 n8n 容器"], detail
    if not wait_http(f"http://127.0.0.1:{N8N_PORT}/healthz", 120):
        return problems + ["測試用 n8n 沒有在 120 秒內啟動"], detail
    logs = sh(["docker", "logs", N8N_TEST]).stdout + sh(["docker", "logs", N8N_TEST]).stderr
    if "ismatch" in logs and "ncryption" in logs:
        problems.append("n8n 日誌出現加密金鑰不符")
    names = sorted(l.split("|", 1)[1] for l in sh(["docker", "exec", N8N_TEST, "n8n", "list:workflow"]).stdout.splitlines() if "|" in l)
    detail["workflows"] = len(names)
    if names != manifest["n8n"]["workflow_names"]:
        problems.append(f"流程名稱不符（備份 {manifest['n8n']['workflow_names']}、還原 {names}）")
    sh(["docker", "stop", N8N_TEST])
    import sqlite3
    c = sqlite3.connect(os.path.join(ndir, "database.sqlite"))
    integ = c.execute("pragma integrity_check").fetchone()[0]
    creds = c.execute("select count(*) from credentials_entity").fetchone()[0]
    execs = c.execute("select count(*) from execution_entity").fetchone()[0]
    c.close()
    detail.update(credentials=creds, executions=execs, integrity=integ)
    if integ != "ok":
        problems.append("還原後的資料庫完整性不是 ok")
    if creds != manifest["n8n"]["counts"]["credentials_entity"]:
        problems.append("憑證筆數不符")
    if execs < manifest["n8n"]["counts"]["execution_entity"]:
        problems.append("執行紀錄筆數比備份時少")
    return problems, detail


def check_tars(bdir, manifest):
    problems, detail = [], {}
    for rel in manifest["files"]:
        if rel.endswith(".tar.gz"):
            try:
                with tarfile.open(os.path.join(bdir, rel)) as t:
                    members = t.getmembers()
                    for m in members:
                        if m.isfile():
                            t.extractfile(m).read()
                detail[rel] = len(members)
                if rel == "trial.tar.gz" and not any(m.name.endswith("usage-manual.md") for m in members):
                    problems.append("trial.tar.gz 裡找不到 usage-manual.md")
                if any("n8n-prune-backup" in m.name for m in members):
                    problems.append("trial.tar.gz 不該包含 n8n-prune-backup")
            except Exception as e:
                problems.append(f"{rel} 讀取失敗：{type(e).__name__}")
    return problems, detail


def verify(bdir, report_dir=REPORT_DIR):
    with open(os.path.join(bdir, "MANIFEST.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    tmp = tempfile.mkdtemp(prefix="restore-test-")
    report = {"backup": os.path.basename(bdir.rstrip("/")), "tested_at": datetime.datetime.now().isoformat(timespec="seconds"), "checks": {}}
    try:
        p = check_files(bdir, manifest)
        report["checks"]["files"] = {"ok": not p, "problems": p}
        if not p:
            for name, fn in (("qdrant", check_qdrant), ("n8n", check_n8n)):
                problems, detail = fn(bdir, manifest, tmp)
                report["checks"][name] = {"ok": not problems, "problems": problems, "detail": detail}
            problems, detail = check_tars(bdir, manifest)
            report["checks"]["tars"] = {"ok": not problems, "problems": problems, "detail": detail}
    finally:
        sh(["docker", "rm", "-f", QDRANT_TEST, N8N_TEST])
        sh(["docker", "volume", "rm", "-f", QDRANT_VOLUME])
        shutil.rmtree(tmp, ignore_errors=True)
    report["passed"] = all(c["ok"] for c in report["checks"].values()) and len(report["checks"]) == 4
    os.makedirs(report_dir, exist_ok=True)
    out = os.path.join(report_dir, "restore-test-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report, out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    bdir = argv[0] if argv else os.path.join(STAGING, open(os.path.join(STAGING, "LATEST")).read().strip())
    report, out = verify(bdir)
    for name, c in report["checks"].items():
        print(f"{'通過' if c['ok'] else '失敗'}  {name}  {c.get('detail', '')}  {'; '.join(c['problems'])}")
    print(f"報告：{out}\n結果：{'全部通過' if report['passed'] else '有失敗，這份備份視為無效'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
