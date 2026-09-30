#!/usr/bin/env python3
"""GX10 備份整理（每晚由 root 排程執行）。規格：/home/psf01/trial/backup-spec.md

把要備份的東西整理成一份「一致的備份包」，放在 STAGING/<日期時間>/，由 Mac 用唯讀金鑰來拉取
（Mac 端限制成 rrsync -ro，只看得到 STAGING）。GX10 上不存任何 Mac 的密碼。

備份內容
  qdrant/<集合>.snapshot   Qdrant 快照（線上、一致；集合名稱每次現查）
  n8n/database.sqlite      n8n 資料庫（SQLite 線上備份 API，做完 integrity_check）
  n8n/config               n8n 的憑證加密金鑰檔（拿到資料庫加這個檔就能解出所有憑證）
  trial.tar.gz             /home/psf01/trial（排除 n8n-prune-backup、__pycache__）
  eval-task2-deeptech.tar.gz
  docker-inspect.json      四個容器的建立參數（可能含 token，敏感）
  root-crontab.txt
  MANIFEST.json、SHA256SUMS，最後才放 DONE

一致性與失敗處理
  - 整包先做在 .tmp-<時間>，全部成功後才改名並放 DONE；Mac 只拉有 DONE 的
  - 任何一步失敗：不留半成品，STAGING/FAILED 寫下失敗的步驟與例外種類（不寫內容）
  - STAGING 只留最新 KEEP 份成功的（給 Mac 拉取用，不是備份本身）
  - 權限：資料夾 700、檔案 600、擁有者 OWNER_UID（內含問答全文與憑證加密金鑰）
  - log 只有筆數、大小、雜湊，不含問答內容
結束碼：0 成功；1 失敗。
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import urllib.request

STAGING = "/home/psf01/backup-staging"
QDRANT_URL = "http://localhost:6333"
N8N_DIR = "/var/lib/docker/volumes/n8n_data/_data"
TRIAL_DIR = "/home/psf01/trial"
EVAL_SUBDIRS = ["/home/psf01/eval/task2-deeptech"]
LOG = "/home/psf01/trial/backup/backup.log"
OWNER_UID = 1000
KEEP = 2
CONTAINERS = ["n8n", "qdrant", "vllm-server", "vllm-embed"]
TRIAL_EXCLUDES = ("n8n-prune-backup", "__pycache__")


class BackupError(Exception):
    def __init__(self, step, detail=""):
        super().__init__(f"{step}: {detail}")
        self.step = step


def log(msg, log_path=LOG):
    line = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http(method, url, data=None, timeout=120):
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def collection_fingerprint(qdrant_url, name):
    """所有點（編號、內容、向量）的雜湊，還原後用來證明內容一致。"""
    points, offset = [], None
    while True:
        body = {"limit": 256, "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        res = json.loads(http("POST", f"{qdrant_url}/collections/{name}/points/scroll", json.dumps(body).encode()))["result"]
        points += res["points"]
        offset = res.get("next_page_offset")
        if offset is None:
            break
    points.sort(key=lambda p: str(p["id"]))
    return hashlib.sha256(json.dumps(points, sort_keys=True, ensure_ascii=False).encode()).hexdigest(), len(points)


def backup_qdrant(qdrant_url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    names = [c["name"] for c in json.loads(http("GET", f"{qdrant_url}/collections"))["result"]["collections"]]
    info = {}
    for name in names:
        fp, n = collection_fingerprint(qdrant_url, name)
        snap = json.loads(http("POST", f"{qdrant_url}/collections/{name}/snapshots", timeout=600))["result"]["name"]
        try:
            data = http("GET", f"{qdrant_url}/collections/{name}/snapshots/{snap}", timeout=600)
        finally:
            try:
                http("DELETE", f"{qdrant_url}/collections/{name}/snapshots/{snap}")
            except Exception:
                pass
        with open(os.path.join(out_dir, f"{name}.snapshot"), "wb") as f:
            f.write(data)
        info[name] = {"points": n, "fingerprint": fp}
    return info


def backup_n8n(n8n_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(n8n_dir, "database.sqlite")
    dst_path = os.path.join(out_dir, "database.sqlite")
    s = sqlite3.connect(src, timeout=30)
    d = sqlite3.connect(dst_path)
    s.backup(d)
    s.close()
    d.close()
    chk = sqlite3.connect(dst_path)
    integ = chk.execute("pragma integrity_check").fetchone()[0]
    counts = {t: chk.execute(f"select count(*) from {t}").fetchone()[0]
              for t in ("workflow_entity", "credentials_entity", "execution_entity")}
    names = sorted(r[0] for r in chk.execute("select name from workflow_entity"))
    chk.close()
    if integ != "ok":
        raise BackupError("n8n-integrity", "integrity_check 不是 ok")
    cfg = os.path.join(n8n_dir, "config")
    shutil.copy2(cfg, os.path.join(out_dir, "config"))
    return {"counts": counts, "workflow_names": names, "config_sha256": sha256_file(cfg)}


def make_tar(src_dir, out_path, excludes=()):
    def flt(ti):
        parts = ti.name.split("/")
        return None if any(p in excludes for p in parts) else ti
    with tarfile.open(out_path, "w:gz") as t:
        t.add(src_dir, arcname=os.path.basename(src_dir.rstrip("/")), filter=flt)


def collect_docker(out_path, containers=CONTAINERS):
    r = subprocess.run(["docker", "inspect"] + containers, capture_output=True, text=True)
    if r.returncode != 0:
        raise BackupError("docker-inspect", f"rc={r.returncode}")
    with open(out_path, "w") as f:
        f.write(r.stdout)


def collect_crontab(out_path):
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    with open(out_path, "w") as f:
        f.write(r.stdout if r.returncode == 0 else "")


def secure(path, owner_uid):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
        for f in files:
            os.chmod(os.path.join(root, f), 0o600)
    os.chmod(path, 0o700)
    if os.geteuid() == 0:
        for root, dirs, files in os.walk(path):
            for x in [root] + [os.path.join(root, n) for n in dirs + files]:
                os.chown(x, owner_uid, owner_uid)


def write_manifest(tmp, extra):
    files = {}
    for root, _, names in os.walk(tmp):
        for n in sorted(names):
            p = os.path.join(root, n)
            files[os.path.relpath(p, tmp)] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    manifest = dict(extra, created=datetime.datetime.now().isoformat(timespec="seconds"), files=files)
    with open(os.path.join(tmp, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(tmp, "SHA256SUMS"), "w") as f:
        for rel in sorted(files):
            f.write(f"{files[rel]['sha256']}  {rel}\n")
    return manifest


def rotate(staging, keep=KEEP):
    done = sorted(d for d in os.listdir(staging)
                  if os.path.isfile(os.path.join(staging, d, "DONE")))
    for old in done[:-keep] if keep > 0 else done:
        shutil.rmtree(os.path.join(staging, old))
    for d in os.listdir(staging):          # 遺留的半成品
        if d.startswith(".tmp-"):
            shutil.rmtree(os.path.join(staging, d), ignore_errors=True)
    return done[:-keep] if keep > 0 else done


def run(staging=STAGING, qdrant_url=QDRANT_URL, n8n_dir=N8N_DIR, trial_dir=TRIAL_DIR, eval_dirs=None,
        owner_uid=OWNER_UID, keep=KEEP, with_docker=True, now=None, log_path=LOG):
    L = lambda m: log(m, log_path)
    eval_dirs = EVAL_SUBDIRS if eval_dirs is None else eval_dirs
    now = now or datetime.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    os.makedirs(staging, mode=0o700, exist_ok=True)
    os.chmod(staging, 0o700)
    if os.geteuid() == 0:
        os.chown(staging, owner_uid, owner_uid)
    tmp = os.path.join(staging, ".tmp-" + stamp)
    step = "start"
    try:
        os.makedirs(tmp)
        step = "qdrant"
        q = backup_qdrant(qdrant_url, os.path.join(tmp, "qdrant"))
        step = "n8n"
        n = backup_n8n(n8n_dir, os.path.join(tmp, "n8n"))
        step = "trial"
        make_tar(trial_dir, os.path.join(tmp, "trial.tar.gz"), TRIAL_EXCLUDES)
        step = "eval"
        for d in eval_dirs:
            make_tar(d, os.path.join(tmp, f"eval-{os.path.basename(d)}.tar.gz"))
        step = "docker"
        if with_docker:
            collect_docker(os.path.join(tmp, "docker-inspect.json"))
            collect_crontab(os.path.join(tmp, "root-crontab.txt"))
        step = "manifest"
        manifest = write_manifest(tmp, {"qdrant": q, "n8n": n, "host": os.uname().nodename})
        secure(tmp, owner_uid)
        final = os.path.join(staging, stamp)
        os.rename(tmp, final)
        with open(os.path.join(final, "DONE"), "w") as f:
            f.write(stamp + "\n")
        secure(final, owner_uid)
        with open(os.path.join(staging, "LATEST.tmp"), "w") as f:
            f.write(stamp + "\n")
        os.rename(os.path.join(staging, "LATEST.tmp"), os.path.join(staging, "LATEST"))
        failed = os.path.join(staging, "FAILED")
        if os.path.exists(failed):
            os.remove(failed)
        removed = rotate(staging, keep)
        for p in (os.path.join(staging, "LATEST"),):
            os.chmod(p, 0o600)
            if os.geteuid() == 0:
                os.chown(p, owner_uid, owner_uid)
        total = sum(v["size"] for v in manifest["files"].values())
        L(f"備份完成 {stamp} 檔案 {len(manifest['files'])} 個 共 {total / 1e6:.2f}MB "
          f"qdrant集合={ {k: v['points'] for k, v in q.items()} } n8n={n['counts']} 輪替刪除暫存 {removed}")
        return 0
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        detail = getattr(e, "step", step)
        with open(os.path.join(staging, "FAILED"), "w") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} 步驟={detail} 例外={type(e).__name__}\n")
        if os.geteuid() == 0:
            os.chown(os.path.join(staging, "FAILED"), owner_uid, owner_uid)
        L(f"備份失敗 步驟={detail} 例外={type(e).__name__}（已寫入 FAILED，未留下半成品）")
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="GX10 備份整理")
    ap.add_argument("--staging", default=STAGING)
    ap.add_argument("--qdrant-url", default=QDRANT_URL)
    ap.add_argument("--n8n-dir", default=N8N_DIR)
    ap.add_argument("--trial-dir", default=TRIAL_DIR)
    ap.add_argument("--keep", type=int, default=KEEP)
    ap.add_argument("--log", default=LOG)
    a = ap.parse_args(argv)
    return run(a.staging, a.qdrant_url, a.n8n_dir, a.trial_dir, keep=a.keep, log_path=a.log)


if __name__ == "__main__":
    sys.exit(main())
