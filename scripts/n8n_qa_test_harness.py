#!/usr/bin/env python3
"""
n8n_qa_test_harness.py

Reliably submit a question to the PSF EIM n8n form and fetch the execution
it actually produced -- fixing a race condition in the ad-hoc verification
pattern used while building the QA gate (submit, then immediately assume
"the newest execution in the list" is the one just triggered).

Root cause: POSTing to the form webhook returns almost instantly (n8n has
only accepted the request), but the execution record isn't necessarily
written to n8n's execution store yet, and generation itself takes anywhere
from under a second to over a minute. A "query latest, trust it" read done
right after the POST can still see the *previous* execution.

Fix: track the highest execution id that existed *before* submitting (an
id, not a wall-clock timestamp -- avoids clock skew between this machine
and GX10), then poll for a *new* id whose "On form submission" trigger data
literally matches the question string just submitted, and whose run has
actually progressed past the QA stage (not just the trigger). Only that
combination counts as "found the right execution" -- an id merely being
numerically newest is never sufficient on its own.

Standard library only, matching this repo's other scripts.

Usage:
  N8N_BASE_URL=http://192.168.1.128:5678 \
  N8N_API_KEY=... \
  python3 scripts/n8n_qa_test_harness.py "問題1" "問題2" ...

  # Or run the built-in 10-question stability sweep:
  python3 scripts/n8n_qa_test_harness.py --self-test

  # Or re-verify a batch of already-known (execution_id, expected_question)
  # pairs by id -- no submission, no race condition possible, just checks
  # that each id's recorded trigger question matches what it's supposed to be:
  python3 scripts/n8n_qa_test_harness.py --verify-ids 24:"問題1" 25:"問題2" ...
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

N8N_BASE_URL = os.environ.get("N8N_BASE_URL", "http://192.168.1.128:5678")
N8N_API_KEY = os.environ.get("N8N_API_KEY")
N8N_WORKFLOW_ID = os.environ.get("N8N_WORKFLOW_ID", "XctHEK3cQdOp6WBd")
FORM_WEBHOOK_URL = os.environ.get(
    "N8N_FORM_WEBHOOK_URL",
    f"{N8N_BASE_URL}/form/9fe3bbe9-faac-4162-a750-d7b6c0648cdb",
)
QUESTION_FIELD = os.environ.get("N8N_QUESTION_FIELD", "field-0")

POLL_INTERVAL_S = 3
DEFAULT_TIMEOUT_S = 110

# Nodes that must have output before an execution counts as "actually ran
# the QA pipeline", not just the trigger + SSH call.
REQUIRED_NODES = ("Code in JavaScript", "QA Parse & Threshold")


def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def http_json(method: str, url: str, payload=None, headers=None, timeout=15):
    """Returns (status_code, parsed_body_or_None)."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return e.code, {"_raw_error_body": raw.decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        return None, {"_connection_error": str(e.reason)}


def n8n_api(method: str, path: str, payload=None, timeout=15):
    if not N8N_API_KEY:
        die("N8N_API_KEY 環境變數沒有設定。")
    return http_json(
        method,
        f"{N8N_BASE_URL}{path}",
        payload=payload,
        headers={"X-N8N-API-KEY": N8N_API_KEY},
        timeout=timeout,
    )


def submit_question(question: str) -> None:
    """POST the question to the form webhook as multipart/form-data."""
    boundary = uuid.uuid4().hex
    parts = [
        f"--{boundary}",
        f'Content-Disposition: form-data; name="{QUESTION_FIELD}"',
        "",
        question,
        f"--{boundary}--",
        "",
    ]
    body = "\r\n".join(parts).encode("utf-8")
    req = urllib.request.Request(FORM_WEBHOOK_URL, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        die(f"送出表單失敗 (HTTP {e.code}): {e.read().decode('utf-8', errors='replace')}")
    except urllib.error.URLError as e:
        die(f"無法連到表單端點 {FORM_WEBHOOK_URL}: {e.reason}")


def get_max_execution_id() -> int:
    """Highest execution id currently on record, or 0 if none / on error."""
    status, resp = n8n_api(
        "GET", f"/api/v1/executions?workflowId={N8N_WORKFLOW_ID}&limit=1"
    )
    if status != 200 or not resp or not resp.get("data"):
        return 0
    try:
        return int(resp["data"][0]["id"])
    except (KeyError, ValueError, TypeError):
        return 0


def fetch_execution(execution_id) -> dict:
    status, resp = n8n_api(
        "GET", f"/api/v1/executions/{execution_id}?includeData=true"
    )
    if status != 200 or resp is None:
        return {}
    return resp


def get_run_data(execution: dict) -> dict:
    return execution.get("data", {}).get("resultData", {}).get("runData", {})


def get_trigger_question(execution: dict):
    run = get_run_data(execution)
    trig = run.get("On form submission")
    if not trig:
        return None
    try:
        return trig[0]["data"]["main"][0][0]["json"].get("你的問題")
    except (KeyError, IndexError, TypeError):
        return None


def execution_is_complete(execution: dict) -> bool:
    run = get_run_data(execution)
    return all(node in run and run[node] for node in REQUIRED_NODES)


def poll_for_execution(question: str, baseline_id: int,
                        timeout_s: int = DEFAULT_TIMEOUT_S,
                        poll_interval_s: int = POLL_INTERVAL_S) -> dict:
    """
    Poll until an execution newer than baseline_id both (a) recorded exactly
    this question at the trigger and (b) has progressed through the QA
    pipeline. Returns a result dict; on timeout, includes a snapshot of the
    last thing seen so a timeout is debuggable, not just "it failed".
    """
    deadline = time.time() + timeout_s
    last_snapshot = {"checked_ids": [], "note": "no candidate ids seen yet"}

    while time.time() < deadline:
        status, resp = n8n_api(
            "GET", f"/api/v1/executions?workflowId={N8N_WORKFLOW_ID}&limit=20"
        )
        if status != 200 or not resp:
            last_snapshot = {"note": f"executions list query failed (status={status})",
                              "response": resp}
            time.sleep(poll_interval_s)
            continue

        candidate_ids = sorted(
            (int(e["id"]) for e in resp.get("data", []) if int(e["id"]) > baseline_id)
        )
        checked = []
        for eid in candidate_ids:
            execution = fetch_execution(eid)
            actual_q = get_trigger_question(execution)
            complete = execution_is_complete(execution)
            checked.append({"id": eid, "question": actual_q, "complete": complete})
            if actual_q == question:
                if complete:
                    return {
                        "success": True,
                        "execution_id": eid,
                        "execution": execution,
                    }
                # Right question, but still mid-flight -- keep polling, this
                # is expected for slow (thinking-mode) generations.
        last_snapshot = {"checked_ids": checked, "note": "no complete+matching id yet"}
        log(f"    ...輪詢中 (已等 {int(time.time() - (deadline - timeout_s))}s)")
        time.sleep(poll_interval_s)

    return {
        "success": False,
        "reason": f"timeout after {timeout_s}s",
        "last_snapshot": last_snapshot,
    }


def run_one(question: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> dict:
    log(f"送出:「{question}」")
    baseline_id = get_max_execution_id()
    t0 = time.time()
    submit_question(question)
    result = poll_for_execution(question, baseline_id, timeout_s=timeout_s)
    result["elapsed_s"] = round(time.time() - t0, 1)
    result["question"] = question
    status = "PASS" if result.get("success") else "FAIL"
    log(f"  -> {status}，耗時 {result['elapsed_s']}s"
        + (f"，execution {result.get('execution_id')}" if result.get("success") else ""))
    return result


SELF_TEST_QUESTIONS = [
    "PSF EIM 的定價是多少？",
    "PSF EIM 的五大產品矩陣是什麼？",
    "PSF EIM 的導入時程大概要多久？",
    "PSF EIM 跟一般 ERP/CRM 有什麼不同？",
    "企業智慧架構總共分成幾層？",
    "PSF EIM 裡負責客戶服務的 Agent 是誰？",
    "PSF EIM 的核心價值主張是什麼？",
    "PSF EIM Service Matrix™ 跟 PSF Customer Success Agent™ 有什麼關聯？",
    "PSF EIM 支援哪些程式語言的 SDK？",
    "PSF EIM 的第六個產品矩陣叫什麼名字？",
]


def print_report(results: list) -> bool:
    print(f"\n{'=' * 78}")
    print(f"{'#':<3}{'PASS':<6}{'耗時(s)':<9}{'execution':<11}問題")
    print("=" * 78)
    all_pass = True
    for i, r in enumerate(results, 1):
        ok = r.get("success", False)
        all_pass = all_pass and ok
        eid = r.get("execution_id", "-")
        print(f"{i:<3}{'PASS' if ok else 'FAIL':<6}{r['elapsed_s']:<9}{str(eid):<11}{r['question']}")
        if not ok:
            print(f"    逾時快照: {json.dumps(r.get('last_snapshot', {}), ensure_ascii=False)[:500]}")
    print("=" * 78)
    passed = sum(1 for r in results if r.get("success"))
    print(f"結果: {passed}/{len(results)} 通過")
    return all_pass


def verify_known_ids(pairs: list) -> bool:
    """
    pairs: list of (execution_id, expected_question). No submission, no
    polling -- fetches each id directly and checks the recorded question,
    to audit historical executions for the same class of mismatch without
    re-running the model.
    """
    print(f"\n{'=' * 78}")
    print(f"{'exec_id':<10}{'PASS':<6}問題")
    print("=" * 78)
    all_ok = True
    for eid, expected in pairs:
        execution = fetch_execution(eid)
        actual = get_trigger_question(execution)
        ok = actual == expected
        all_ok = all_ok and ok
        print(f"{eid:<10}{'PASS' if ok else 'FAIL':<6}{expected}")
        if not ok:
            print(f"    預期: {expected!r}")
            print(f"    實際: {actual!r}")
    print("=" * 78)
    print("全部相符:" , all_ok)
    return all_ok


def main():
    args = sys.argv[1:]
    if not args:
        die("用法: n8n_qa_test_harness.py \"問題1\" [\"問題2\" ...] | --self-test | --verify-ids id:問題 ...")

    if args[0] == "--self-test":
        results = [run_one(q) for q in SELF_TEST_QUESTIONS]
        ok = print_report(results)
        sys.exit(0 if ok else 1)

    if args[0] == "--verify-ids":
        pairs = []
        for item in args[1:]:
            eid_str, _, question = item.partition(":")
            pairs.append((int(eid_str), question))
        ok = verify_known_ids(pairs)
        sys.exit(0 if ok else 1)

    results = [run_one(q) for q in args]
    ok = print_report(results)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
