#!/usr/bin/env python3
"""
code_review_agent.py

Manually-triggered code review for .py files in this repo. Runs four
independent layers -- syntax, style consistency, security, and (when a spec
document is given) spec alignment -- and prints one text report. It never
approves or blocks anything; a human reads the report and decides, same
principle as the QA gate and the delivery-center approval flow.

Scope: .py files only. No n8n workflow JSON, no .md docs -- if that scope
ever needs to grow, that's a separate task, not something to fold in here.

Layers 1-3 are plain static analysis (ast, regex) -- deterministic, no
network calls. Layer 4 (spec alignment) is the one layer that genuinely
needs reading comprehension, not pattern matching, so it calls the same
local Qwen3-30B-A3B endpoint rag_answer.py already uses, the same way the
QA gate's LLM judge does. Skipped entirely when no --spec is given.

Usage:
  python3 scripts/code_review_agent.py --files scripts/foo.py
  python3 scripts/code_review_agent.py --commit ca35295
  python3 scripts/code_review_agent.py --commit ca35295 --spec docs/some-spec.md
  python3 scripts/code_review_agent.py --files scripts/foo.py --baseline scripts/rag_answer.py scripts/n8n_qa_test_harness.py

  # Gate mode, used by githooks/pre-push -- reviews everything that changed
  # between two refs; exits non-zero on any blocking finding unless a valid
  # --override already covers it:
  python3 scripts/code_review_agent.py --range <old_sha>..<new_sha>

  # Human override for a blocked --range: re-runs the same review, and if
  # there are still blocking findings, records who/when/why/which findings
  # were overridden (locally + in Langfuse), good for OVERRIDE_TTL_MINUTES:
  python3 scripts/code_review_agent.py --override --range <old_sha>..<new_sha> \\
      --approver "your name" --reason "why this should go through anyway"
"""
import argparse
import ast
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

DEFAULT_BASELINE = ["scripts/rag_answer.py", "scripts/n8n_qa_test_harness.py"]
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8000/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-30B-A3B")

# Override records: how long a human's "yes, push it anyway" is good for
# before the same blocking findings need a fresh, re-justified override --
# this is deliberately not a permanent bypass.
OVERRIDE_TTL_MINUTES = 30
OVERRIDE_LOG_FILENAME = ".code_review_overrides.jsonl"

# Read from the environment, never hardcoded -- this script's own Layer 3
# check would (rightly) flag a literal key here.
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")

# A handful of common Simplified-only characters that have a different
# Traditional form -- enough to catch an accidental slip (this repo's own
# system prompt requires Traditional/Taiwan usage), not an exhaustive
# converter.
SIMPLIFIED_ONLY_CHARS = {
    "并": "並", "个": "個", "国": "國", "为": "為", "这": "這", "说": "說",
    "让": "讓", "还": "還", "没": "沒", "问": "問", "题": "題", "习": "習",
    "对": "對", "从": "從", "时": "時", "现": "現", "实": "實", "关": "關",
    "开": "開", "system": "system",  # placeholder removed below
}
del SIMPLIFIED_ONLY_CHARS["system"]

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{10,}"), "看起來像 API secret key (sk- 開頭)"),
    (re.compile(r"pk-[A-Za-z0-9-]{10,}"), "看起來像 API public key (pk- 開頭)"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "看起來像 AWS access key"),
    (re.compile(r"-----BEGIN (RSA|OPENSSH|EC|PGP|PRIVATE) KEY-----"), "內嵌私鑰區塊"),
    (re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9._-]{15,}"), "硬寫的 Bearer token"),
]
HARDCODED_ASSIGN_RE = re.compile(
    r"""(?i)\b(password|passwd|secret|api_key|apikey|token|access_key)\s*=\s*["']([^"']{4,})["']"""
)
INJECTION_PATTERNS = [
    (re.compile(r"\beval\s*\("), "使用 eval()"),
    (re.compile(r"\bexec\s*\("), "使用 exec()"),
    (re.compile(r"\bpickle\.(loads?|load)\s*\("), "使用 pickle.load(s)()，對不信任輸入不安全"),
    (re.compile(r"shell\s*=\s*True"), "subprocess 用 shell=True"),
    (re.compile(r"os\.system\s*\("), "使用 os.system()"),
]


def log(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def finding(level, layer, file, message, line=None):
    return {"level": level, "layer": layer, "file": file, "line": line, "message": message}


# ---------------------------------------------------------------------------
# Layer 1: syntax + imports
# ---------------------------------------------------------------------------

def _stdlib_modules():
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    # Fallback for < 3.10, not expected to be hit on this repo's 3.12.
    return {"os", "sys", "re", "json", "time", "argparse", "subprocess",
            "urllib", "ast", "typing", "collections", "itertools", "functools"}


def _requirements_packages(repo_root):
    path = os.path.join(repo_root, "requirements.txt")
    if not os.path.isfile(path):
        return set()
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[<>=\[!~]", line, maxsplit=1)[0].strip()
            if name:
                names.add(name.lower().replace("-", "_"))
    return names


def _is_os_path_attr(node, attr):
    """Matches the `os.path.<attr>` attribute chain."""
    return (isinstance(node, ast.Attribute) and node.attr == attr
            and isinstance(node.value, ast.Attribute) and node.value.attr == "path"
            and isinstance(node.value.value, ast.Name) and node.value.value.id == "os")


def _eval_path_expr(node, file_dir):
    """
    Evaluates just enough of an os.path.{join,dirname,abspath} expression
    tree to resolve this repo's own sys.path.insert/append idioms (e.g.
    `os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")`)
    to a real directory, relative to file_dir (the directory of the file
    being reviewed). Returns None for anything else -- this is deliberately
    narrow, not a general expression evaluator.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if _is_os_path_attr(node.func, "join") and node.args:
            parts = [_eval_path_expr(a, file_dir) for a in node.args]
            if all(p is not None for p in parts):
                return os.path.join(*parts)
            return None
        if _is_os_path_attr(node.func, "dirname") and node.args:
            inner = _eval_path_expr(node.args[0], file_dir)
            return os.path.dirname(inner) if inner is not None else None
        if _is_os_path_attr(node.func, "abspath") and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == "__file__":
                # A stand-in path inside file_dir -- callers only ever take
                # dirname() of this, so the fake basename never surfaces.
                return os.path.join(file_dir, "__file__")
            return _eval_path_expr(arg, file_dir)
    return None


def _sys_path_extra_dirs(tree, file_dir):
    """
    Directories a file adds to sys.path at runtime via sys.path.insert/
    append -- e.g. tests/test_report_helpers.py adds scripts/ this way to
    import generate_report. Without this, the import-declared check has no
    way to know that import is local, and flags it as an undeclared
    dependency every time.
    """
    extra_dirs = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("insert", "append"):
            continue
        obj = node.func.value
        if not (isinstance(obj, ast.Attribute) and obj.attr == "path"
                and isinstance(obj.value, ast.Name) and obj.value.id == "sys"):
            continue
        if not node.args:
            continue
        resolved = _eval_path_expr(node.args[-1], file_dir)
        if resolved:
            extra_dirs.append(os.path.normpath(resolved))
    return extra_dirs


def check_syntax_and_imports(path, source, repo_root):
    findings = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        findings.append(finding("blocking", "syntax", path,
                                 f"語法錯誤，無法解析：{e.msg}", line=e.lineno))
        return findings

    stdlib = _stdlib_modules()
    declared = _requirements_packages(repo_root)
    local_dir = os.path.dirname(os.path.join(repo_root, path))
    search_dirs = [local_dir] + _sys_path_extra_dirs(tree, local_dir)

    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [(alias.name.split(".")[0], node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import, assumed local
            if node.module:
                modules = [(node.module.split(".")[0], node.lineno)]
        for mod, lineno in modules:
            if mod in stdlib:
                continue
            if mod.lower().replace("-", "_") in declared:
                continue
            if any(os.path.isfile(os.path.join(d, f"{mod}.py")) for d in search_dirs):
                continue
            findings.append(finding(
                "suggestion", "syntax", path,
                f"import '{mod}' 不是標準函式庫、沒列在 requirements.txt、"
                f"也不是本地模組，確認是不是漏了宣告依賴", line=lineno,
            ))
    return findings


# ---------------------------------------------------------------------------
# Layer 2: style consistency against baseline files
# ---------------------------------------------------------------------------

def _has_usage_block(source):
    return bool(re.search(r'"""[\s\S]*?Usage:', source))


def _snake_case_violations(tree, path):
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                continue
            if re.search(r"[A-Z]", name) and "_" not in name:
                findings.append(finding(
                    "suggestion", "style", path,
                    f"函式名稱 '{name}' 看起來是 camelCase，這個 repo 一律用 snake_case",
                    line=node.lineno,
                ))
    return findings


def _detection_rule_line_ranges(tree):
    """
    Line numbers spanned by this file's own detection-rule constants
    (SIMPLIFIED_ONLY_CHARS, SECRET_PATTERNS, ...). Building a pattern that
    matches a given dangerous call or character means writing that exact
    substring into the pattern itself, so scanning these definitions with
    their own rules produces a self-referential false positive on this
    file specifically -- skip just the lines those constants span.
    """
    rule_names = {
        "SIMPLIFIED_ONLY_CHARS", "SECRET_PATTERNS",
        "HARDCODED_ASSIGN_RE", "INJECTION_PATTERNS",
    }
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in rule_names:
                end = getattr(node, "end_lineno", node.lineno)
                skip.update(range(node.lineno, end + 1))
    return skip


def check_style(path, source, tree, baseline_sources, skip_lines=frozenset()):
    findings = []

    is_script = 'if __name__ == "__main__"' in source
    baseline_all_have_usage = baseline_sources and all(
        _has_usage_block(b) for b in baseline_sources.values()
    )
    if is_script and baseline_all_have_usage and not _has_usage_block(source):
        findings.append(finding(
            "suggestion", "style", path,
            "這是一支可執行腳本，但沒有 module docstring 裡的 Usage: 範例區塊——"
            "兩支基準檔案 (rag_answer.py / n8n_qa_test_harness.py) 都有這個慣例",
        ))

    findings.extend(_snake_case_violations(tree, path))

    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in skip_lines:
            continue
        for simp, trad in SIMPLIFIED_ONLY_CHARS.items():
            if simp in line:
                findings.append(finding(
                    "suggestion", "style", path,
                    f"字元「{simp}」是簡體字，這個 repo 的使用者可見文字要用繁體"
                    f"（可能要用「{trad}」）", line=lineno,
                ))

    return findings


# ---------------------------------------------------------------------------
# Layer 3: security
# ---------------------------------------------------------------------------

def check_security(path, source, skip_lines=frozenset()):
    findings = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, start=1):
        if lineno in skip_lines:
            continue

        # 一行同時符合「像密鑰的格式」跟「變數名+字面字串指派」時只報一次
        # ——兩條規則本來就常常一起命中同一個硬寫密鑰，之前會各報一次，
        # 讓報告看起來像兩個問題,其實是同一行同一件事。
        secret_matched = False
        for pattern, desc in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(finding(
                    "blocking", "security", path,
                    f"{desc}，看起來是硬寫在程式碼裡的密鑰/憑證", line=lineno,
                ))
                secret_matched = True
                break

        if not secret_matched:
            m = HARDCODED_ASSIGN_RE.search(line)
            if m:
                var, value = m.group(1), m.group(2)
                findings.append(finding(
                    "blocking", "security", path,
                    f"變數 '{var}' 直接指派了字面字串值 '{value[:6]}...'，"
                    f"憑證/密鑰不該寫死在程式碼裡，應該讀環境變數", line=lineno,
                ))

        for pattern, desc in INJECTION_PATTERNS:
            if pattern.search(line):
                findings.append(finding(
                    "blocking", "security", path,
                    f"{desc}，確認輸入來源是否可信、有沒有注入風險", line=lineno,
                ))

    return findings


# ---------------------------------------------------------------------------
# Layer 4: spec alignment (the one layer that calls the local LLM)
# ---------------------------------------------------------------------------

SPEC_SYSTEM_PROMPT = (
    "你是嚴格的程式碼審查員,專門檢查「實作是否真的符合規格文件描述的行為」,"
    "不是檢查語法或風格。你會看到一份規格文件跟對應的程式碼。請:\n"
    "1. 從規格文件裡拆出具體、可驗證的行為陳述(不是標題或說明性文字,是「系統應該做"
    "什麼」這種可以對照程式碼驗證的句子)\n"
    "2. 針對每一條,去程式碼裡找對應邏輯,判斷是「satisfied」(程式碼確實這樣做)、"
    "「violated」(程式碼沒有這樣做,或做了相反的事)、還是「unclear」(規格陳述在"
    "程式碼裡找不到明確對應,無法判斷)\n"
    "3. 特別注意:程式碼看起來能執行、測試會過,不代表邏輯符合規格——這是最常見的"
    "失效模式,只看程式碼能不能跑是不夠的\n"
    "只能輸出一個 JSON 物件,格式為:\n"
    '{"claims": [{"claim": "<規格陳述原句或摘要>", "status": '
    '"satisfied|violated|unclear", "evidence": "<20-60字的程式碼對應說明>"}]}\n'
    "不要有其他文字、不要用 markdown code fence。"
)


def call_chat(messages, enable_thinking=True, max_tokens=1500, timeout=180):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        CHAT_URL, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        return None, f"呼叫 vLLM 失敗: {e}"
    try:
        content = body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None, f"vLLM 回應格式不如預期: {body}"
    return content, None


def _strip_think(text):
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text):
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def check_spec_alignment(path, source, spec_text):
    user_content = (
        f"規格文件:\n{spec_text}\n\n---\n\n"
        f"程式碼 ({path}):\n```python\n{source}\n```"
    )
    raw, err = call_chat(
        [
            {"role": "system", "content": SPEC_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        enable_thinking=True,
    )
    if err:
        return [finding("suggestion", "spec", path, f"規格比對失敗，跳過這層：{err}")], []

    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        claims = parsed.get("claims", [])
    except (json.JSONDecodeError, AttributeError):
        return [finding("suggestion", "spec", path,
                         f"規格比對的模型輸出無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")], []

    findings = []
    for c in claims:
        status = c.get("status")
        if status == "violated":
            findings.append(finding(
                "blocking", "spec", path,
                f"規格陳述「{c.get('claim', '?')}」— 程式碼不符：{c.get('evidence', '')}",
            ))
    return findings, claims


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def review_file(path, source, repo_root, baseline_sources, spec_text):
    all_findings = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return [finding("blocking", "syntax", path,
                         f"語法錯誤，無法解析：{e.msg}", line=e.lineno)], []

    skip_lines = _detection_rule_line_ranges(tree)
    all_findings.extend(check_syntax_and_imports(path, source, repo_root))
    all_findings.extend(check_style(path, source, tree, baseline_sources, skip_lines))
    all_findings.extend(check_security(path, source, skip_lines))

    claims = []
    if spec_text:
        spec_findings, claims = check_spec_alignment(path, source, spec_text)
        all_findings.extend(spec_findings)

    return all_findings, claims


def git_show(repo_root, ref, path=None):
    target = f"{ref}:{path}" if path else ref
    result = subprocess.run(
        ["git", "show", target], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        die(f"git show {target} 失敗: {result.stderr.strip()}")
    return result.stdout


def changed_py_files(repo_root, ref):
    # Plain `git diff-tree` shows nothing for a merge commit unless told how
    # to handle multiple parents -- diff against the first parent explicitly
    # so merge commits (e.g. a branch merge) are reviewed like any other.
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}^1", ref],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", ref],
            cwd=repo_root, capture_output=True, text=True,
        )
        if result.returncode != 0:
            die(f"git diff 失敗: {result.stderr.strip()}")
    return [p for p in result.stdout.splitlines() if p.endswith(".py")]


def parse_range(range_str):
    if ".." not in range_str:
        die(f"--range 需要 OLD..NEW 格式，收到：{range_str}")
    old, _, new = range_str.partition("..")
    if not old or not new:
        die(f"--range 需要 OLD..NEW 格式，收到：{range_str}")
    return old, new


def changed_py_files_range(repo_root, old, new):
    result = subprocess.run(
        ["git", "diff", "--name-only", old, new], cwd=repo_root, capture_output=True, text=True
    )
    if result.returncode != 0:
        die(f"git diff {old}..{new} 失敗: {result.stderr.strip()}")
    return [p for p in result.stdout.splitlines() if p.endswith(".py")]


def apply_only_file_filter(targets, only_files):
    """Restricts targets to the given paths -- e.g. a merge commit range
    that pulled in unrelated files from the other side of the merge, and
    the reviewer only wants to look at the ones they actually touched."""
    if not only_files:
        return targets
    only_set = set(only_files)
    return [(p, s) for p, s in targets if p in only_set]


def gather_range_targets(repo_root, range_str, only_files=None):
    old, new = parse_range(range_str)
    py_files = changed_py_files_range(repo_root, old, new)
    targets = [(p, git_show(repo_root, new, p)) for p in py_files]
    return apply_only_file_filter(targets, only_files)


def review_targets(repo_root, targets, baseline_sources, spec_text):
    all_findings_by_file = {}
    all_claims_by_file = {}
    for path, source in targets:
        log(f"審查 {path} ...")
        findings, claims = review_file(path, source, repo_root, baseline_sources, spec_text)
        all_findings_by_file[path] = findings
        all_claims_by_file[path] = claims
    return all_findings_by_file, all_claims_by_file


def blocking_findings_of(all_findings_by_file):
    return [f for fs in all_findings_by_file.values() for f in fs if f["level"] == "blocking"]


# ---------------------------------------------------------------------------
# Override records: a blocking finding can only be pushed through by a human
# who runs --override with their name and a reason. The record is scoped to
# the exact set of blocking findings it was issued for (via a hash) and
# expires -- it is not a standing bypass. Kept both locally (fast, no
# network needed to check) and in Langfuse (append-only, matches the
# delivery-center approval flow's "tamper-evident external record"
# principle) under their own trace name/tag/score name so they never mix
# with the delivery-center's pending-review/human_decision records.
# ---------------------------------------------------------------------------

def compute_blocking_signature(range_str, blocking_findings):
    material = range_str + "|" + "\n".join(
        sorted(f"{f['file']}:{f.get('line')}:{f['message']}" for f in blocking_findings)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def override_log_path(repo_root):
    return os.path.join(repo_root, OVERRIDE_LOG_FILENAME)


def read_valid_override(repo_root, token):
    path = override_log_path(repo_root)
    if not os.path.isfile(path):
        return None
    now = datetime.now(timezone.utc)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("token") == token:
                try:
                    expires_at = datetime.fromisoformat(rec["expires_at"])
                except (KeyError, ValueError):
                    continue
                if now <= expires_at:
                    return rec
    return None


def append_override_record(repo_root, record):
    with open(override_log_path(repo_root), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _post_json(url, payload, headers=None, timeout=20):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return None, str(e.reason).encode("utf-8")


def send_override_to_langfuse(range_str, approver, reason, blocking_findings, token, expires_at_iso):
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        log("沒有設定 LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY，略過寫入 Langfuse（本機紀錄仍會寫入）。")
        return False

    auth = "Basic " + base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode("utf-8")
    ).decode("ascii")
    headers = {"Authorization": auth, "x-langfuse-ingestion-version": "4"}

    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    now_ns = f"{int(time.time() * 1000)}000000"

    trace_body = {
        "resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "code-review-agent"}}]},
            "scopeSpans": [{
                "scope": {"name": "code-review-agent"},
                "spans": [{
                    "traceId": trace_id,
                    "spanId": span_id,
                    "name": "code_review_override",
                    "kind": 1,
                    "startTimeUnixNano": now_ns,
                    "endTimeUnixNano": now_ns,
                    "attributes": [
                        {"key": "langfuse.trace.name", "value": {"stringValue": "code-review-override"}},
                        {"key": "langfuse.trace.tags",
                         "value": {"arrayValue": {"values": [{"stringValue": "code-review-override"}]}}},
                        {"key": "langfuse.observation.metadata.approver", "value": {"stringValue": approver}},
                        {"key": "langfuse.observation.metadata.reason", "value": {"stringValue": reason}},
                        {"key": "langfuse.observation.metadata.range", "value": {"stringValue": range_str}},
                        {"key": "langfuse.observation.metadata.blocking_count",
                         "value": {"intValue": len(blocking_findings)}},
                        {"key": "langfuse.observation.metadata.token", "value": {"stringValue": token}},
                        {"key": "langfuse.observation.metadata.expires_at", "value": {"stringValue": expires_at_iso}},
                    ],
                    "status": {"code": 0},
                }],
            }],
        }],
    }
    status, _ = _post_json(f"{LANGFUSE_BASE_URL}/api/public/otel/v1/traces", trace_body, headers=headers)
    if status != 200:
        log(f"寫入 Langfuse trace 失敗 (status={status})，本機紀錄仍然有效。")
        return False

    score_body = {
        "batch": [{
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "score-create",
            "body": {
                "id": str(uuid.uuid4()),
                "traceId": trace_id,
                "name": "code_review_override",
                "value": 1,
                "dataType": "BOOLEAN",
                "comment": reason,
                "metadata": {"approver": approver, "range": range_str,
                             "blocking_count": len(blocking_findings)},
            },
        }],
    }
    _post_json(f"{LANGFUSE_BASE_URL}/api/public/ingestion", score_body, headers=headers)
    return True


def handle_range(args, repo_root, baseline_sources, spec_text):
    targets = gather_range_targets(repo_root, args.range, args.only_file)
    if not targets:
        if args.only_file:
            log(f"{args.range} 之間，--only-file 指定的檔案沒有一個有變更，放行。")
        else:
            log(f"{args.range} 之間沒有變更任何 .py 檔案，放行。")
        sys.exit(0)

    all_findings_by_file, all_claims_by_file = review_targets(repo_root, targets, baseline_sources, spec_text)
    clean = print_report(all_findings_by_file, all_claims_by_file, spec_given=bool(spec_text))
    if clean:
        sys.exit(0)

    blocking = blocking_findings_of(all_findings_by_file)
    token = compute_blocking_signature(args.range, blocking)
    override = read_valid_override(repo_root, token)
    if override:
        print()
        print("## Override 生效")
        print(f"審核人：{override['approver']}（{override['created_at']}）")
        print(f"理由：{override['reason']}")
        print(f"有效期限至 {override['expires_at']}，上面列出的阻斷級問題已經人工看過，放行。")
        sys.exit(0)

    print()
    print("## 被擋下")
    print(f"發現 {len(blocking)} 項阻斷級問題，需要人工 override 才能繼續。執行：")
    print(f'  python3 scripts/code_review_agent.py --override --range "{args.range}" \\')
    print('    --approver "你的名字" --reason "為什麼即使有這些警告還是要推送"')
    sys.exit(1)


def handle_override(args, repo_root, baseline_sources, spec_text):
    if not args.range:
        die("--override 需要搭配 --range 使用。")
    if not args.approver or not args.reason:
        die("--override 需要 --approver 跟 --reason，兩個都要填。")

    targets = gather_range_targets(repo_root, args.range, args.only_file)
    if not targets:
        log(f"{args.range} 之間沒有變更任何 .py 檔案，沒有東西需要 override。")
        sys.exit(0)

    all_findings_by_file, _ = review_targets(repo_root, targets, baseline_sources, spec_text)
    blocking = blocking_findings_of(all_findings_by_file)
    if not blocking:
        log("重新跑過一次審查，這個範圍現在沒有阻斷級問題，不需要 override。")
        sys.exit(0)

    token = compute_blocking_signature(args.range, blocking)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=OVERRIDE_TTL_MINUTES)
    record = {
        "token": token,
        "range": args.range,
        "approver": args.approver,
        "reason": args.reason,
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "blocking_count": len(blocking),
        "blocking_messages": [
            f"[{f['layer']}] {f['file']}:{f.get('line')} — {f['message']}" for f in blocking
        ],
    }
    append_override_record(repo_root, record)
    sent = send_override_to_langfuse(
        args.range, args.approver, args.reason, blocking, token, expires_at.isoformat()
    )

    print(f"已記錄 override：{len(blocking)} 項阻斷級問題，審核人 {args.approver}")
    print(f"理由：{args.reason}")
    print(f"Token: {token}，有效期限至 {expires_at.isoformat()}（{OVERRIDE_TTL_MINUTES} 分鐘）")
    print(f"本機紀錄：{override_log_path(repo_root)}")
    print(f"Langfuse：{'已寫入' if sent else '略過(沒有設定金鑰，或寫入失敗)'}")
    print()
    print("現在可以重新執行 git push。")
    sys.exit(0)


def load_baseline_sources(repo_root, baseline_paths):
    sources = {}
    for p in baseline_paths:
        full = os.path.join(repo_root, p)
        if os.path.isfile(full):
            with open(full, "r", encoding="utf-8") as f:
                sources[p] = f.read()
        else:
            log(f"風格基準檔案不存在，略過：{p}")
    return sources


def print_report(all_findings_by_file, all_claims_by_file, spec_given):
    blocking = [f for fs in all_findings_by_file.values() for f in fs if f["level"] == "blocking"]
    suggestions = [f for fs in all_findings_by_file.values() for f in fs if f["level"] == "suggestion"]

    print("## 摘要")
    if blocking:
        print(f"發現 {len(blocking)} 項阻斷級問題、{len(suggestions)} 項建議級問題，需要處理阻斷級問題。")
    elif suggestions:
        print(f"沒有阻斷級問題，{len(suggestions)} 項建議級問題可以參考。")
    else:
        print("沒有發現任何問題。")
    print()

    print("## 阻斷級問題")
    if not blocking:
        print("（無）")
    else:
        for f in blocking:
            loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
            print(f"- [{f['layer']}] {loc} — {f['message']}")
    print()

    print("## 建議級問題")
    if not suggestions:
        print("（無）")
    else:
        for f in suggestions:
            loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
            print(f"- [{f['layer']}] {loc} — {f['message']}")
    print()

    print("## 規格對齊評估")
    if not spec_given:
        print("未提供規格文件，略過。")
    else:
        any_claims = False
        for path, claims in all_claims_by_file.items():
            if not claims:
                continue
            any_claims = True
            print(f"\n{path}:")
            for c in claims:
                mark = {"satisfied": "✓", "violated": "✗", "unclear": "?"}.get(c.get("status"), "?")
                print(f"  [{mark}] {c.get('claim', '?')} — {c.get('evidence', '')}")
        if not any_claims:
            print("模型沒有輸出可解析的規格陳述比對結果。")
    print()

    print("## 風格備註")
    style_notes = [f for fs in all_findings_by_file.values() for f in fs if f["layer"] == "style"]
    if not style_notes:
        print("（無）")
    else:
        for f in style_notes:
            loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
            print(f"- {loc} — {f['message']}")

    return len(blocking) == 0


def main():
    ap = argparse.ArgumentParser(description="手動觸發的 .py 檔案程式碼審查，只出報告不核准/阻擋。")
    ap.add_argument("--files", nargs="*", default=[], help="要審查的檔案路徑（相對於 repo 根目錄）")
    ap.add_argument("--commit", default=None, help="審查這個 commit 裡變更的 .py 檔案（用 after 狀態的完整檔案內容）")
    ap.add_argument("--range", default=None,
                     help="審查 OLD..NEW 之間變更的 .py 檔案；有阻斷級問題就 exit 1，供 pre-push hook 使用")
    ap.add_argument("--override", action="store_true",
                     help="對 --range 目前的阻斷級問題留下人工核准紀錄（需要 --approver 跟 --reason）")
    ap.add_argument("--approver", default=None, help="--override 用：審核人名字")
    ap.add_argument("--reason", default=None, help="--override 用：為什麼即使有阻斷級問題還是要放行")
    ap.add_argument("--only-file", action="append", default=None,
                     help="只審查這些檔案，即使 --commit/--range 的範圍裡改了更多"
                          "（例如 merge commit 會拉進其他不相關的變更）；可重複指定")
    ap.add_argument("--spec", default=None, help="規格文件路徑，提供時才會跑第4層規格對齊比對")
    ap.add_argument("--baseline", nargs="*", default=DEFAULT_BASELINE,
                     help="風格基準檔案，預設是 rag_answer.py 跟 n8n_qa_test_harness.py")
    ap.add_argument("--repo-root", default=os.getcwd(), help="repo 根目錄（預設當前目錄）")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    baseline_sources = load_baseline_sources(repo_root, args.baseline)
    spec_text = None
    if args.spec:
        spec_full = args.spec if os.path.isabs(args.spec) else os.path.join(repo_root, args.spec)
        if not os.path.isfile(spec_full):
            die(f"規格文件不存在：{args.spec}")
        with open(spec_full, "r", encoding="utf-8") as f:
            spec_text = f.read()

    if args.override:
        handle_override(args, repo_root, baseline_sources, spec_text)
        return
    if args.range:
        handle_range(args, repo_root, baseline_sources, spec_text)
        return

    targets = []  # list of (relative_path, source)
    if args.commit:
        for p in changed_py_files(repo_root, args.commit):
            source = git_show(repo_root, args.commit, p)
            targets.append((p, source))
        targets = apply_only_file_filter(targets, args.only_file)
        if not targets:
            log(f"commit {args.commit} 裡沒有變更任何 .py 檔案（或都被 --only-file 篩掉了）。")
    for p in args.files:
        full = p if os.path.isabs(p) else os.path.join(repo_root, p)
        if not os.path.isfile(full):
            die(f"檔案不存在：{p}")
        with open(full, "r", encoding="utf-8") as f:
            rel = os.path.relpath(full, repo_root)
            targets.append((rel, f.read()))

    if not targets:
        die("沒有要審查的檔案，用 --files、--commit 或 --range 指定。")

    all_findings_by_file, all_claims_by_file = review_targets(
        repo_root, targets, baseline_sources, spec_text
    )

    clean = print_report(all_findings_by_file, all_claims_by_file, spec_given=bool(spec_text))
    # Exit code reflects presence of blocking findings for scripting convenience,
    # but this is informational only -- nothing upstream should treat it as
    # an approve/reject gate.
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
