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
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_BASELINE = ["scripts/rag_answer.py", "scripts/n8n_qa_test_harness.py"]
CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8000/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-30B-A3B")

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
            if os.path.isfile(os.path.join(local_dir, f"{mod}.py")):
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


def check_style(path, source, tree, baseline_sources):
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

def check_security(path, source):
    findings = []
    lines = source.splitlines()

    for lineno, line in enumerate(lines, start=1):
        for pattern, desc in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append(finding(
                    "blocking", "security", path,
                    f"{desc}，看起來是硬寫在程式碼裡的密鑰/憑證", line=lineno,
                ))

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

    all_findings.extend(check_syntax_and_imports(path, source, repo_root))
    all_findings.extend(check_style(path, source, tree, baseline_sources))
    all_findings.extend(check_security(path, source))

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
    ap.add_argument("--spec", default=None, help="規格文件路徑，提供時才會跑第4層規格對齊比對")
    ap.add_argument("--baseline", nargs="*", default=DEFAULT_BASELINE,
                     help="風格基準檔案，預設是 rag_answer.py 跟 n8n_qa_test_harness.py")
    ap.add_argument("--repo-root", default=os.getcwd(), help="repo 根目錄（預設當前目錄）")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    targets = []  # list of (relative_path, source)
    if args.commit:
        for p in changed_py_files(repo_root, args.commit):
            source = git_show(repo_root, args.commit, p)
            targets.append((p, source))
        if not targets:
            log(f"commit {args.commit} 裡沒有變更任何 .py 檔案。")
    for p in args.files:
        full = p if os.path.isabs(p) else os.path.join(repo_root, p)
        if not os.path.isfile(full):
            die(f"檔案不存在：{p}")
        with open(full, "r", encoding="utf-8") as f:
            rel = os.path.relpath(full, repo_root)
            targets.append((rel, f.read()))

    if not targets:
        die("沒有要審查的檔案，用 --files 或 --commit 指定。")

    spec_text = None
    if args.spec:
        spec_full = args.spec if os.path.isabs(args.spec) else os.path.join(repo_root, args.spec)
        if not os.path.isfile(spec_full):
            die(f"規格文件不存在：{args.spec}")
        with open(spec_full, "r", encoding="utf-8") as f:
            spec_text = f.read()

    baseline_sources = load_baseline_sources(repo_root, args.baseline)

    all_findings_by_file = {}
    all_claims_by_file = {}
    for path, source in targets:
        log(f"審查 {path} ...")
        findings, claims = review_file(path, source, repo_root, baseline_sources, spec_text)
        all_findings_by_file[path] = findings
        all_claims_by_file[path] = claims

    clean = print_report(all_findings_by_file, all_claims_by_file, spec_given=bool(spec_text))
    # Exit code reflects presence of blocking findings for scripting convenience,
    # but this is informational only -- nothing upstream should treat it as
    # an approve/reject gate.
    sys.exit(0 if clean else 1)


if __name__ == "__main__":
    main()
