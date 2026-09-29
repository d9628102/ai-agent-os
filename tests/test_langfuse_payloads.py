"""
測試重點：Langfuse 只收「執行資料」與治理紀錄的最小欄位（2026-09 決議）。
問題、文件內容、檢索片段、回答內容、評審理由、核准備註、override 理由一律不送。

- build_override_trace_body()／build_override_score_body()／send_override_to_langfuse()：
  override 只送決定、核准人、時間；用金絲雀字串走完整的 handle_override()，確認
  reason 與 blocking 訊息完整寫進本機紀錄、但送給 Langfuse 的請求裡一個字都沒有
- n8n/workflows/*.json（repo 內的流程匯出）：audit_workflow() 逐項檢查
  · 頂層只能有可重現流程的欄位（不能有 shared——裡面有使用者 email）
  · 憑證只能是 {id, name}，內容不能有金鑰／token／密碼
  · 打 Langfuse 的 HTTP 節點：網址、body 來源、header 都要在白名單內
  · 組 Langfuse 內容的 Code 節點：屬性 key 必須在白名單、值與 comment／metadata
    不能引用問題／回答／備註／理由等變數
- 用改壞的流程副本確認 audit_workflow() 真的抓得到每一種洩漏

這是靜態檢查：抓得到「寫進程式碼的欄位」，抓不到執行時才組出來的內容。
執行時的實際內容由離線金絲雀驗證（用真實節點程式碼＋金絲雀字串）另外確認。
"""

import copy
import json
import os
import re
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import code_review_agent as cra
from code_review_agent import (
    build_override_score_body,
    build_override_trace_body,
    handle_override,
    override_log_path,
    send_override_to_langfuse,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_DIR = os.path.join(ROOT, "n8n", "workflows")
NOW = datetime(2026, 9, 29, 4, 0, 0, tzinfo=timezone.utc)

CANARY_REASON = "CANARY_REASON_7f3a91 為什麼要放行"
CANARY_MESSAGE = "CANARY_FINDING_7f3a91 hardcoded secret"
CANARY_RANGE = "aaaa111..bbbb222"


# ---------------------------------------------------------------------------
# override：只送決定、核准人、時間
# ---------------------------------------------------------------------------

def _trace_attr_keys(body):
    return [a["key"] for a in body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]]


def test_override_trace_only_has_decision_approver_time():
    body = build_override_trace_body("審核人", "t" * 32, "s" * 16, NOW)
    assert sorted(_trace_attr_keys(body)) == sorted([
        "langfuse.trace.name", "langfuse.trace.tags",
        "langfuse.observation.metadata.approver", "langfuse.observation.metadata.decided_at",
    ])
    attrs = {a["key"]: a["value"] for a in body["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]}
    assert attrs["langfuse.observation.metadata.approver"] == {"stringValue": "審核人"}
    assert attrs["langfuse.observation.metadata.decided_at"] == {"stringValue": NOW.isoformat()}


def test_override_score_has_no_comment_and_minimal_metadata():
    body = build_override_score_body("審核人", "t" * 32, "e" * 36, "c" * 36, NOW)
    event = body["batch"][0]
    assert event["timestamp"] == NOW.isoformat()
    score = event["body"]
    assert score["comment"] is None
    assert score["value"] == 1 and score["name"] == "code_review_override"
    assert score["traceId"] == "t" * 32
    assert score["metadata"] == {"approver": "審核人", "decided_at": NOW.isoformat()}


def test_override_builders_cannot_be_given_free_text():
    # 函式簽名刻意不接受 reason／range／token 等參數：想多送欄位就得改簽名，這條測試會擋下
    import inspect
    assert list(inspect.signature(send_override_to_langfuse).parameters) == ["approver"]
    assert list(inspect.signature(build_override_trace_body).parameters) == ["approver", "trace_id", "span_id", "now"]
    assert list(inspect.signature(build_override_score_body).parameters) == [
        "approver", "trace_id", "event_id", "score_id", "now"]


def test_send_override_skips_without_keys(monkeypatch):
    monkeypatch.setattr(cra, "LANGFUSE_PUBLIC_KEY", None)
    monkeypatch.setattr(cra, "LANGFUSE_SECRET_KEY", None)
    posted = []
    monkeypatch.setattr(cra, "_post_json", lambda *a, **k: posted.append(a) or (200, b""))
    assert send_override_to_langfuse("審核人") is False
    assert posted == []


def test_send_override_posts_two_requests_with_expected_urls(monkeypatch):
    monkeypatch.setattr(cra, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(cra, "LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(cra, "LANGFUSE_BASE_URL", "https://example.invalid")
    posted = []
    monkeypatch.setattr(cra, "_post_json", lambda url, payload, headers=None, **k: posted.append((url, payload)) or (200, b""))
    assert send_override_to_langfuse("審核人") is True
    assert [u for u, _ in posted] == [
        "https://example.invalid/api/public/otel/v1/traces", "https://example.invalid/api/public/ingestion"]
    # 兩個請求指向同一個 trace
    trace_id = posted[0][1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]
    assert posted[1][1]["batch"][0]["body"]["traceId"] == trace_id


def test_send_override_trace_failure_stops_and_returns_false(monkeypatch):
    monkeypatch.setattr(cra, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(cra, "LANGFUSE_SECRET_KEY", "sk-test")
    posted = []
    monkeypatch.setattr(cra, "_post_json", lambda url, payload, headers=None, **k: posted.append(url) or (500, b""))
    assert send_override_to_langfuse("審核人") is False
    assert len(posted) == 1


def test_handle_override_keeps_free_text_local_and_sends_none_of_it(monkeypatch, tmp_path, capsys):
    """走完整的 handle_override()：reason 與 blocking 訊息帶金絲雀。
    本機紀錄要完整保留（否則就不是「只存本機」而是「丟掉」），送給 Langfuse 的
    兩個請求裡不能出現金絲雀、git range、override token。"""
    monkeypatch.setattr(cra, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(cra, "LANGFUSE_SECRET_KEY", "sk-test")
    sent = []
    monkeypatch.setattr(cra, "_post_json", lambda url, payload, headers=None, **k: sent.append(payload) or (200, b""))
    blocking = [{"layer": "L1", "file": "scripts/x.py", "line": 3, "message": CANARY_MESSAGE}]
    monkeypatch.setattr(cra, "gather_range_targets", lambda *a, **k: ["scripts/x.py"])
    monkeypatch.setattr(cra, "review_targets", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(cra, "blocking_findings_of", lambda *a, **k: blocking)

    args = SimpleNamespace(range=CANARY_RANGE, approver="審核人", reason=CANARY_REASON, only_file=None)
    with pytest.raises(SystemExit) as excinfo:
        handle_override(args, str(tmp_path), {}, None)
    assert excinfo.value.code == 0

    local = open(override_log_path(str(tmp_path)), encoding="utf-8").read()
    record = json.loads(local.strip().splitlines()[-1])
    assert record["reason"] == CANARY_REASON
    assert record["range"] == CANARY_RANGE
    assert any(CANARY_MESSAGE in m for m in record["blocking_messages"])

    assert len(sent) == 2
    wire = json.dumps(sent, ensure_ascii=False)
    for secret in (CANARY_REASON, "CANARY_REASON", CANARY_MESSAGE, "CANARY_FINDING", CANARY_RANGE, record["token"]):
        assert secret not in wire
    assert "審核人" in wire
    assert "Langfuse：已寫入" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# n8n 流程匯出：audit_workflow()
# ---------------------------------------------------------------------------

ALLOWED_TOP_LEVEL = {"id", "name", "active", "nodes", "connections", "settings", "meta", "pinData", "tags"}
ALLOWED_ATTR_KEYS = {
    "service.name", "langfuse.trace.name", "langfuse.trace.tags", "langfuse.observation.type",
    "langfuse.observation.model.name", "gen_ai.usage.output_tokens",
    "langfuse.observation.metadata.think_used", "langfuse.observation.metadata.total_time_s",
    "langfuse.observation.metadata.source", "langfuse.observation.metadata.script",
}
REQUIRED_RAG_ATTR_KEYS = {
    "langfuse.observation.model.name", "gen_ai.usage.output_tokens",
    "langfuse.observation.metadata.total_time_s", "langfuse.observation.metadata.script",
}
ALLOWED_BODY_EXPRESSIONS = {
    "={{ $json.langfuse_body }}", "={{ $json.qa_batch_body }}",
    "={{ $json.tag_body }}", "={{ $json.decision_body }}",
}
ALLOWED_HEADER_NAMES = {"x-langfuse-ingestion-version", "Content-Type"}
LANGFUSE_URL_PREFIX = "https://cloud.langfuse.com/api/public/"
FORBIDDEN_IDENTIFIERS = (
    "question", "answer", "note", "reason", "reasoning", "judge_reasoning", "stdout", "output",
    "context", "retrieved_context_full", "retrieved_headings", "prompt", "completion",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_IDENTIFIERS) + r")\b")
_STRING_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`(?:[^`\\]|\\.)*`")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_SECRET_PATTERNS = {
    "Langfuse 金鑰": re.compile(r"(?:sk|pk)-lf-[A-Za-z0-9_-]+"),
    "Bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{12,}"),
    "Basic 認證標頭": re.compile(r"Basic\s+[A-Za-z0-9+/=]{12,}"),
    "密碼欄位": re.compile(r"\"password\"\s*:\s*\"[^\"]+\""),
    "金鑰／token 欄位": re.compile(r"\"(?:api[_-]?key|secret|token)\"\s*:\s*\"[^\"]{6,}\""),
    "私鑰": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
}


def _strip_js_comments(code):
    return re.sub(r"(?m)^\s*//.*$", "", code)


def _strip_strings(expr):
    return _STRING_LITERAL_RE.sub('""', expr)


def _call_arguments(code, callee):
    """回傳 code 裡每一個 callee( ... ) 呼叫的頂層引數文字清單（略過字串內的括號）。"""
    calls = []
    start = 0
    while True:
        i = code.find(callee, start)
        if i < 0:
            return calls
        j = i + len(callee)
        depth, k, in_str, args, cur = 1, j, None, [], ""
        while k < len(code) and depth > 0:
            ch = code[k]
            if in_str:
                cur += ch
                if ch == "\\":
                    k += 1
                    cur += code[k] if k < len(code) else ""
                elif ch == in_str:
                    in_str = None
            elif ch in "'\"`":
                in_str = ch
                cur += ch
            elif ch in "([{":
                depth += 1
                cur += ch
            elif ch in ")]}":
                depth -= 1
                if depth > 0:
                    cur += ch
            elif ch == "," and depth == 1:
                args.append(cur)
                cur = ""
            else:
                cur += ch
            k += 1
        args.append(cur)
        calls.append(args)
        start = k


def _forbidden_in(expr):
    return sorted(set(_FORBIDDEN_RE.findall(_strip_strings(expr))))


def audit_workflow(wf):
    """回傳違規描述清單，空清單代表通過。純邏輯，不執行任何 JS。"""
    problems = []
    raw = json.dumps(wf, ensure_ascii=False)

    extra = sorted(set(wf) - ALLOWED_TOP_LEVEL)
    if extra:
        problems.append(f"頂層有不該放進 repo 的欄位：{extra}")
    for label, pattern in _SECRET_PATTERNS.items():
        if pattern.search(raw):
            problems.append(f"匯出檔裡出現疑似{label}")
    if _EMAIL_RE.search(raw):
        problems.append("匯出檔裡出現 email")

    for node in wf.get("nodes", []):
        name = node.get("name")
        for cred_type, cred in (node.get("credentials") or {}).items():
            if not isinstance(cred, dict) or set(cred) != {"id", "name"}:
                problems.append(f"{name}：憑證 {cred_type} 除了 id、name 之外不能有別的內容")
        params = node.get("parameters") or {}
        code = params.get("jsCode")
        url = params.get("url", "")

        if node.get("type") == "n8n-nodes-base.httpRequest" and "langfuse" in url:
            if not url.startswith(LANGFUSE_URL_PREFIX):
                problems.append(f"{name}：網址不在白名單（{url}）")
            if params.get("jsonBody") not in ALLOWED_BODY_EXPRESSIONS:
                problems.append(f"{name}：body 來源不在白名單（{params.get('jsonBody')!r}）")
            headers = {h.get("name") for h in (params.get("headerParameters") or {}).get("parameters", [])}
            if headers - ALLOWED_HEADER_NAMES:
                problems.append(f"{name}：header 不在白名單：{sorted(headers - ALLOWED_HEADER_NAMES)}")
            if params.get("sendQuery"):
                problems.append(f"{name}：不該帶 query 參數")

        if code and re.search(r"resourceSpans|langfuse|batch\s*:", code):
            body_code = _strip_js_comments(code)
            for key in re.findall(r"key:\s*\"([^\"]+)\"", body_code):
                if key not in ALLOWED_ATTR_KEYS:
                    problems.append(f"{name}：屬性 key 不在白名單：{key}")
            for line in body_code.splitlines():
                if re.search(r"key:\s*\"", line) and "value:" in line:
                    bad = _forbidden_in(line.split("value:", 1)[1])
                    if bad:
                        problems.append(f"{name}：屬性值引用了 {bad}：{line.strip()}")
            for expr in re.findall(r"comment:\s*([^\n]+)", body_code):
                bad = _forbidden_in(expr)
                if bad:
                    problems.append(f"{name}：comment 引用了 {bad}：{expr.strip()}")
            for args in _call_arguments(body_code, "scoreEvent("):
                if len(args) >= 4:
                    bad = _forbidden_in(args[3])
                    if bad:
                        problems.append(f"{name}：scoreEvent 的 comment 引用了 {bad}")
            for inner in re.findall(r"metadata:\s*\{([^}]*)\}", body_code):
                bad = _forbidden_in(inner)
                if bad:
                    problems.append(f"{name}：metadata 引用了 {bad}：{inner.strip()}")
    return problems


def _load_workflows():
    files = sorted(f for f in os.listdir(WORKFLOW_DIR) if f.endswith(".json"))
    return {f: json.load(open(os.path.join(WORKFLOW_DIR, f), encoding="utf-8")) for f in files}


WORKFLOWS = _load_workflows()
RAG = WORKFLOWS["psf-eim-rag-qa-form.json"]
APPROVAL = WORKFLOWS["psf-eim-qa-approval.json"]


def test_workflow_export_files_present():
    assert sorted(WORKFLOWS) == ["psf-eim-qa-approval.json", "psf-eim-rag-qa-form.json"]


@pytest.mark.parametrize("filename", sorted(WORKFLOWS))
def test_real_workflow_exports_pass_audit(filename):
    assert audit_workflow(WORKFLOWS[filename]) == []


def _node(wf, name):
    return next(n for n in wf["nodes"] if n["name"] == name)


def test_langfuse_bound_nodes_are_the_expected_ones():
    # 結構回歸：哪些節點會打 Langfuse 是被盤點過的，新增或少掉都要有人看
    def langfuse_nodes(wf):
        return sorted(n["name"] for n in wf["nodes"]
                      if n["type"] == "n8n-nodes-base.httpRequest" and "langfuse" in n["parameters"].get("url", ""))
    assert langfuse_nodes(RAG) == ["Log to Langfuse", "Send QA Scores to Langfuse", "Tag Pending Review"]
    assert langfuse_nodes(APPROVAL) == ["Send Decision to Langfuse"]


def test_rag_trace_still_sends_execution_data():
    code = _node(RAG, "Code in JavaScript")["parameters"]["jsCode"]
    keys = set(re.findall(r"key:\s*\"([^\"]+)\"", _strip_js_comments(code)))
    assert REQUIRED_RAG_ATTR_KEYS <= keys
    assert "status: { code: 1 }" in code


def _mutated(wf, node_name, old, new):
    copy_wf = copy.deepcopy(wf)
    node = _node(copy_wf, node_name)
    assert node["parameters"]["jsCode"].count(old) == 1, old
    node["parameters"]["jsCode"] = node["parameters"]["jsCode"].replace(old, new)
    return copy_wf


PROMPT_LINE = '                { key: "langfuse.observation.metadata.source", value: { stringValue: "n8n" } },\n'


@pytest.mark.parametrize("label,wf_factory", [
    ("重新加回 gen_ai.prompt", lambda: _mutated(
        RAG, "Code in JavaScript", PROMPT_LINE,
        PROMPT_LINE + '                { key: "gen_ai.prompt", value: { stringValue: question || "" } },\n')),
    ("重新加回 gen_ai.completion", lambda: _mutated(
        RAG, "Code in JavaScript", PROMPT_LINE,
        PROMPT_LINE + '                { key: "gen_ai.completion", value: { stringValue: answer } },\n')),
    ("白名單內的 key 但值換成回答", lambda: _mutated(
        RAG, "Code in JavaScript", 'value: { stringValue: "rag_answer.py" }', "value: { stringValue: answer }")),
    ("faithfulness 的 comment 帶評審理由", lambda: _mutated(
        RAG, "QA Parse & Threshold", "judge_parse_error ? 'judge 輸出無法解析' : null),\n    scoreEvent('relevance'",
        "reasoning || (judge_parse_error ? 'judge 輸出無法解析' : null)),\n    scoreEvent('relevance'")),
    ("scoreEvent 的 comment 帶回答", lambda: _mutated(
        RAG, "QA Parse & Threshold", "scoreEvent('relevance', relevance ?? 0, 'NUMERIC', null)",
        "scoreEvent('relevance', relevance ?? 0, 'NUMERIC', src.answer)")),
    ("核准紀錄帶 note", lambda: _mutated(
        APPROVAL, "Build Decision Score", "comment: null,", "comment: note || null,")),
    ("核准紀錄的 metadata 帶 note", lambda: _mutated(
        APPROVAL, "Build Decision Score", "metadata: { approver, decided_at, source: 'qa-approve-webhook' }",
        "metadata: { approver, decided_at, source: 'qa-approve-webhook', note }")),
    ("多出白名單以外的屬性 key", lambda: _mutated(
        RAG, "Code in JavaScript", PROMPT_LINE,
        PROMPT_LINE + '                { key: "langfuse.observation.input", value: { stringValue: "x" } },\n')),
])
def test_audit_catches_each_kind_of_leak(label, wf_factory):
    assert audit_workflow(wf_factory()), label


def test_audit_ignores_forbidden_words_inside_comments_and_string_literals():
    # 註解與字串常數裡出現「回答」「question」等字不算違規（例如固定的 needs_review 說明文字）
    wf = _mutated(RAG, "QA Parse & Threshold", "const qa_batch_body = {",
                  "// question answer reason note 只是註解\nconst qa_batch_body = {")
    assert audit_workflow(wf) == []


def test_audit_catches_unsafe_export_metadata_and_credentials():
    with_shared = copy.deepcopy(RAG)
    with_shared["shared"] = [{"project": {"name": "LIN SW <someone@example.com>"}}]
    problems = audit_workflow(with_shared)
    assert any("頂層" in p for p in problems) and any("email" in p for p in problems)

    leaky_cred = copy.deepcopy(RAG)
    _node(leaky_cred, "Log to Langfuse")["credentials"]["httpBasicAuth"]["data"] = "x"
    assert any("憑證" in p for p in audit_workflow(leaky_cred))

    with_key = copy.deepcopy(RAG)
    _node(with_key, "Log to Langfuse")["parameters"]["note"] = "pk-lf-" + "a" * 20
    assert any("金鑰" in p for p in audit_workflow(with_key))


def test_audit_catches_unexpected_langfuse_http_node_settings():
    wrong_url = copy.deepcopy(RAG)
    _node(wrong_url, "Log to Langfuse")["parameters"]["url"] = "https://langfuse.example.com/api/public/otel/v1/traces"
    assert any("網址" in p for p in audit_workflow(wrong_url))

    wrong_body = copy.deepcopy(RAG)
    _node(wrong_body, "Log to Langfuse")["parameters"]["jsonBody"] = "={{ $json }}"
    assert any("body" in p for p in audit_workflow(wrong_body))

    extra_header = copy.deepcopy(RAG)
    _node(extra_header, "Log to Langfuse")["parameters"]["headerParameters"]["parameters"].append(
        {"name": "x-question", "value": "={{ $json.question }}"})
    assert any("header" in p for p in audit_workflow(extra_header))


def test_call_arguments_handles_nested_parens_and_strings():
    args = _call_arguments("x = scoreEvent('a,b', f(1, 2), 'N', (c ? 'p)q' : null));", "scoreEvent(")
    assert args == [["'a,b'", " f(1, 2)", " 'N'", " (c ? 'p)q' : null)"]]
