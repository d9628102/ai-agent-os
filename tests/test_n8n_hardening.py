"""
測試重點（2026-09 決議：輸入 token、失敗記錄、指令注入修正，合併一次部署）：

1. 輸入 token：rag_answer.answer_one() 的回傳帶 prompt_tokens、耗時那行多「輸入 N tokens」；
   n8n 表單的 Code 節點（真實節點程式碼，用 node 實際執行）三條 regex 對新舊格式都抓得到，
   Langfuse 屬性 gen_ai.usage.input_tokens 是整數
2. 指令注入：表單問題以 base64 傳給 SSH 指令。用真實的 Encode Question 節點程式碼與真實的
   指令範本，把 $(echo CANARY)、雙引號、反引號、換行、單引號、開頭的 --、很長的文字組進
   shell，確認它們原封不動成為單一參數，沒有被執行
3. 錯誤記錄 workflow：Error Trigger 只送固定欄位；把帶金絲雀（含問題文字）的錯誤訊息與堆疊
   餵給真實節點程式碼，輸出裡不能有金絲雀；失敗節點名稱限定在已知節點
4. 兩個正式 workflow 都指到錯誤記錄 workflow

需要 node 來執行 n8n Code 節點的程式碼：PATH 上的 node，沒有就用正式 n8n 容器裡的 node
（docker exec）；兩者都沒有時這些測試會 skip 並說明原因（不會悄悄通過）。
"""

import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

import rag_answer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WF_DIR = os.path.join(ROOT, "n8n", "workflows")


def _wf(name):
    return json.load(open(os.path.join(WF_DIR, name), encoding="utf-8"))


FORM = _wf("psf-eim-rag-qa-form.json")
APPROVAL = _wf("psf-eim-qa-approval.json")
ERRLOG = _wf("psf-eim-error-logger.json")


def _node(wf, name):
    return next(n for n in wf["nodes"] if n["name"] == name)


def _js_runner():
    if shutil.which("node"):
        return ["node"]
    if shutil.which("docker"):
        probe = subprocess.run(["docker", "exec", "n8n", "node", "-v"], capture_output=True, text=True)
        if probe.returncode == 0:
            return ["docker", "exec", "-i", "n8n", "node"]
    return None


RUNNER = _js_runner()
needs_node = pytest.mark.skipif(RUNNER is None, reason="找不到 node（PATH 與 n8n 容器都沒有），無法執行 n8n Code 節點程式碼")

_HARNESS = """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const code = input.code;
const refs = input.refs || {};
const $input = { item: { json: input.json } };
const $ = (name) => ({ first: () => ({ json: refs[name] }), item: { json: refs[name] } });
const fn = new Function('$input', '$', code);
process.stdout.write(JSON.stringify(fn($input, $)));
"""


def run_node_code(node, item_json, refs=None):
    """用 n8n Code 節點的方式執行節點裡的 jsCode（$input.item.json 與 $('節點名').first().json）。"""
    payload = json.dumps({"code": node["parameters"]["jsCode"], "json": item_json, "refs": refs or {}})
    result = subprocess.run(RUNNER + ["-e", _HARNESS], input=payload, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# 1. 輸入 token
# ---------------------------------------------------------------------------

def _run_answer_one(monkeypatch, capsys, usage):
    hits = [{"score": 0.9, "payload": {"heading_path": "H > A", "text": "片段"}}]
    monkeypatch.setattr(rag_answer, "embed_one", lambda q: [0.0])
    monkeypatch.setattr(rag_answer, "search", lambda *a, **k: hits)
    monkeypatch.setattr(rag_answer, "build_context", lambda h: "CTX")
    monkeypatch.setattr(rag_answer, "generate", lambda *a, **k: ("這是回答", "", usage))
    result = rag_answer.answer_one("問題", "col", 8, 2048, 0.2, False, True, enable_thinking=False)
    return result, capsys.readouterr().out


def test_answer_one_returns_prompt_tokens_and_prints_them(monkeypatch, capsys):
    result, out = _run_answer_one(monkeypatch, capsys, {"prompt_tokens": 1234, "completion_tokens": 56})
    assert result["prompt_tokens"] == 1234 and result["completion_tokens"] == 56
    assert "輸入 1234 tokens, 輸出 56 tokens" in out


def test_answer_one_missing_prompt_tokens_is_none_not_an_error(monkeypatch, capsys):
    result, out = _run_answer_one(monkeypatch, capsys, {"completion_tokens": 56})
    assert result["prompt_tokens"] is None
    assert "輸入 ? tokens, 輸出 56 tokens" in out


def test_answer_one_keeps_all_previous_result_keys(monkeypatch, capsys):
    result, _ = _run_answer_one(monkeypatch, capsys, {"prompt_tokens": 1, "completion_tokens": 2})
    for key in ("question", "visible", "thinking_chars", "top_score", "top_heading", "total", "gen",
                "hits", "context", "completion_tokens", "think_used", "thinking"):
        assert key in result


@needs_node
def test_form_code_node_reads_new_format_input_output_and_time(monkeypatch, capsys):
    _, out = _run_answer_one(monkeypatch, capsys, {"prompt_tokens": 1234, "completion_tokens": 56})
    node = _node(FORM, "Code in JavaScript")
    res = run_node_code(node, {"stdout": out}, {"On form submission": {"你的問題": "問題"}})
    assert res["prompt_tokens"] == 1234 and res["completion_tokens"] == 56
    assert res["total_time_s"] is not None and res["answer"] == "這是回答"
    attrs = {a["key"]: a["value"] for a in
             res["langfuse_body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": 1234}
    assert attrs["gen_ai.usage.output_tokens"] == {"intValue": 56}


@needs_node
def test_form_code_node_still_reads_old_format_without_input_tokens():
    old = ("\n【回答】\n舊格式回答\n\n【耗時】embedding 0.1s | 檢索 0.1s | 生成 1.00s | 端到端 1.50s"
           "  (輸出 77 tokens, 約 77.0 tokens/s)\n")
    res = run_node_code(_node(FORM, "Code in JavaScript"), {"stdout": old}, {"On form submission": {"你的問題": "q"}})
    assert res["completion_tokens"] == 77 and res["prompt_tokens"] is None
    attrs = {a["key"]: a["value"] for a in
             res["langfuse_body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["attributes"]}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": 0}


# ---------------------------------------------------------------------------
# 2. 指令注入：問題以 base64 傳入，不經 shell 解析
# ---------------------------------------------------------------------------

def test_ssh_command_no_longer_interpolates_the_question():
    command = _node(FORM, "Execute a command")["parameters"]["command"]
    assert "你的問題" not in command
    assert command.count("{{") == 1 and "{{ $json.question_b64 }}" in command
    assert "base64 -d" in command and " -- " in command


def test_form_flow_goes_through_encode_node_before_ssh():
    conn = FORM["connections"]
    assert conn["On form submission"]["main"][0][0]["node"] == "Encode Question"
    assert conn["Encode Question"]["main"][0][0]["node"] == "Execute a command"


LONG_TEXT = ("很長的問題 " * 6000) + "$(echo LONG_CANARY)"
INJECTION_QUESTIONS = [
    "$(echo CANARY_SUBST)",
    "`echo CANARY_BACKTICK`",
    'a" ; echo CANARY_QUOTE ; echo "b',
    "' ; echo CANARY_SINGLE ; '",
    "第一行\n第二行 $(echo CANARY_NEWLINE)",
    "--help",
    "-x --show-think",
    "100% \\n \\\\ ${HOME} $HOME * ? [a-z] ~ ; | & > /tmp/CANARY_REDIRECT",
    "一般的中文問題：智財歸屬是誰？",
    "emoji 😀 與 全形＂引號＂",
    LONG_TEXT,
]

_STUB = """#!/usr/bin/env python3
import json, sys
json.dump(sys.argv[1:], sys.stdout, ensure_ascii=False)
"""


@needs_node
@pytest.mark.parametrize("question", INJECTION_QUESTIONS, ids=lambda q: q[:24].replace("\n", "\\n"))
def test_question_reaches_the_script_as_one_literal_argument(question, tmp_path):
    encoded = run_node_code(_node(FORM, "Encode Question"), {"你的問題": question})
    assert set(encoded) == {"question_b64"} and re.fullmatch(r"[A-Za-z0-9+/=]*", encoded["question_b64"])

    template = _node(FORM, "Execute a command")["parameters"]["command"]
    prefix = "=cd /home/psf01/Desktop/ai-agent-os && python3 scripts/rag_answer.py"
    assert template.startswith(prefix)
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB)
    command = (f"cd {tmp_path} && python3 {stub}" + template[len(prefix):]).replace(
        "{{ $json.question_b64 }}", encoded["question_b64"])
    assert "{{" not in command

    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["--show-context", "--", question.rstrip("\n")]
    assert result.stderr == ""
    # 沒有任何東西被執行：沒有被建立的檔案，輸出裡沒有「被執行」的痕跡
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stub.py"]
    assert not os.path.exists("/tmp/CANARY_REDIRECT")


@needs_node
def test_encode_node_handles_missing_and_non_string_question():
    node = _node(FORM, "Encode Question")
    assert run_node_code(node, {})["question_b64"] == ""
    assert run_node_code(node, {"你的問題": None})["question_b64"] == ""
    assert run_node_code(node, {"你的問題": 123})["question_b64"] == "MTIz"


# ---------------------------------------------------------------------------
# 3. 錯誤記錄 workflow
# ---------------------------------------------------------------------------

CANARY_QUESTION = "CANARY_QUESTION_9c1e 智財歸屬是誰"
ERROR_EVENT = {
    "execution": {
        "id": "4321",
        "url": "http://192.168.1.128:5678/workflow/x/executions/4321",
        "retryOf": None,
        "error": {
            "message": f'Command failed: python3 scripts/rag_answer.py "{CANARY_QUESTION}" --show-context',
            "stack": f"Error: {CANARY_QUESTION}\n    at SSH.execute",
            "description": CANARY_QUESTION,
        },
        "lastNodeExecuted": "Execute a command",
        "mode": "webhook",
    },
    "workflow": {"id": "XctHEK3cQdOp6WBd", "name": "My workflow"},
}


def _error_attrs(res):
    span = res["langfuse_body"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    return {a["key"]: a["value"] for a in span["attributes"]}, span


@needs_node
def test_error_logger_sends_only_the_fixed_fields():
    res = run_node_code(_node(ERRLOG, "Build Error Trace"), ERROR_EVENT)
    assert set(res) == {"langfuse_body"}
    attrs, span = _error_attrs(res)
    assert attrs["langfuse.observation.metadata.workflow_name"] == {"stringValue": "My workflow"}
    assert attrs["langfuse.observation.metadata.failed_node"] == {"stringValue": "Execute a command"}
    assert attrs["langfuse.observation.metadata.execution_id"] == {"stringValue": "4321"}
    assert attrs["langfuse.trace.name"] == {"stringValue": "psf-eim-workflow-error"}
    assert span["status"] == {"code": 2}
    assert "message" not in span["status"]


@needs_node
def test_error_logger_wire_body_contains_no_canary_from_error_or_stack():
    res = run_node_code(_node(ERRLOG, "Build Error Trace"), ERROR_EVENT)
    wire = json.dumps(res, ensure_ascii=False)
    for secret in ("CANARY", CANARY_QUESTION, "rag_answer.py", "Command failed", "SSH.execute", "192.168.1.128"):
        assert secret not in wire


@needs_node
@pytest.mark.parametrize("field,value,expected_key,expected", [
    ("lastNodeExecuted", f"Execute a command {CANARY_QUESTION}", "failed_node", "unknown"),
    ("lastNodeExecuted", None, "failed_node", "unknown"),
    ("lastNodeExecuted", "Encode Question", "failed_node", "Encode Question"),
    ("id", CANARY_QUESTION, "execution_id", "unknown"),
    ("id", "12 34", "execution_id", "unknown"),
    ("id", 77, "execution_id", "77"),
])
def test_error_logger_limits_node_name_and_execution_id_to_known_shapes(field, value, expected_key, expected):
    event = json.loads(json.dumps(ERROR_EVENT))
    event["execution"][field] = value
    res = run_node_code(_node(ERRLOG, "Build Error Trace"), event)
    attrs, _ = _error_attrs(res)
    assert attrs[f"langfuse.observation.metadata.{expected_key}"] == {"stringValue": expected}
    assert CANARY_QUESTION not in json.dumps(res, ensure_ascii=False)


@needs_node
def test_error_logger_unknown_workflow_name_becomes_unknown():
    event = json.loads(json.dumps(ERROR_EVENT))
    event["workflow"]["name"] = CANARY_QUESTION
    res = run_node_code(_node(ERRLOG, "Build Error Trace"), event)
    attrs, _ = _error_attrs(res)
    assert attrs["langfuse.observation.metadata.workflow_name"] == {"stringValue": "unknown"}
    assert CANARY_QUESTION not in json.dumps(res, ensure_ascii=False)


@needs_node
def test_error_logger_survives_missing_fields():
    res = run_node_code(_node(ERRLOG, "Build Error Trace"), {})
    attrs, _ = _error_attrs(res)
    assert attrs["langfuse.observation.metadata.failed_node"] == {"stringValue": "unknown"}
    assert attrs["langfuse.observation.metadata.execution_id"] == {"stringValue": "unknown"}


def test_error_logger_code_never_reads_error_message_stack_or_description():
    code = _node(ERRLOG, "Build Error Trace")["parameters"]["jsCode"]
    body = re.sub(r"(?m)^\s*//.*$", "", code)
    for forbidden in (".error", "message", "stack", "description", "$json", "stdout", "$('"):
        assert forbidden not in body, forbidden


def test_error_logger_known_names_cover_every_node_and_workflow_that_can_fail():
    code = _node(ERRLOG, "Build Error Trace")["parameters"]["jsCode"]
    known_nodes = json.loads(re.search(r"KNOWN_NODES = (\[.*?\]);", code).group(1))
    known_wfs = json.loads(re.search(r"KNOWN_WORKFLOWS = (\[.*?\]);", code).group(1))
    assert set(known_nodes) == {n["name"] for n in FORM["nodes"]} | {n["name"] for n in APPROVAL["nodes"]}
    assert known_wfs == [FORM["name"], APPROVAL["name"]]


def test_error_logger_structure():
    assert [n["type"] for n in ERRLOG["nodes"]] == [
        "n8n-nodes-base.errorTrigger", "n8n-nodes-base.code", "n8n-nodes-base.httpRequest"]
    assert ERRLOG["connections"]["Error Trigger"]["main"][0][0]["node"] == "Build Error Trace"
    assert ERRLOG["connections"]["Build Error Trace"]["main"][0][0]["node"] == "Send Error to Langfuse"
    http = _node(ERRLOG, "Send Error to Langfuse")
    assert http["parameters"]["jsonBody"] == "={{ $json.langfuse_body }}"
    assert http["parameters"]["options"]["response"]["response"]["neverError"] is True
    assert http["onError"] == "continueRegularOutput"


def test_both_production_workflows_point_at_the_error_logger():
    assert ERRLOG["id"]
    assert FORM["settings"]["errorWorkflow"] == ERRLOG["id"]
    assert APPROVAL["settings"]["errorWorkflow"] == ERRLOG["id"]
    assert "errorWorkflow" not in ERRLOG["settings"]      # 錯誤記錄自己失敗時不要遞迴觸發自己
