#!/usr/bin/env python3
"""
test_generation_agent.py

測試生成代理 MVP:給定一或多個目標函式,呼叫模型生成 pytest 測試草稿,寫進
一個 DRAFT 檔案——絕對不碰真正的 tests/ 目錄、不自動判斷通過就收錄。草稿
是不是真的測到對的東西,一律要人工逐條看過斷言邏輯才能核准,核准後才手動
搬進 tests/、改成正式檔名、接進 pre-push 閘門(githooks/pre-push 目前是
寫死跑哪些測試檔案,新檔案要手動加進那份清單,不會自動被撿到)。

跟 code_review_agent.py 的關係:同樣是「呼叫模型做本來要人工做的判斷,但
絕對不讓它自己核准/收錄」——這裡的人工核准點更早,在測試案例被信任、進
版控之前就要經過人看,不是等測試失敗才靠 override。

用法:
  python3 scripts/test_generation_agent.py \\
      --target scripts/generate_report.py:slugify_heading \\
      --target scripts/generate_report.py:filter_citations \\
      --style-reference tests/test_detect_think_reason.py \\
      --hint "相關度剛好等於門檻值 0.6 的邊界情況" \\
      --hint "過濾後清單為空的情況" \\
      --hint "slugify_heading 遇到中文、™符號、問號等特殊字元的情況" \\
      --hint "兩個標題文字不同但目前會不會被 slugify 成一樣的 slug（記錄現況,不是要求修掉）" \\
      --output tests/test_report_helpers_DRAFT.py
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8000/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Qwen/Qwen3-30B-A3B")

SYSTEM_PROMPT = (
    "你是資深 Python 測試工程師,幫這個 repo 的既有函式生成 pytest 測試草稿。\n"
    "你會看到:(1) 這個 repo 現有測試檔案的完整內容,當作風格基準——命名慣例、"
    "parametrize 寫法、註解風格、檔案開頭的『測試重點』說明區塊,新測試要跟這個"
    "風格一致;(2) 一個或多個要測試的函式原始碼;(3) 一份必須覆蓋的邊界情況清單。\n"
    "規則:\n"
    "1. 只能輸出一份完整、語法正確的 Python 測試檔案,不要有任何其他文字、"
    "不要用 markdown code fence 包起來\n"
    "2. 每條測試案例的斷言要針對函式的『實際行為』,不能寫恆真的斷言(例如"
    "assert True、或者斷言值等於函式呼叫結果本身這種繞圈子)\n"
    "3. 清單裡列的每一種邊界情況都要有至少一條測試對應,不能省略\n"
    "4. import 目標函式的方式:用 sys.path.insert 把 repo 的 scripts/ 目錄加進"
    "path,再直接 import 函式名稱(照現有測試檔案的做法)\n"
    "5. 檔案開頭要有一段 docstring 說明這份測試涵蓋了哪些情境、為什麼"
)


def log(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def die(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def extract_function_source(repo_root: str, file_path: str, func_name: str) -> str:
    full = os.path.join(repo_root, file_path)
    if not os.path.isfile(full):
        die(f"找不到檔案：{file_path}")
    with open(full, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                die(f"找到 {func_name} 但無法取出原始碼片段（{file_path}）")
            return segment
    die(f"在 {file_path} 裡找不到函式 {func_name}")


def parse_target(spec: str):
    if ":" not in spec:
        die(f"--target 需要 <檔案路徑>:<函式名稱> 格式，收到：{spec}")
    file_path, _, func_name = spec.rpartition(":")
    return file_path, func_name


def call_llm(messages, timeout=300):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
        # 設計測試案例需要推理邊界情況,不是快速分類,值得開思考模式。
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        CHAT_URL, data=json.dumps(payload).encode("utf-8"), method="POST"
    )
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        die(f"呼叫模型失敗：{e}")
    try:
        return body["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        die(f"模型回應格式不如預期：{body}")


def strip_think_and_fence(text: str) -> str:
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = re.sub(r"^```(python)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def run_pytest_on_draft(repo_root: str, draft_path: str):
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    python = venv_python if os.path.isfile(venv_python) else sys.executable
    result = subprocess.run(
        [python, "-m", "pytest", draft_path, "-v"],
        cwd=repo_root, capture_output=True, text=True,
    )
    return result.returncode, result.stdout + result.stderr


def main():
    ap = argparse.ArgumentParser(
        description="生成 pytest 測試草稿（不寫進正式 tests/，只出草稿檔+跑一次 pytest 附上結果）。"
    )
    ap.add_argument("--target", action="append", required=True,
                     help="<檔案路徑>:<函式名稱>，可重複指定多個")
    ap.add_argument("--style-reference", required=True, help="風格基準測試檔案路徑")
    ap.add_argument("--hint", action="append", default=[],
                     help="這次一定要覆蓋的情境描述，可重複指定多個")
    ap.add_argument("--output", required=True, help="草稿輸出路徑（例如 tests/test_xxx_DRAFT.py）")
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    style_full = os.path.join(repo_root, args.style_reference)
    if not os.path.isfile(style_full):
        die(f"風格基準檔案不存在：{args.style_reference}")
    with open(style_full, "r", encoding="utf-8") as f:
        style_reference = f.read()

    function_blocks = []
    for spec in args.target:
        file_path, func_name = parse_target(spec)
        log(f"抓取 {file_path}:{func_name} 的原始碼...")
        source = extract_function_source(repo_root, file_path, func_name)
        function_blocks.append(f"# 來自 {file_path} 的 {func_name}()\n{source}")

    hints_text = "\n".join(f"- {h}" for h in args.hint) if args.hint else "（沒有額外指定，自行判斷合理的邊界情況）"

    user_content = (
        f"風格基準（現有測試檔案的完整內容,新測試要照這個風格寫）：\n"
        f"```python\n{style_reference}\n```\n\n"
        f"---\n\n要測試的函式：\n\n```python\n" + "\n\n".join(function_blocks) + "\n```\n\n"
        f"---\n\n這次一定要覆蓋的邊界情況：\n{hints_text}"
    )

    log("呼叫模型生成測試草稿（思考模式開啟，可能要一段時間）...")
    raw = call_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])
    draft_code = strip_think_and_fence(raw)

    try:
        ast.parse(draft_code)
    except SyntaxError as e:
        die(f"生成的草稿語法錯誤，不寫入檔案：{e}\n\n原始輸出：\n{raw[:2000]}")

    out_full = os.path.join(repo_root, args.output)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    with open(out_full, "w", encoding="utf-8") as f:
        f.write(draft_code)
    log(f"草稿已寫入 {args.output}")

    log("立刻跑一次 pytest，把目前的 PASS/FAIL 結果附上（不代表核准，只是給審核時多一點依據）...")
    rc, output = run_pytest_on_draft(repo_root, args.output)
    print("\n" + "=" * 78)
    print("pytest 執行結果")
    print("=" * 78)
    print(output)
    print("=" * 78)
    log(f"pytest exit code: {rc}（草稿仍然只是草稿，不因為這裡全過就自動核准）")


if __name__ == "__main__":
    main()
