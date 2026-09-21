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

生成後的機械式檢查(find_orphan_think_tags):孤立、沒有配對開頭 <think> 的
結尾 </think> 標籤,已經連續在四輪不同功能的測試草稿裡出現過。SYSTEM_PROMPT
的第 6 條已經明文禁止,但實測證明 prompt 規則對模型的約束力不到 100%,
所以改成用字串掃描機械攔截:抓到就視為生成失敗、重新生成一次,兩次都有
就直接中止、不寫草稿檔。理由是人工審核的時間應該花在看斷言邏輯對不對,
不該每次都重新驗證「標籤配對有沒有問題」這種機械檢查本來就該擋掉的事。

Usage:
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
import re
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
    "5. 檔案開頭要有一段 docstring 說明這份測試涵蓋了哪些情境、為什麼\n"
    "6. 如果要測試『輸出被 <think>...</think> 包住時要能正確剝殼』這種情境,"
    "一定要用真正配對的開頭 <think> 加結尾 </think> 標籤,不能只寫孤立的"
    "結尾 </think> 標籤(沒有對應開頭標籤的孤立結尾標籤不會被剝殼邏輯處理,"
    "這種輸入不反映真實模型輸出的樣子,寫了只會產生一條註定失敗、且測不到"
    "真正想測的剝殼行為的假案例)——這個錯誤已經在多輪測試草稿裡重複出現,"
    "生成時務必避免"
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


def call_llm(messages, max_tokens=8192, timeout=300):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
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
        choice = body["choices"][0]
    except (KeyError, IndexError, TypeError):
        die(f"模型回應格式不如預期：{body}")
    # 截斷要單獨報，不要讓它偽裝成語法錯誤。實際踩過：目標函式多、--hint 多的
    # 時候，模型光是思考就把額度用完，輸出只有一個沒有結尾的 <think>，剝殼剝
    # 不掉（剝殼只吃配對標籤），整段推理散文就被當成草稿原始碼，最後由語法
    # 檢查以「invalid character '，'」報錯——訊息完全指錯方向。
    if choice.get("finish_reason") == "length":
        die(f"模型輸出被 max_tokens（{max_tokens}）截斷，草稿不完整、不寫入檔案。"
            f"目標函式多或 --hint 多的時候，思考內容本身就可能把額度用光、"
            f"還沒開始寫程式碼。用 --max-tokens 調高後重跑。")
    try:
        return choice["message"].get("content") or ""
    except (KeyError, TypeError):
        die(f"模型回應格式不如預期：{body}")


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(python)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def strip_think_and_fence(text: str) -> str:
    """剝掉開頭那一段模型自己的推理區塊，以及包住程式碼的 markdown fence。

    推理區塊的結尾邊界用「切掉之後剩下的內容能不能當成 Python 解析」決定，
    而不是用字串樣式去猜。這個作法是連續踩三次坑之後才收斂出來的：

    1. 原本是全文取代所有配對的 <think>...</think>。但這支代理生成的是
       「測試程式碼」，測試資料本身就可能合法地包含配對標籤字串
       （SYSTEM_PROMPT 第 6 條正是要求這樣寫），結果那些測試資料被連帶
       刪掉：草稿語法還是對的、測試還會過，但已經測不到剝殼行為。更糟的是
       這讓管線系統性偏袒錯誤樣式——寫對（配對標籤）被刪掉，寫錯（孤立結尾
       標籤）反而原封不動存活，正是孤立標籤問題連續四輪重複出現的成因之一。
    2. 改成錨定開頭、非貪婪取第一個 </think>。但模型推理時會在散文裡行內
       引用這個標籤字串（實際踩到：「沒有對應開頭的</think>）」），那個引用
       被當成區塊結尾，剩下的推理散文被當成程式碼，以「invalid character
       '）'」這種完全指錯方向的語法錯誤收場。
    3. 再改成要求結尾標籤必須是該行最後一個東西。模型下一次就把引用寫在
       行尾了，同樣被誤認——純字串樣式擋不住「散文裡提到這個標籤」這件事。

    所以改成直接編碼真正的需求：逐個 </think> 邊界試，第一個「剩下的內容
    是合法 Python」的就是真正的結尾。推理散文不會是合法 Python，所以行內
    引用會自動被跳過。全部邊界都試不出合法 Python 時（例如整段被 max_tokens
    截斷、根本還沒產出程式碼），原樣回傳，讓呼叫端的語法檢查去報錯。
    """
    body = text.lstrip()
    if body.startswith("<think>"):
        for m in re.finditer(r"</think>[ \t]*\n?", body):
            candidate = _strip_fence(body[m.end():])
            if _parses_as_python(candidate):
                return candidate
    return _strip_fence(body)


def find_orphan_think_tags(code: str):
    """找出草稿裡孤立的結尾 </think> 標籤（前面沒有配對的開頭 <think>），
    回傳 1-based 行號清單，沒有問題時回傳空清單。純字串掃描、不解析 Python
    語法——這些標籤幾乎都是出現在測試資料的字串常值裡，字串層級掃描最直接，
    也不會因為草稿本身語法有問題就掃不出來。

    配對規則就是一般的巢狀計數：遇到 <think> 深度加一，遇到 </think> 時
    深度為 0 就是孤立標籤，否則深度減一。刻意不管「有開頭但沒結尾」那種
    情況——剝殼邏輯（strip_think_and_fence）只吃配對標籤，沒結尾的開頭
    標籤留在原地會直接讓語法檢查失敗，不需要這裡重複攔一次。"""
    depth = 0
    orphans = []
    for m in re.finditer(r"</?think>", code):
        if m.group(0) == "<think>":
            depth += 1
        elif depth == 0:
            orphans.append(code.count("\n", 0, m.start()) + 1)
        else:
            depth -= 1
    return orphans


CONFIG_PATH_PATTERN = re.compile(r"[\w][\w\-./]*\.(?:json|yaml|yml)\b")

_MAX_CONFIG_CHARS = 20000


def find_referenced_config_files(repo_root: str, text: str):
    """掃描函式原始碼（含 docstring）裡看起來像設定檔路徑的字串（結尾是
    .json/.yaml/.yml 的相對路徑），解析成 repo 內真的存在的檔案。

    用來在生成測試草稿前，把目標函式實際讀寫的設定檔真實內容自動附加進
    生成請求——不能只靠人在 --hint 裡手動貼 schema。scoring_templates.json
    的結構已經因為沒貼而被生成代理臆測出不相容的假結構三次（見
    docs/rag-findings.md「虛構跟真實系統不相容的資料結構」），操作習慣層級
    的提醒證明不可靠，這裡改成工具本身的行為，不再依賴人記得。

    純字串掃描，不解析 AST——設定檔路徑常常只出現在 docstring 裡（例如
    load_scoring_template() 本身不寫死路徑，是呼叫端傳進來的參數，但
    docstring 裡明文寫了它讀的是哪個檔案），AST 抓不到 docstring 以外的
    註解或字串常值組合，字串掃描反而更直接、更不會漏。

    回傳 [(相對路徑, 絕對路徑), ...]，依出現順序、去重；不存在的路徑（誤判
    成路徑格式的字串）直接濾掉，不回報。"""
    seen = set()
    found = []
    for m in CONFIG_PATH_PATTERN.finditer(text):
        candidate = m.group(0)
        if candidate in seen:
            continue
        full = os.path.join(repo_root, candidate)
        if os.path.isfile(full):
            seen.add(candidate)
            found.append((candidate, full))
    return found


def build_config_reference_block(referenced_configs):
    """把偵測到的設定檔真實內容組成要附加進 user_content 的文字區塊。
    單一檔案超過 _MAX_CONFIG_CHARS 就截斷並明確標註，避免罕見的巨大設定檔
    把生成請求撐爆——目前實際遇到的設定檔（scoring_templates.json 等）都
    遠小於這個上限，這裡只是防呆。"""
    if not referenced_configs:
        return ""
    parts = []
    for rel_path, full_path in referenced_configs:
        with open(full_path, "r", encoding="utf-8") as f:
            body = f.read()
        truncated_note = ""
        if len(body) > _MAX_CONFIG_CHARS:
            body = body[:_MAX_CONFIG_CHARS]
            truncated_note = "\n（檔案過大，已截斷——若測試需要更完整內容請自行擴充 --hint）"
        parts.append(f"# {rel_path} 的真實內容{truncated_note}\n{body}")
    return (
        "\n\n---\n\n以下是目標函式原始碼（含 docstring）裡提到的設定檔的"
        "真實內容，不是虛構範例。生成測試時任何用到這份設定檔結構的 mock/"
        "測試資料都必須跟這裡的真實 schema 一致，不能自己發明不相容的結構：\n\n"
        "```\n" + "\n\n".join(parts) + "\n```"
    )


def generate_draft(messages, max_tokens=8192):
    """呼叫模型生成一份草稿、剝殼、檢查語法，回傳草稿原始碼。語法錯誤直接
    中止（維持原本行為）——抽成獨立函式是為了讓孤立 think 標籤的機械檢查
    可以用同一份 messages 重新生成一次。"""
    raw = call_llm(messages, max_tokens=max_tokens)
    draft_code = strip_think_and_fence(raw)
    try:
        ast.parse(draft_code)
    except SyntaxError as e:
        die(f"生成的草稿語法錯誤，不寫入檔案：{e}\n\n原始輸出：\n{raw[:2000]}")
    return draft_code


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
    ap.add_argument("--max-tokens", type=int, default=8192,
                     help="生成上限（預設 8192）。目標函式多或 --hint 多的時候，"
                          "思考內容可能把額度用光，這時要調高")
    ap.add_argument("--repo-root", default=os.getcwd())
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    style_full = os.path.join(repo_root, args.style_reference)
    if not os.path.isfile(style_full):
        die(f"風格基準檔案不存在：{args.style_reference}")
    with open(style_full, "r", encoding="utf-8") as f:
        style_reference = f.read()

    function_blocks = []
    referenced_configs = []
    seen_config_paths = set()
    for spec in args.target:
        file_path, func_name = parse_target(spec)
        log(f"抓取 {file_path}:{func_name} 的原始碼...")
        source = extract_function_source(repo_root, file_path, func_name)
        function_blocks.append(f"# 來自 {file_path} 的 {func_name}()\n{source}")
        for rel_path, full_path in find_referenced_config_files(repo_root, source):
            if rel_path not in seen_config_paths:
                seen_config_paths.add(rel_path)
                referenced_configs.append((rel_path, full_path))

    if referenced_configs:
        log(f"偵測到目標函式提及設定檔路徑：{[r for r, _ in referenced_configs]}，"
            f"自動讀取真實內容附加進生成請求（不需要在 --hint 裡手動貼）...")

    hints_text = "\n".join(f"- {h}" for h in args.hint) if args.hint else "（沒有額外指定，自行判斷合理的邊界情況）"

    user_content = (
        f"風格基準（現有測試檔案的完整內容,新測試要照這個風格寫）：\n"
        f"```python\n{style_reference}\n```\n\n"
        f"---\n\n要測試的函式：\n\n```python\n" + "\n\n".join(function_blocks) + "\n```\n\n"
        f"---\n\n這次一定要覆蓋的邊界情況：\n{hints_text}"
        f"{build_config_reference_block(referenced_configs)}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    log("呼叫模型生成測試草稿（思考模式開啟，可能要一段時間）...")
    draft_code = generate_draft(messages, args.max_tokens)

    orphans = find_orphan_think_tags(draft_code)
    if orphans:
        log(f"機械檢查攔下：草稿第 {orphans} 行出現孤立的 </think> 結尾標籤"
            f"（沒有配對的開頭 <think>）。這是重複出現過四輪的生成錯誤，"
            f"直接視為生成失敗，重新生成一次...")
        draft_code = generate_draft(messages, args.max_tokens)
        orphans = find_orphan_think_tags(draft_code)
        if orphans:
            die(f"重新生成後第 {orphans} 行仍然有孤立的 </think> 標籤，"
                f"不寫入草稿檔。人工審核不該花時間抓這種機械性錯誤——"
                f"請調整 --hint 或 SYSTEM_PROMPT 後重跑。")

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
