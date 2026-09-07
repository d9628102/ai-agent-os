#!/usr/bin/env python3
"""
run_think_ab_test.py

Repeated A/B of Qwen3's thinking mode for the follow-up in
docs/rag-findings.md Known Issue #7: is the observed quality difference
stable, or was the single run sampling luck?

Runs rag_answer.py as a SEPARATE SUBPROCESS for every repeat — each run is
a fresh process making fresh requests, so nothing (seeds, caches, module
state) is shared between repeats. Then writes every full answer verbatim
to a markdown log, so the results stay auditable rather than collapsing
into a summary.

The two behaviours being counted need a human to judge, so the log leaves
an annotation table with automatic HINTS only, clearly marked as hints:
  - Q3 越界: does the answer characterize what ERP/CRM themselves do,
    which the source document never states?
  - Q4 辨析: does the answer distinguish Service Matrix's "Customer
    Success" module from the "PSF Customer Success Agent™"?

Usage:
  python3 scripts/run_think_ab_test.py                 # 3 repeats per mode
  python3 scripts/run_think_ab_test.py --repeats 5
"""
import argparse
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAG_ANSWER = os.path.join(SCRIPT_DIR, "rag_answer.py")

QUESTION_HEADER_RE = re.compile(r"^Q: (.+)$", re.MULTILINE)

# Phrases that would mean the model described the comparison target itself.
# Hints only — "而非" and "通常" also appear in perfectly grounded sentences.
Q3_OVERREACH_HINTS = [
    "通常", "而非", "傳統 ERP", "傳統的 ERP", "業務流程管理",
    "客戶關係管理", "局限於", "聚焦於", "一般 ERP 系統", "ERP 系統是",
]
# The distinction we want to see survive: naming the module and the Agent
# as different things.
Q4_DISTINCTION_HINTS = ["模組", "Core Module", "並非同一", "不是 Agent", "而非 Agent"]


def die(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg):
    print(f"[INFO] {msg}", flush=True)


def parse_answers(stdout: str):
    """
    Pull each question's 【回答】 block out of rag_answer.py's output.
    Returns [(question, answer_text), ...] in order.
    """
    results = []
    # Split the transcript at each "Q: ..." header, keeping what follows.
    parts = QUESTION_HEADER_RE.split(stdout)
    # parts = [preamble, q1, body1, q2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        question = parts[i].strip()
        body = parts[i + 1]
        if "【回答】" not in body:
            results.append((question, "(解析不到【回答】區塊)"))
            continue
        answer = body.split("【回答】", 1)[1]
        # The timing line closes the answer block.
        answer = answer.split("【耗時】", 1)[0]
        results.append((question, answer.strip()))
    return results


def hint(answer: str, needles):
    found = [n for n in needles if n in answer]
    return found


def run_once(queries_file, think: bool, top_k, max_tokens, temperature):
    cmd = [sys.executable, RAG_ANSWER, "--queries-file", queries_file,
           "--top-k", str(top_k), "--max-tokens", str(max_tokens),
           "--temperature", str(temperature)]
    # Always pass the mode explicitly. Relying on rag_answer.py's default for
    # either arm would silently run both arms in the same mode whenever that
    # default changes — and the A/B would compare nothing while still
    # producing plausible-looking output.
    cmd.append("--think" if think else "--no-think")

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    elapsed = time.time() - started

    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        die(
            f"rag_answer.py 回傳非 0 ({proc.returncode})。上面是它的完整輸出。"
            f"常見原因：vllm-server/vllm-embed 沒在跑，或 collection 不存在。"
        )
    return parse_answers(proc.stdout), elapsed, proc.stdout


def main():
    ap = argparse.ArgumentParser(description="Repeated thinking-mode A/B for RAG answers.")
    ap.add_argument("--queries-file",
                    default=os.path.join(SCRIPT_DIR, "data", "psf-eim-qa-q3q4-only.txt"))
    ap.add_argument("--repeats", type=int, default=3, help="每種模式重複幾次 (預設 3)")
    ap.add_argument("--output",
                    default=os.path.join(REPO_ROOT, "docs", "think-mode-ab-test-results.md"))
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    if not os.path.isfile(args.queries_file):
        die(f"找不到題目檔: {args.queries_file}")
    if not os.path.isfile(RAG_ANSWER):
        die(f"找不到 {RAG_ANSWER}")

    modes = [("think 開啟", True), ("think 關閉", False)]
    total_runs = len(modes) * args.repeats
    with open(args.queries_file, encoding="utf-8") as qf:
        n_questions = sum(1 for ln in qf if ln.strip() and not ln.startswith("#"))
    log(f"共 {total_runs} 次獨立執行（{len(modes)} 種模式 × {args.repeats} 次），"
        f"每次涵蓋 {n_questions} 題 = {total_runs * n_questions} 次生成。"
        f"每次執行都是全新子行程，不共用狀態。")
    log("think 開啟時單次約 18s+，整體可能需要數分鐘，屬預期耗時。")

    runs = []          # (mode_label, think, run_no, [(q, a)], elapsed)
    raw_transcripts = []
    for mode_label, think in modes:
        for n in range(1, args.repeats + 1):
            log(f"執行中：{mode_label} 第 {n}/{args.repeats} 次…")
            answers, elapsed, stdout = run_once(
                args.queries_file, think, args.top_k, args.max_tokens, args.temperature
            )
            runs.append((mode_label, think, n, answers, elapsed))
            raw_transcripts.append((mode_label, n, stdout))
            log(f"  完成，耗時 {elapsed:.1f}s，取得 {len(answers)} 題回答。")

    questions = [q for q, _ in runs[0][3]]
    if len(questions) < 2:
        die(f"預期至少 2 題，實際解析到 {len(questions)} 題。請檢查題目檔與輸出格式。")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# 思考模式 A/B 重複測試結果\n\n")
        f.write(f"對應 `docs/rag-findings.md` Known Issue #7 的待辦。"
                f"每種模式重複 {args.repeats} 次，"
                f"每次皆為獨立子行程呼叫 `scripts/rag_answer.py`，不共用任何狀態。\n\n")
        f.write(f"- 題目檔：`{os.path.relpath(args.queries_file, REPO_ROOT)}`\n")
        f.write(f"- 參數：top_k={args.top_k}, max_tokens={args.max_tokens}, "
                f"temperature={args.temperature}\n")
        f.write(f"- 產生時間：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 耗時\n\n| 模式 | 次數 | 耗時 |\n|---|---|---|\n")
        for mode_label, _, n, _, elapsed in runs:
            f.write(f"| {mode_label} | 第 {n} 次 | {elapsed:.1f}s |\n")
        f.write("\n")

        f.write("## 自動偵測提示（僅供參考，需人工確認）\n\n")
        f.write("> 這些是關鍵詞比對的結果，不是判定。"
                "「而非」「通常」也可能出現在完全有依據的句子裡，"
                "最終判斷請看下方完整回答。\n\n")
        f.write("| 題目 | 模式 | 次數 | 命中的關鍵詞 |\n|---|---|---|---|\n")
        for mode_label, _, n, answers, _ in runs:
            for idx, (q, a) in enumerate(answers):
                needles = Q3_OVERREACH_HINTS if idx == 0 else Q4_DISTINCTION_HINTS
                label = "Q3 越界跡象" if idx == 0 else "Q4 辨析跡象"
                found = hint(a, needles)
                f.write(f"| {label} | {mode_label} | 第 {n} 次 | "
                        f"{'、'.join(found) if found else '（無）'} |\n")
        f.write("\n")

        f.write("## 人工標註表（請填寫）\n\n")
        f.write("| 題目 | 模式 | 第1次 | 第2次 | 第3次 | 出現機率 |\n|---|---|---|---|---|---|\n")
        blanks = " | ".join(["有/無"] * args.repeats)
        for label in ["Q3 越界", "Q4 辨析保留"]:
            for mode_label, _ in modes:
                f.write(f"| {label} | {mode_label} | {blanks} | /{args.repeats} |\n")
        f.write("\n")

        f.write("## 完整回答（逐次保留）\n\n")
        for qi, question in enumerate(questions):
            f.write(f"### {'Q3' if qi == 0 else 'Q4'}：{question}\n\n")
            for mode_label, _, n, answers, _ in runs:
                if qi >= len(answers):
                    continue
                f.write(f"#### {mode_label} — 第 {n} 次\n\n")
                f.write(answers[qi][1].strip() + "\n\n")

    log(f"完成。結果已寫入 {args.output}")
    log("請人工填寫「人工標註表」，或把這份檔案內容貼回對話中由我判讀。")


if __name__ == "__main__":
    main()
