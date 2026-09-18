#!/usr/bin/env python3
"""
red_flag_detection.py

從已檢索的文件片段裡抽取「現象／為何是紅旗／必要動作」三段式紅旗結構＋
嚴重度分級——簡報工廠往 DD 報告方向擴充的第二步。技術路線跟
numeric_consistency.py 一樣：不重新設計抽取邏輯，用一次獨立的 LLM 呼叫
對現有 answer_one() 的 context/visible 做結構化抽取，純邏輯（JSON 解析、
去重）拆成獨立函式方便測試。

紅旗的判斷標準是「主張 vs. 查核結果不符」或「數字 vs. 數字對不上」這種
矛盾結構，不是泛泛的風險陳述——這樣才能把風險矩陣裡的一般性風險（例如
「團隊規模未揭露」）跟真正的紅旗分開。判斷依據是內容本身有沒有這種矛盾
結構，不是看有沒有出現「紅旗」字樣，因為實際文件不會每次都幫我們把紅旗
標好。

不做外部查證——紅旗裡如果有競品比較這類資訊，那是分析師已經查過、寫進
被 ingest 文件裡的既成事實，我們只是把這段已經存在的文字結構化抽取出來，
不是自己去做那次查證。
"""
import json
import re

from rag_common import http_json

RED_FLAG_SYSTEM_PROMPT = (
    "你是嚴謹的盡職調查分析師。你會看到一則問答的問題、系統回答，以及檢索到的"
    "文件片段全文。你的任務是判斷『系統回答所描述的內容本身』有沒有構成一個"
    "『紅旗』。\n"
    "先弄清楚這個任務『不是』什麼：這不是在查核『系統回答有沒有正確引用文件』"
    "——那是另一件事（已經有別的機制在做）。就算系統回答完全正確、逐字忠實"
    "於文件內容，答案描述的『情境本身』仍然可能是一個紅旗（例如：文件記載"
    "『某公司宣稱X，但查證結果是Y』，系統回答正確轉述了這段內容，答案本身"
    "沒有錯，但這段內容描述的『X vs. Y 對不上』這件事就是紅旗）。所以：如果"
    "你發現系統回答忠實地反映了文件內容，這不代表『沒有紅旗』，你還是要繼續"
    "判斷『回答所轉述的那個情境本身』是否構成主張對不上的矛盾。\n"
    "重要規則——先在心裡把『系統回答』裡實際提到的具體內容列出來，只根據這些"
    "內容判斷是否構成紅旗，不能是檢索片段全文裡剛好出現、但系統回答根本沒有"
    "提到或轉述的其他內容——即使那段內容本身描述了一個真實存在的紅旗，只要"
    "系統回答完全沒提到相關主題，這題就不算紅旗，那個紅旗應該留給真正問到"
    "相關主題的問題去抓，不是隨便哪一題都算。檢索片段全文的作用是幫你確認"
    "系統回答轉述的內容有沒有依據、補充原始數字或說法，不是拿來判斷『回答"
    "對不對』。\n"
    "另外要小心：不要因為檢索片段全文裡沒有逐字出現系統回答提到的某個細節，"
    "就認定是『矛盾』——片段全文只給你部分內容，沒看到不等於不存在或被否定，"
    "只有片段全文明確寫了『相反的事實』時才算真正的矛盾。\n"
    "紅旗的定義是：系統回答所描述的內容裡，有具體的『主張 vs. 查核結果不符』"
    "或『數字 vs. 數字對不上』的矛盾結構，不是泛泛的風險陳述或負面資訊。"
    "以下情況不算紅旗，即使聽起來像風險，都不要判定為 is_red_flag=true：\n"
    "- 單純建議『應確認』『待查證』『需要進一步了解』的提醒，但沒有指出兩個"
    "互相矛盾的具體主張或數字（例如：某位顧問的實際投入程度需要再確認、"
    "是否為正式董事待查——這是正常的盡調建議，不是矛盾，不算紅旗）\n"
    "- 單純列出能力缺口、風險因子、或負面因素，但沒有『兩件事對不上』的"
    "結構（例如「團隊規模未揭露」「無公開財務主管」是風險陳述，不是矛盾，"
    "不算紅旗）\n"
    "- 正面評價、背景介紹、或建議性質的內容（例如財務揭露誠實度的正面評語、"
    "投資結構的建議機制、創辦人學經歷介紹）\n"
    "判斷依據是內容本身有沒有『兩件事對不上』的矛盾結構，不是看有沒有出現"
    "『紅旗』字樣。\n"
    "如果構成紅旗，輸出：\n"
    "1. title：15字以內的簡短標題\n"
    "2. phenomenon（現象）：具體觀察到的矛盾是什麼，包含引用的原始數字/說法\n"
    "3. why_it_matters（為何是紅旗）：這個矛盾為什麼值得投資人/審查者注意\n"
    "4. required_action（必要動作）：建議要求對方補充或查證什麼\n"
    "5. severity：high/medium/low 三選一，依對投資判斷的影響程度評定\n"
    "如果不構成紅旗，is_red_flag 填 false，其他欄位都填 null。\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"is_red_flag": true/false, "title": "<字串或null>", '
    '"phenomenon": "<字串或null>", "why_it_matters": "<字串或null>", '
    '"required_action": "<字串或null>", "severity": "<high/medium/low或null>"}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)

_VALID_SEVERITIES = ("high", "medium", "low")


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _not_red_flag(parse_error):
    return {
        "is_red_flag": False,
        "title": None,
        "phenomenon": None,
        "why_it_matters": None,
        "required_action": None,
        "severity": None,
        "parse_error": parse_error,
    }


def parse_red_flag_response(raw: str) -> dict:
    """把模型輸出解析成紅旗判斷結果。is_red_flag=False 是正常判斷（不是
    錯誤），parse_error 只在真正解析失敗，或 is_red_flag=true 卻缺欄位/
    severity 不合法時才有值——後者視同抽取失敗，不半殘顯示一個缺內容的
    紅旗。"""
    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        is_red_flag = bool(parsed.get("is_red_flag"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return _not_red_flag(f"紅旗判斷結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    if not is_red_flag:
        return _not_red_flag(None)

    title = parsed.get("title")
    phenomenon = parsed.get("phenomenon")
    why_it_matters = parsed.get("why_it_matters")
    required_action = parsed.get("required_action")
    severity = parsed.get("severity")

    if (not title or not phenomenon or not why_it_matters or not required_action
            or severity not in _VALID_SEVERITIES):
        return _not_red_flag(
            f"is_red_flag=true 但欄位不完整或 severity 不是合法值：{parsed}"
        )

    return {
        "is_red_flag": True,
        "title": title,
        "phenomenon": phenomenon,
        "why_it_matters": why_it_matters,
        "required_action": required_action,
        "severity": severity,
        "parse_error": None,
    }


def detect_red_flag(question, answer, context, chat_url, chat_model, timeout=180):
    """對一題的回答＋檢索片段全文，判斷是否構成紅旗。永遠回傳一個結構
    一致的 dict，不拋例外。

    開思考模式：這裡要求的判斷（先列出答案的具體主張、再逐一核對片段全文
    有沒有明確反駁其中一項，而不是看到片段全文裡任何紅旗字樣就套用）需要
    多步驟比對，跟這個 repo 已經用 A/B 測試驗證過的結論一致——關掉思考會
    讓模型在需要跨內容比對的任務上走捷徑、產生站不住腳的判斷（見
    rag_answer.py 的 DEFAULT_THINKING 設計理由）。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": RED_FLAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"問題：{question}\n\n系統回答：\n{answer}\n\n"
                    f"檢索到的文件片段全文：\n{context}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return _not_red_flag(f"呼叫紅旗判斷模型失敗 (HTTP {status}): {resp}")
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return _not_red_flag(f"紅旗判斷模型回應格式不如預期: {resp}")

    return parse_red_flag_response(raw)


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def dedupe_red_flags(entries):
    """entries: [{"question": str, "detection": detect_red_flag() 的回傳值,
    "heading": str|None}, ...]

    只處理 detection["is_red_flag"] 為真的項目。用正規化後的 title（去頭尾
    空白、轉小寫）做完全字串比對去重——同一個紅旗被多題答案各自抓到時合併
    成一筆，記錄是哪些問題都指向它。title 寫法不同（同一紅旗被不同問法下
    了不同標題）不會被合併，這是刻意的簡化，跟數字一致性檢查「先用字串直
    接比對，不夠再進階」同一個原則。

    回傳依 severity（high > medium > low）排序的清單，同 severity 保留原
    始出現順序。
    """
    seen = {}
    order = []
    for entry in entries:
        det = entry["detection"]
        if not det["is_red_flag"]:
            continue
        key = det["title"].strip().lower()
        if key not in seen:
            seen[key] = {
                "title": det["title"],
                "phenomenon": det["phenomenon"],
                "why_it_matters": det["why_it_matters"],
                "required_action": det["required_action"],
                "severity": det["severity"],
                "matched_questions": [],
                "headings": [],
            }
            order.append(key)
        seen[key]["matched_questions"].append(entry["question"])
        if entry.get("heading"):
            seen[key]["headings"].append(entry["heading"])

    merged = [seen[key] for key in order]
    merged.sort(key=lambda rf: _SEVERITY_ORDER.get(rf["severity"], len(_SEVERITY_ORDER)))
    return merged
