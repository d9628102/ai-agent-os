#!/usr/bin/env python3
"""
veto_check.py

「五不合作」否決檢查——簡報工廠往完整 DD 報告模板方向擴充的第五步
（數字一致性、紅旗、跨文件比對、九格評分之後）。依據《PSF六壬合夥
生態系統》第十四節「五不合作」：人不明、資源不實、權責不清、利益
不明、風險不揭露，任一條觸發，不管九格總分多高都建議不合作。這是
獨立疊加在九格評分之上的檢查，不是評分的第十個維度、也不取代評分。

五條規則不是同一種判斷邏輯：
- 資源不實是「矛盾型」——文件裡有明確的『主張 vs 查核結果對不上』的
  矛盾結構（跟 red_flag_detection.py 判斷的矛盾同一類），例如簡報宣稱
  智慧醫療全方位布局，財報卻顯示代理收入+酒精銷售。
- 其餘四條（人不明、權責不清、利益不明、風險不揭露）是「缺失型」——
  不是兩個主張互相矛盾，是文件/回答明確自己承認資訊揭露不完整（例如
  「僅揭露38.2%股權，61.8%未知」）。缺失型跟矛盾型的判斷基準完全相反：
  矛盾型要防止「沒看到反駁就誤判矛盾」，缺失型要防止「沒看到揭露就
  誤判缺失」——單純這題答案沒提到分潤/法律/責任歸屬，不算觸發，只有
  文件/回答明確寫了「未揭露/不透明/不明/僅揭露X%其他未知」這類揭露
  不完整的語句才算。

人不明、權責不清、利益不明歸類為缺失型是本檔案作者依規則名稱語意
（「不明」「不清」跟「不揭露」同一類，跟「不實」不同）做的判斷，不是
文件裡逐條都有明確範例。核准時已用日羿智能報告做過真實案例查核：
利益不明找到部分佐證（跟風險不揭露共享同一個「股權未揭露」事實，非
獨立驗證），人不明跟權責不清這份文件裡完全沒有案例，維持語意推論、
未經驗證——見 veto_check_spec.md「語意分類的真實案例查核結果」。

跟數字一致性檢查、紅旗判斷、評分一樣，veto 判斷用獨立的 LLM 呼叫，
不跟其他判斷共用一次呼叫；每題單獨呼叫一次（跟紅旗判斷同一個粒度，
不像評分把多題彙總成一個 bundle），因為 veto 需要「追溯到具體證據
（哪句回答、哪個檢索片段）」，單題呼叫最容易保留這個可追溯性，也
不需要跨題比對（矛盾型/缺失型判斷的都是這一題回答+這一題檢索片段
內部的問題）。開思考模式——跟紅旗判斷/評分同一個理由，這裡的判斷
需要多步驟比對或明確揭露語句辨識，關掉思考容易走捷徑。
"""
import json
import re

from rag_common import JSON_OUTPUT_REMINDER, http_json

VETO_RULES = {
    "人不明": {
        "mode": "缺失",
        "description": "主事者不清、決策者不明、責任歸屬不明",
    },
    "資源不實": {
        "mode": "矛盾",
        "description": "只說有資源，但不能驗證",
    },
    "權責不清": {
        "mode": "缺失",
        "description": "誰做什麼、誰負責什麼不清楚",
    },
    "利益不明": {
        "mode": "缺失",
        "description": "分潤、收款、股權、費用不清楚",
    },
    "風險不揭露": {
        "mode": "缺失",
        "description": "法律、債務、糾紛、黑箱、隱性股東不揭露",
    },
}

_JSON_SCHEMA_BLOCK = (
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"is_veto_triggered": true/false, "phenomenon": "<字串或null>", '
    '"basis": "<字串或null>", "suggested_action": "<字串或null>"}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)

_CONTRADICTION_MODE_BLOCK = (
    "這一條veto規則屬於『矛盾型』判斷。先弄清楚這個任務『不是』什麼："
    "這不是在查核『系統回答有沒有正確、忠實引用檢索片段』——那是另一件"
    "事（已經有別的機制在做）。就算系統回答完全正確、逐字忠實於片段"
    "內容、每個主張都有片段依據支持，答案描述的『情境本身』仍然可能"
    "觸發veto（例如：片段記載『簡報宣稱X，財報顯示Y』，系統回答正確、"
    "忠實地轉述了這段內容，回答本身沒有錯、也完全有片段依據，但這段"
    "內容描述的『X vs. Y 對不上』這件事本身就是矛盾，一樣算觸發）。所以："
    "如果你發現系統回答忠實反映了片段內容、每個主張都找得到依據，這不"
    "代表『沒有矛盾、不觸發』，你還是要繼續判斷『回答所轉述的那個情境"
    "本身』（例如簡報宣稱的內容 vs. 財報/查核顯示的事實）是否構成主張"
    "對不上的矛盾結構。\n"
    "另外要注意：這條規則的文字定義如果講的是『只說有資源但不能驗證』，"
    "不要因此把『查核資料已經明確證實主張是假的/誇大的』排除在外——"
    "『查到反證』比『查不到依據』是更強、更明確的觸發情況，不是例外，"
    "一樣算觸發，不要因為查核資料寫得很具體、很有數字，就反過來認為"
    "『這樣就不算不能驗證，所以不觸發』。判斷重點永遠是『宣稱的內容 vs. "
    "查核到的事實』這兩者對不對得上，不是『宣稱的內容有沒有被查核』。\n"
    "先在心裡把系統回答裡實際提到的具體主張列出來，只根據這些主張判斷"
    "是否構成矛盾，不能是檢索片段全文裡剛好出現、但系統回答根本沒有"
    "提到或轉述的其他內容。只有檢索片段全文明確寫了『相反的事實』時才"
    "算真正的矛盾——片段全文只給你部分內容，沒看到反駁不等於不存在或"
    "被否定，不要因為沒看到反駁就誤判成矛盾；但也不要因為回答忠實轉述"
    "了片段、或每個主張都有依據，就放過回答內容本身描述的矛盾情境。\n"
    "如果構成觸發，輸出：\n"
    "1. phenomenon（現象）：具體觀察到的矛盾是什麼，包含引用的原始"
    "主張跟反駁的數字/說法\n"
    "2. basis（依據）：這個矛盾具體引用了回答或片段裡的哪一句話/哪個"
    "數字，不能寫「根據整體評估」這種空泛的話\n"
    "3. suggested_action（建議）：以「建議不合作」為前提，具體寫出"
    "除非對方能提供/證明什麼，否則維持不合作的建議\n"
    "如果不構成觸發，is_veto_triggered 填 false，其他欄位都填 null。"
)

_ABSENCE_MODE_BLOCK = (
    "這一條veto規則屬於『缺失型』判斷——判斷的不是『兩個主張互相矛盾』，"
    "是『文件/回答明確自己承認或指出這個主題的資訊揭露不完整』。重要"
    "規則：這一題的答案『沒有提到』這個主題，不算觸發——沒講到跟明確"
    "講了『沒講清楚/沒揭露/不透明/不明/僅揭露一部分其他未知』是兩件"
    "不同的事，只有後者才算觸發。只有回答或檢索片段裡出現類似『未揭露』"
    "『不透明』『不明』『尚待確認』『僅揭露X%，其餘未知/不明』這類明確"
    "指出資訊不完整的語句時，才判定為觸發；單純這一題沒有涉及這個主題，"
    "或這一題的內容講得清楚、沒有這類語句，都不算觸發，不要因為答案"
    "沒有主動提到這個主題就假設它有缺失。\n"
    "如果構成觸發，輸出：\n"
    "1. phenomenon（現象）：具體是什麼資訊沒有揭露完整，包含引用原始"
    "數字/說法（例如揭露了多少比例、還剩多少不明）\n"
    "2. basis（依據）：具體引用回答或片段裡『明確承認揭露不完整』的"
    "那句話，不能寫「根據整體評估」這種空泛的話\n"
    "3. suggested_action（建議）：以「建議不合作」為前提，具體寫出"
    "除非對方能補充/揭露什麼，否則維持不合作的建議\n"
    "如果不構成觸發，is_veto_triggered 填 false，其他欄位都填 null。"
)


def _build_veto_system_prompt(rule_name: str) -> str:
    rule = VETO_RULES[rule_name]
    mode_block = _CONTRADICTION_MODE_BLOCK if rule["mode"] == "矛盾" else _ABSENCE_MODE_BLOCK
    return (
        "你是嚴謹的盡職調查分析師。你會看到一則問答的問題、系統回答，以及"
        "檢索到的文件片段全文。你的任務是判斷這一題的內容有沒有觸發"
        "《PSF六壬合夥生態系統》第十四節「五不合作」裡的「"
        f"{rule_name}」這一條——定義是「{rule['description']}」。\n"
        f"{mode_block}\n"
        f"{JSON_OUTPUT_REMINDER}\n"
        f"{_JSON_SCHEMA_BLOCK}"
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _not_triggered(parse_error):
    return {
        "is_veto_triggered": False,
        "phenomenon": None,
        "basis": None,
        "suggested_action": None,
        "parse_error": parse_error,
    }


def parse_veto_response(raw: str) -> dict:
    """把模型輸出解析成veto判斷結果。is_veto_triggered=False 是正常判斷
    （不是錯誤），parse_error 只在真正解析失敗，或 is_veto_triggered=true
    卻缺欄位時才有值——後者視同抽取失敗，不半殘顯示一個沒有依據的veto
    （跟 red_flag_detection.py 的 is_red_flag=true 防呆同一個原則）。"""
    cleaned = _strip_fence(_strip_think(raw))
    try:
        parsed = json.loads(cleaned)
        is_triggered = bool(parsed.get("is_veto_triggered"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return _not_triggered(f"veto判斷結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    if not is_triggered:
        return _not_triggered(None)

    phenomenon = parsed.get("phenomenon")
    basis = parsed.get("basis")
    suggested_action = parsed.get("suggested_action")

    if not phenomenon or not basis or not suggested_action:
        return _not_triggered(
            f"is_veto_triggered=true 但欄位不完整：{parsed}"
        )

    return {
        "is_veto_triggered": True,
        "phenomenon": phenomenon,
        "basis": basis,
        "suggested_action": suggested_action,
        "parse_error": None,
    }


def detect_veto(rule_name, question, answer, context, chat_url, chat_model, timeout=180):
    """對一題的回答＋檢索片段全文，判斷是否觸發指定的veto規則。永遠回傳
    一個結構一致的 dict（含 "rule" 欄位），不拋例外。rule_name 必須是
    VETO_RULES 裡的合法規則名稱——呼叫端（generate_report.py）負責在跑
    報告前就驗證問題清單裡的規則名稱合法，這裡直接假設合法、用
    VETO_RULES[rule_name] 查詢，錯字會在更早的階段被擋下。"""
    system_prompt = _build_veto_system_prompt(rule_name)
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
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
        result = _not_triggered(f"呼叫veto判斷模型失敗 (HTTP {status}): {resp}")
        result["rule"] = rule_name
        return result
    try:
        raw = resp["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        result = _not_triggered(f"veto判斷模型回應格式不如預期: {resp}")
        result["rule"] = rule_name
        return result

    result = parse_veto_response(raw)
    result["rule"] = rule_name
    return result


def summarize_veto_results(veto_entries):
    """veto_entries: [{"question": str, "rule": str, "detection": dict,
    "heading": str|None}, ...]，detection 是 detect_veto() 的回傳值。

    依 VETO_RULES 的固定順序（人不明→資源不實→權責不清→利益不明→
    風險不揭露）分組，只保留實際有被標記測試過的規則；同一條規則可能
    被多題各自標記、各自呼叫，"triggered" 是這條規則底下任何一次呼叫
    is_veto_triggered 為真的 OR 結果，"entries" 只放觸發的那些次呼叫
    （附上對應的 question/heading，方便報告追溯）。純邏輯，不牽涉 LLM，
    回傳值用 dict 保留固定的五條規則順序，跟 dimension_results 用維度
    名稱當 key 的模式一致。
    """
    by_rule = {}
    for entry in veto_entries:
        by_rule.setdefault(entry["rule"], []).append(entry)

    results = {}
    for rule_name in VETO_RULES:
        if rule_name not in by_rule:
            continue
        entries = by_rule[rule_name]
        triggered_entries = []
        for entry in entries:
            det = entry["detection"]
            if det["is_veto_triggered"]:
                triggered_entries.append({
                    "question": entry["question"],
                    "heading": entry["heading"],
                    "phenomenon": det["phenomenon"],
                    "basis": det["basis"],
                    "suggested_action": det["suggested_action"],
                })
        results[rule_name] = {
            "triggered": bool(triggered_entries),
            "entries": triggered_entries,
        }
    return results
