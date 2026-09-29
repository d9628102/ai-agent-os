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

from marketing_faithfulness import _norm_for_quote, load_first_json_object, unbacked_quote_pieces
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


# 兩個模式區塊裡「建議」欄位的前提句——五不合作的後果是「建議不合作」。
# 模板自帶的 veto 規則（例如深科技模板的 Gate 3）後果不一樣（「完成智財
# 歸屬確認前不可投資」），只替換這兩句，模式區塊其餘判斷邏輯完全沿用，
# 不另起一套。
_DEFAULT_ACTION_CLAUSES = {
    "矛盾": "以「建議不合作」為前提，具體寫出除非對方能提供/證明什麼，否則維持不合作的建議",
    "缺失": "以「建議不合作」為前提，具體寫出除非對方能補充/揭露什麼，否則維持不合作的建議",
}


def _mode_block_with_consequence(mode: str, consequence: str) -> str:
    """回傳把「建議」前提句換成指定後果的模式區塊。原句找不到時丟錯，
    不默默送出一個還寫著「建議不合作」的 prompt——模式區塊之後被改寫、
    前提句對不上時，要在這裡立刻炸出來。"""
    block = _CONTRADICTION_MODE_BLOCK if mode == "矛盾" else _ABSENCE_MODE_BLOCK
    clause = _DEFAULT_ACTION_CLAUSES[mode]
    if clause not in block:
        raise ValueError(f"{mode}型模式區塊裡找不到預設的建議前提句，無法替換成「{consequence}」")
    verb = "提供/證明" if mode == "矛盾" else "補充/揭露"
    return block.replace(
        clause, f"以「{consequence}」為前提，具體寫出除非對方能{verb}什麼，否則維持「{consequence}」的建議",
    )


def _build_veto_system_prompt(rule_name: str, rules: dict = None) -> str:
    """rules 是 None 時用五不合作（VETO_RULES），組出的 prompt 跟加入模板
    規則之前逐字相同。rules 是模板自帶的規則（深科技的 Gate 3）時，改用
    規則自己的 source_label／原文判準／後果，判斷邏輯（矛盾型/缺失型模式
    區塊）照舊。"""
    if rules is None:
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

    rule = rules[rule_name]
    mode_block = _mode_block_with_consequence(rule["mode"], rule["consequence"])
    verbatim = rule.get("criterion_verbatim")
    verbatim_line = (f"原文判準（出自範例案子，公司名稱以本案為準）：「{verbatim}」\n"
                     if verbatim else "")
    return (
        "你是嚴謹的盡職調查分析師。你會看到一則問答的問題、系統回答，以及"
        "檢索到的文件片段全文。你的任務是判斷這一題的內容有沒有觸發"
        f"{rule['source_label']}的「{rule_name}」這一條——定義是「{rule['description']}」。\n"
        f"{verbatim_line}"
        f"{mode_block}\n"
        f"{JSON_OUTPUT_REMINDER}\n"
        f"{_JSON_SCHEMA_BLOCK}"
    )


# ---------------------------------------------------------------------------
# 「有引文驗證的缺失型」判斷（規則設定 verify_quote=true 才走這條路，目前只有
# 深科技模板的 Gate 3）。沒有這個旗標的規則（所有五不合作）完全走上面原本的
# prompt、解析與顯示，一個字都不動。
#
# 為什麼需要：問答系統（rag_answer.py）被要求「文件沒寫就回答『文件中未提及』
# 並說明相關內容為何不足以回答」，而原本的缺失型 prompt 把「回答或檢索片段裡出現
# 『未揭露』『不明』…」都當證據，於是沉默被讀成「承認不完整」（2026-09-29 用
# PSF EIM 公開文件實測：4 個沉默案例 2 個誤觸發）。做法是三態＋逐字引文＋程式
# 驗證：
#   admitted  文件片段裡有一句話表示資訊目前不完整，quote 是那句話（逐字）
#   silent    文件片段沒談到這個主題（不觸發，但報告要標「需人工確認」）
#   clear     文件談到了、而且資訊完整
# 「admitted」必須附一句程式驗證過確實出現在檢索片段裡的引文，驗證不過就降級成
# silent——所以最壞的漏判結果是「⏸ 需人工確認」，不是「✅ 未觸發」。
# ---------------------------------------------------------------------------

STATUS_ADMITTED = "admitted"
STATUS_SILENT = "silent"
STATUS_CLEAR = "clear"
STATUS_UNDETERMINED = "undetermined"
_MODEL_STATUSES = (STATUS_ADMITTED, STATUS_SILENT, STATUS_CLEAR)

# 引文正規化（去空白、標點、Markdown *）之後至少要有這麼多字才算「可驗證」，
# 避免「未揭露」這種兩三個字的片段輕易在文件任何地方湊巧比對成功。
MIN_QUOTE_CHARS = 6

REASON_NO_QUOTE = "模型判為承認不完整，但沒有給出足夠長的引文，無法驗證"
REASON_QUOTE_NOT_IN_CONTEXT = "模型判為承認不完整，但引用的句子不在檢索到的文件片段裡（可能來自系統回答、改寫或摘要）"

_VERIFIED_ABSENCE_PROMPT = (
    "你是嚴謹的盡職調查分析師。你會看到一則問答的問題、系統回答，以及檢索到的『文件片段全文』。"
    "你的任務是判斷『文件本身』有沒有表示下面這項資訊目前不完整。\n"
    "規則名稱：{name}；定義：{desc}\n"
    "重要：『系統回答』是另一個模型讀完文件片段後寫的。它寫「文件中未提及」「未說明」只代表『檢索到的片段沒有涵蓋這個主題』，"
    "**不是文件承認資訊不完整**，絕對不能拿系統回答裡的話當證據。證據只能來自『文件片段全文』。\n"
    "請三選一，輸出 status：\n"
    "- \"admitted\"：文件片段裡有一句話，表示這項資訊『目前是不完整的』。下面三種都算：\n"
    "  (a) 直接寫未揭露／不明／未具名／尚未取得／尚待確認／有待補充；\n"
    "  (b) 文件把這項資訊列為『必須在投資前釐清的問題』，或列為『要求對方取得／補充／提供』的項目（要求取得，就表示現在還沒有）；\n"
    "  (c) 只揭露一部分、其餘未知（例如「僅揭露 38.2%，其餘不明」）。\n"
    "  此時 quote 必須是表達這一點的那一句話，從文件片段逐字複製，不能改寫、不能摘要、不能翻譯、不能把不同地方的字拼在一起。\n"
    "- \"silent\"：文件片段根本沒有談到這個主題（沒寫，也沒說它不完整、待釐清或要求補充）。這不算觸發，quote 填 null。\n"
    "- \"clear\"：文件片段談到了這個主題，而且資訊看起來完整，沒有說不完整。quote 填 null。\n"
    "判斷時只看『文件片段全文』：文件有沒有一句話把這個主題說成『還沒有／還不清楚／要去取得』？沒有就是 silent 或 clear，不要自己推論。\n"
    "status 是 admitted 時，另外輸出：phenomenon（一句話說明具體缺了什麼）、"
    "suggested_action（以「{consequence}」為前提，具體寫出除非對方能補充／取得什麼，否則維持「{consequence}」的建議）；其他情況這兩個欄位填 null。\n"
    "{json_reminder}\n"
    "請只輸出一個 JSON 物件：{{\"status\": \"admitted\"|\"silent\"|\"clear\", \"quote\": \"<逐字引文或null>\", "
    "\"phenomenon\": \"<一句話或null>\", \"suggested_action\": \"<字串或null>\"}}\n"
    "不要有任何其他文字、不要用 markdown code fence。"
)


def _build_verified_absence_prompt(rule_name: str, rules: dict) -> str:
    """有引文驗證的缺失型 system prompt。rules 是模板自帶的規則（含 description、
    consequence）。純邏輯。"""
    rule = rules[rule_name]
    return _VERIFIED_ABSENCE_PROMPT.format(
        name=rule_name, desc=rule["description"], consequence=rule["consequence"],
        json_reminder=JSON_OUTPUT_REMINDER,
    )


def verify_quote_in_context(quote, context: str):
    """檢查 quote 是不是（逐字）出現在檢索片段全文裡。回傳 (通過與否, 原因或None,
    找不到的片段清單)。沿用行銷文案 Guard E 的比對：引文在標點處拆成片段逐段比對，
    正規化時去掉空白、標點與 Markdown 的 `*`。純邏輯。"""
    if not isinstance(quote, str) or len(_norm_for_quote(quote)) < MIN_QUOTE_CHARS:
        return False, REASON_NO_QUOTE, []
    missing = unbacked_quote_pieces(quote, _norm_for_quote(context))
    if missing:
        return False, REASON_QUOTE_NOT_IN_CONTEXT, missing
    return True, None, []


def _undetermined(parse_error, rule=None):
    return {
        "status": STATUS_UNDETERMINED, "is_veto_triggered": False, "phenomenon": None, "basis": None,
        "suggested_action": None, "quote": None, "downgrade_reason": None, "missing_pieces": [],
        "parse_error": parse_error,
    }


def parse_verified_absence_response(raw: str, context: str, consequence: str) -> dict:
    """把三態判斷的模型輸出解析並驗證。永遠回傳結構一致的 dict：
    - JSON 解析失敗、不是物件、status 不是三個合法值之一 → undetermined（帶 parse_error）
    - admitted 但引文缺漏／太短／不在檢索片段裡 → 降級成 silent（downgrade_reason 記原因）
    - admitted 且引文驗證通過 → 觸發（basis 是「引文」，phenomenon／suggested_action
      缺漏時補上預設文字，不因為欄位沒填就丟掉一個驗證過的引文）
    純邏輯，不牽涉 LLM。"""
    try:
        parsed = load_first_json_object(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _undetermined(f"veto判斷結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")
    if not isinstance(parsed, dict):
        return _undetermined(f"veto判斷結果不是 JSON 物件：{str(parsed)[:300]}")

    status = parsed.get("status")
    status = status.strip().lower() if isinstance(status, str) else status
    if status not in _MODEL_STATUSES:
        return _undetermined(f"veto判斷結果的 status 不是 admitted／silent／clear：{str(parsed)[:300]}")

    result = {
        "status": status, "is_veto_triggered": False, "phenomenon": None, "basis": None,
        "suggested_action": None, "quote": None, "downgrade_reason": None, "missing_pieces": [],
        "parse_error": None,
    }
    if status != STATUS_ADMITTED:
        return result

    quote = parsed.get("quote")
    ok, reason, missing = verify_quote_in_context(quote, context)
    if not ok:
        result.update({"status": STATUS_SILENT, "downgrade_reason": reason, "missing_pieces": missing,
                       "quote": quote if isinstance(quote, str) else None})
        return result

    result.update({
        "is_veto_triggered": True,
        "quote": quote.strip(),
        "basis": f"「{quote.strip()}」",
        "phenomenon": parsed.get("phenomenon") or "（模型未說明具體現象，請對照引文）",
        "suggested_action": parsed.get("suggested_action") or f"在補齊資訊之前，維持「{consequence}」",
    })
    return result


def detection_status(detection: dict) -> str:
    """一次 veto 判斷結果的狀態：triggered／undetermined／silent／clear。
    新格式（有 status）直接對應；舊格式（五不合作，沒有 status）：is_veto_triggered
    為真＝觸發；沒觸發但有 parse_error＝無法判定（原本會被當成「未觸發」，報告顯示
    ✅，是假放心）；其餘＝clear。純邏輯。"""
    status = detection.get("status")
    if status == STATUS_ADMITTED or (status is None and detection.get("is_veto_triggered")):
        return "triggered"
    if status == STATUS_UNDETERMINED or (status is None and detection.get("parse_error")):
        return STATUS_UNDETERMINED
    if status == STATUS_SILENT:
        return STATUS_SILENT
    return STATUS_CLEAR


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


def detect_veto(rule_name, question, answer, context, chat_url, chat_model, timeout=180,
                rules=None):
    """對一題的回答＋檢索片段全文，判斷是否觸發指定的veto規則。永遠回傳
    一個結構一致的 dict（含 "rule" 欄位），不拋例外。rule_name 必須是
    VETO_RULES 裡的合法規則名稱——呼叫端（generate_report.py）負責在跑
    報告前就驗證問題清單裡的規則名稱合法，這裡直接假設合法、用
    VETO_RULES[rule_name] 查詢，錯字會在更早的階段被擋下。rules 是模板
    自帶的規則（見 _build_veto_system_prompt()）；None 時用五不合作。

    規則設定有 verify_quote=true 時（目前只有深科技的 Gate 3）改走三態＋引文驗證
    （見 _detect_verified_absence()）；其餘規則走下面原本的路徑，回傳的 dict 跟
    加入這個旗標之前完全相同。"""
    active_rules = VETO_RULES if rules is None else rules
    if active_rules[rule_name].get("verify_quote"):
        return _detect_verified_absence(rule_name, question, answer, context, chat_url, chat_model,
                                        timeout, active_rules)
    system_prompt = _build_veto_system_prompt(rule_name, rules)
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


def _detect_verified_absence(rule_name, question, answer, context, chat_url, chat_model, timeout, rules):
    """三態＋引文驗證的單題判斷。輸入與呼叫參數（溫度 0、思考模式、2048 tokens）
    跟原本的缺失型判斷相同，只有 prompt 與解析不同。永遠回傳 dict（含 "rule"），
    不拋例外。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": _build_verified_absence_prompt(rule_name, rules)},
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
        result = _undetermined(f"呼叫veto判斷模型失敗 (HTTP {status}): {resp}")
    else:
        try:
            raw = resp["choices"][0]["message"].get("content") or ""
            result = parse_verified_absence_response(raw, context, rules[rule_name]["consequence"])
        except (KeyError, IndexError, TypeError):
            result = _undetermined(f"veto判斷模型回應格式不如預期: {resp}")
    result["rule"] = rule_name
    return result


def summarize_veto_results(veto_entries, rules=None):
    """veto_entries: [{"question": str, "rule": str, "detection": dict,
    "heading": str|None}, ...]，detection 是 detect_veto() 的回傳值。

    依 VETO_RULES 的固定順序（人不明→資源不實→權責不清→利益不明→
    風險不揭露）分組，只保留實際有被標記測試過的規則；同一條規則可能
    被多題各自標記、各自呼叫，"triggered" 是這條規則底下任何一次呼叫
    is_veto_triggered 為真的 OR 結果，"entries" 只放觸發的那些次呼叫
    （附上對應的 question/heading，方便報告追溯）。純邏輯，不牽涉 LLM，
    回傳值用 dict 保留固定的五條規則順序，跟 dimension_results 用維度
    名稱當 key 的模式一致。rules 是模板自帶的規則時，改依模板規則的
    順序；None 時用五不合作的固定順序。

    每條規則的結果除了 "triggered"／"entries"，還有 "status"（triggered／
    undetermined／silent／clear）與對應的 silent_entries／undetermined_entries。
    判斷結果沒有 status 的舊格式（五不合作）：沒觸發但有 parse_error 的，現在
    算「無法判定」，不再默默當成「未觸發」。
    """
    by_rule = {}
    for entry in veto_entries:
        by_rule.setdefault(entry["rule"], []).append(entry)

    results = {}
    for rule_name in (VETO_RULES if rules is None else rules):
        if rule_name not in by_rule:
            continue
        entries = by_rule[rule_name]
        triggered_entries, silent_entries, undetermined_entries = [], [], []
        for entry in entries:
            det = entry["detection"]
            state = detection_status(det)
            if state == "triggered":
                triggered_entries.append({
                    "question": entry["question"],
                    "heading": entry["heading"],
                    "phenomenon": det["phenomenon"],
                    "basis": det["basis"],
                    "suggested_action": det["suggested_action"],
                })
            elif state == STATUS_SILENT:
                silent_entries.append({
                    "question": entry["question"], "heading": entry.get("heading"),
                    "reason": det.get("downgrade_reason"),
                })
            elif state == STATUS_UNDETERMINED:
                undetermined_entries.append({
                    "question": entry["question"], "heading": entry.get("heading"),
                    "reason": (det.get("parse_error") or "")[:160],
                })
        # 規則的狀態：觸發 > 無法判定 > 文件未提及 > 未觸發。沉默與無法判定都不是
        # 「未觸發」——報告要標「需人工確認」，不能顯示成 ✅。
        if triggered_entries:
            rule_state = "triggered"
        elif undetermined_entries:
            rule_state = STATUS_UNDETERMINED
        elif silent_entries:
            rule_state = STATUS_SILENT
        else:
            rule_state = STATUS_CLEAR
        results[rule_name] = {
            "triggered": bool(triggered_entries),
            "entries": triggered_entries,
            "status": rule_state,
            "silent_entries": silent_entries,
            "undetermined_entries": undetermined_entries,
        }
    return results
