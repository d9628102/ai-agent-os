#!/usr/bin/env python3
"""
marketing_faithfulness.py

行銷文案的正向忠實度檢查——跟 red_flag_detection.py 找的是不同性質的東西：
紅旗判斷找的是「主張 vs. 查核結果對不上」的矛盾結構；這裡沒有矛盾結構可以
比對，判斷的是「文案裡的賣點主張，有沒有在來源文件裡找到具體依據支撐」——
「有沒有撐住」不是「有沒有說謊」，這裡的 prompt 是重新設計的，不是套用紅旗
判斷的矛盾定義改個說法。

跟 scoring.py/veto_check.py 同一套技術路線：獨立的 LLM 呼叫、structured
JSON 抽取、開思考模式（需要逐條比對賣點跟來源文件，關掉思考容易走捷徑）。

每一句先分類成 fact（可查證的事實主張）或 rhetoric（主觀修辭），只有 fact
要判斷有沒有依據。實測過不分類的版本：「讓組織像神經系統般協同運作」「驅動
企業邁向無限可能的未來」這種修辭連續三次被判成「無依據」，理由是「來源文件
沒有這個說法」——模型把「文件裡有沒有這個字句」當成判斷標準，等於把行銷
修辭一律當成捏造。rhetoric 也要列出來，不是默默略過，讓人工可以核對分類本身
有沒有被濫用（把捏造的事實歸成修辭來閃避檢查）。比較級/最高級（業界領先、
第一、唯一）刻意歸在 fact：它們是可以被查證的排名主張，不是純粹的語氣。
"""
import json
import re

from rag_common import JSON_OUTPUT_REMINDER, http_json

FAITHFULNESS_SYSTEM_PROMPT = (
    "你是嚴謹的行銷文案事實查核員。你會看到一份行銷文案全文，以及檢索到的"
    "來源文件片段全文。\n"
    "第一步：把文案逐句拆開（標題、每一條、每一段都要），每一句歸類成兩種"
    "之一，不能漏掉任何一句。claim 欄位必須逐字照抄文案原文的那一句（或那"
    "一段），不要改寫、不要摘要、不要換標點——之後會用逐字比對確認每一行"
    "文案都有被檢查到：\n"
    "- fact（事實主張）：包含任何可以被查證的內容——數字、客戶數/案例數、"
    "客戶或合作夥伴名稱、具體功能/模組/能力、認證/獎項、時程、以及比較級或"
    "最高級的排名主張（例如「業界領先」「第一」「唯一」「最完整」）\n"
    "- rhetoric（主觀修辭）：沒有任何可查證內容的願景、比喻、情緒性語句，"
    "例如「打造企業智慧新紀元」「驅動無限可能的未來」「像神經系統般協同"
    "運作」。只要句子裡含有上面 fact 列出的任何一種內容，就不是 rhetoric\n"
    "- 描述產品、模組、系統或 AI Agent「能做什麼、會帶來什麼效果」的句子"
    "（例如「自動產生每週營運報表」「即時偵測異常交易，守護每一筆資金」「讓"
    "客服全天候秒回」）一律是 fact，即使句子寫得很有修辭感——這類句子是在陳述產品"
    "能力，必須查核\n"
    "- 引用來源的歸因句一律是 fact：「根據某某的調查/研究/觀察/數據」「某某"
    "指出」「研究顯示」「業界普遍認為」這類句子，主張的是『有某個來源說過這件"
    "事』，必須在來源文件裡找得到那個來源跟那段內容，找不到就是無依據——"
    "例如「據內部統計，客戶留存率逐年攀升」「專家指出導入後效率倍增」\n"
    "- 拿不準是 fact 還是 rhetoric 時，一律歸為 fact。把事實誤歸成修辭會讓它"
    "完全不經查核就通過，後果比把修辭誤歸成事實嚴重得多\n"
    "如果一句話同時混有修辭跟事實（例如「以五大產品矩陣打造企業智慧新紀元」），"
    "把事實部分抽出來當成一條 fact，修辭部分另列一條 rhetoric。\n"
    "第二步：只對 fact 判斷來源文件片段裡有沒有具體依據支撐。rhetoric 不做"
    "依據判斷——來源文件裡沒有出現同樣的修辭字句，不代表這句修辭是捏造，"
    "不要因為「文件沒有這個說法」就把修辭判成無依據。\n"
    "重要規則——先弄清楚這個任務『不是』什麼：這不是判斷這條主張是不是假的、"
    "是不是誇大，只判斷來源文件裡有沒有明確記載可以支撐這條事實主張的內容"
    "——是「有沒有撐住」不是「有沒有說謊」。\n"
    "1. 只能根據提供的來源文件片段判斷，不能用你自己的常識或對這類產品/"
    "服務的既有印象去補充沒有寫在文件裡的依據\n"
    "2. fact 且 supported=true 時，evidence 必須具體引用來源文件片段裡的"
    "哪一句/哪個數字，不能寫「文件裡有提到相關內容」這種空泛的話。引用原文"
    "時用「」框起來、逐字照抄——引文會被逐字比對，來源裡找不到的引文會讓"
    "這條被判成無依據\n"
    "3. fact 且 supported=false 時，reason_unsupported 必須具體說明來源文件"
    "片段裡完全沒有提到這件事、或提到的內容跟這條主張對不上\n"
    "4. 一句話裡有好幾個主張時，每一個都要找得到依據，整句才算 supported=true；"
    "只要有一部分找不到依據，整句就是 supported=false，reason_unsupported 要"
    "指出是哪一部分——不能因為句子裡有些名詞在文件裡出現過，就整句放行\n"
    "5. 來源文件寫的是願景、目標、計畫，文案卻寫成已經達成的現況，算"
    "supported=false（例如文件寫『計畫明年進軍日本市場』，文案寫『已在日本"
    "市場大獲好評』）。特別注意「正在」「已經」「目前」「已服務」「持續幫助」"
    "這類描述現況、已發生的用詞：文件必須明確記載這是實際發生的現況（實際"
    "客戶、實際導入、實際成果），文件只有願景、口號、使命宣言時，不能當成"
    "現況主張的依據\n"
    "6. rhetoric 的 supported、evidence、reason_unsupported 一律填 null\n"
    f"{JSON_OUTPUT_REMINDER}\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"claims": [{"claim": "<逐字照抄的原句>", "kind": "<fact或rhetoric>", '
    '"supported": <true/false/null>, "evidence": "<字串或null>", '
    '"reason_unsupported": "<字串或null>"}, ...]}\n'
    "不要有任何其他文字、不要用 markdown code fence。"
)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _strip_fence(text: str) -> str:
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def load_first_json_object(raw: str):
    """剝掉推理區塊跟 code fence 之後解析 JSON。整段解析不了時，依序改試：
    文字中間任何一個 ```json 區塊、從第一個「{」開始的第一個完整物件。
    全部失敗才拋 json.JSONDecodeError（ValueError 子類別）。純邏輯。

    由來：說服力評分對一份滿是「%」的文案，temperature 0 下三次都輸出兩份
    JSON——先一份裸的、但少了最外層的結尾 }（跟 JSON_OUTPUT_REMINDER 記錄
    的百分號/括號 bug 同一家族），接著模型自己再輸出一份包在 ```json 裡、
    括號完整的。只剝頭尾 fence 的舊寫法整段解析失敗，評分直接失敗、重試也
    救不回來。三個解析函式共用這個入口。"""
    body = _strip_think(raw)
    cleaned = _strip_fence(body)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as first_error:
        for block in _FENCED_BLOCK_RE.findall(body):
            try:
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue
        start = cleaned.find("{")
        if start < 0:
            raise first_error
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return obj


INCOMPLETE_JUDGMENT_REASON = (
    "查核結果欄位不完整（模型沒有給出具體依據或理由），無法確認有依據——"
    "視同無依據，需人工核對"
)


def _failed_claims(parse_error):
    return {"claims": None, "malformed_count": 0, "uncovered_lines": [], "parse_error": parse_error}


_NORMALIZE_DROP = set(" \t\r\n　「」『』\"'“”‘’")
_TRAILING_PUNCT = "。，、！？；：.,!?;:"


def normalize_claim_text(text: str) -> str:
    return "".join(ch for ch in text if ch not in _NORMALIZE_DROP).strip(_TRAILING_PUNCT)


def find_uncovered_lines(copy_text: str, claims) -> list:
    """找出文案裡沒有任何一條 claim 逐字落在其中的行——也就是檢查器整行
    漏掉、根本沒檢查到的內容。純邏輯。

    由來：prompt 明文要求「不能漏掉任何一句」，實測一份故事型文案的標題
    （含「終極方案」這種最高級主張）還是整行沒出現在查核結果裡，等於完全沒
    經過檢查就通過。跟孤立 think 標籤同一個教訓：prompt 規則約束不了 100%，
    漏檢要用機械方式攔。比對是逐字子字串（去掉空白、引號、句尾標點），模型
    改寫過的 claim 會比對不到、被當成漏檢——這是故意偏向誤報的方向：多一條
    需要人工看的警告，比漏掉一條沒檢查的主張安全。"""
    normalized_claims = [normalize_claim_text(c["claim"]) for c in claims or []]
    normalized_claims = [c for c in normalized_claims if c]
    uncovered = []
    for line in copy_text.splitlines():
        norm_line = normalize_claim_text(line)
        if not norm_line:
            continue
        if not any(c in norm_line for c in normalized_claims):
            uncovered.append(line.strip())
    return uncovered


ALERT_RED = "red"
ALERT_YELLOW = "yellow"


def _unsupported(claim: str, reason: str, alert: str = ALERT_RED) -> dict:
    """無依據的事實主張。alert 分兩級：red＝來源找不到支撐、或守門規則判定
    捏造；yellow＝模型拿文案自己的措辭當依據（循環引用），依據無法自動核實。
    兩級都算「不能直接交出去」，分開只是為了讓人一眼分辨嚴重程度，不要因為
    循環引用的誤報太多，連真的捏造也被習慣性略過。"""
    return {"claim": claim, "kind": "fact", "supported": False,
            "evidence": None, "reason_unsupported": reason, "alert": alert}


def parse_faithfulness_response(raw: str) -> dict:
    """把模型輸出解析成忠實度檢查結果。整體 JSON 解析失敗、或 claims 不是
    一個清單時，視為整批抽取失敗（parse_error 有值、claims=None）。

    單一一條的判斷不完整時一律 fail-closed，不丟掉：事實主張缺 supported、
    supported=true 卻缺 evidence、supported=false 卻缺 reason_unsupported、
    或 kind 不是合法值，都保留下來並當成「無依據、需人工核對」。原本的做法是
    丟掉那一條，但那樣一條 supported=false 卻漏寫理由的主張會直接消失，
    沒有依據的賣點反而看起來乾淨——跟「沒依據的賣點一定要被看見」的要求
    正好相反。連 claim 文字都沒有的項目無法顯示，只能計數（malformed_count），
    交給呼叫端把它當成需要注意的狀況。"""
    try:
        parsed = load_first_json_object(raw)
        claims = parsed.get("claims")
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return _failed_claims(f"忠實度檢查結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    if not isinstance(claims, list):
        return _failed_claims(f"claims 不是清單：{parsed}")

    cleaned_claims = []
    malformed_count = 0
    for c in claims:
        claim = c.get("claim") if isinstance(c, dict) else None
        if not isinstance(claim, str) or not claim.strip():
            malformed_count += 1
            continue

        if c.get("kind") == "rhetoric":
            cleaned_claims.append({
                "claim": claim, "kind": "rhetoric",
                "supported": None, "evidence": None, "reason_unsupported": None, "alert": None,
            })
            continue

        supported = c.get("supported")
        evidence = c.get("evidence")
        reason_unsupported = c.get("reason_unsupported")
        complete = c.get("kind") == "fact" and (
            (supported is True and evidence) or (supported is False and reason_unsupported)
        )
        if not complete:
            cleaned_claims.append(_unsupported(claim, INCOMPLETE_JUDGMENT_REASON))
            continue

        if supported:
            cleaned_claims.append({
                "claim": claim, "kind": "fact", "supported": True,
                "evidence": evidence, "reason_unsupported": None, "alert": None,
            })
        else:
            cleaned_claims.append(_unsupported(claim, reason_unsupported))

    # 實測模型處理「修辭+事實混在一句」時，偶爾不拆開、而是把整句同時列成
    # fact 跟 rhetoric。fact 那條照樣會被查核，只是報告裡同一句同時顯示
    # 「無依據」跟「修辭」會讓人困惑，所以丟掉跟 fact 完全重複的 rhetoric。
    fact_texts = {c["claim"].strip() for c in cleaned_claims if c["kind"] == "fact"}
    cleaned_claims = [
        c for c in cleaned_claims
        if not (c["kind"] == "rhetoric" and c["claim"].strip() in fact_texts)
    ]

    return {"claims": cleaned_claims, "malformed_count": malformed_count,
            "uncovered_lines": [], "parse_error": None}


_RISKY_RHETORIC_RE = re.compile(
    r"最|第一|唯一|領先|頂尖|首創|獨家|終極|全球|全台|全國|百大|冠軍|No\.?\s*1"
    r"|\d|(?<!每)[一二三四五六七八九十百千萬億兩]+\s*(成|倍|家|年|項|%|％)"
    r"|數以[萬千百]計|(?:數|上|近|逾|破)(?:十|百|千)?(?:萬|千|百|億)"
)
# 模糊量詞（數萬、數十萬、上千、近百…）也是數字主張，要有來源。刻意要求前面
# 帶「數/上/近/逾/破」，不然「千萬別錯過」「萬無一失」這種慣用語會被誤判。
_VAGUE_QUANTITY_RE = re.compile(
    r"數以[萬千百]計|(?:數|上|近|逾|破)(?:十|百|千)?(?:萬|千|百|億)"
)
RECLASSIFIED_REASON = (
    "模型把這句歸成主觀修辭，但句中含有比較級/最高級、規模用詞或數字，"
    "不能當成修辭直接放行——視同無依據，需人工核對或補上依據"
)
_SUPERLATIVE_RE = re.compile(
    r"最|第一|唯一|領先|頂尖|首創|獨家|終極|冠軍|百大|全球|全台|全國|No\.?\s*1"
)
_QUOTED_RE = re.compile(r"[「『“\"]([^」』”\"]{2,})[」』”\"]")
_CITED_AFTER_LABEL_RE = re.compile(r"^文件(?:片段)?\s*\d+\s*[：:]\s*([^，,。]+)")
_EVIDENCE_SPLIT_RE = re.compile(r"[；;\n]|(?=文件(?:片段)?\s*\d+)")
_QUOTE_NORMALIZE_RE = re.compile(r"[\s「」『』“”\"'‘’（）()。，、：:；;.™]")


def _norm_for_quote(text: str) -> str:
    return _QUOTE_NORMALIZE_RE.sub("", text)


_QUOTE_PIECE_SPLIT_RE = re.compile(r"\.{2,}|…+|⋯+|[、，,。：:；;（）()「」『』\n]")


def unbacked_quote_pieces(segment: str, normalized_source: str) -> list:
    """引文在標點處拆成片段逐段比對，回傳來源裡找不到的片段。實測模型引用
    時會用「...」省略中間內容、把清單裡不相鄰的項目串成一句（「Shared
    Intelligence、Shared Learning」）、甚至把標題跟下一行接在一起（「Service
    Matrix™（原 Black Concierge）Core Modules：AI Concierge」）——整段比對會
    把這些內容正確的引用誤判成捏造。拆開後只放棄「片段彼此相鄰」這件事的
    檢查；捏造的引文通常是一整個不存在的詞組（「無需自建複雜系統」「工業化
    方式」），照樣抓得到。純邏輯。"""
    pieces = [p for p in _QUOTE_PIECE_SPLIT_RE.split(segment) if len(_norm_for_quote(p)) >= 2]
    return [p.strip() for p in pieces if _norm_for_quote(p) not in normalized_source]


def cited_segments(evidence: str) -> list:
    """從 evidence 裡抽出「模型當成原文引用」的片段：「」『』等引號框起來的
    文字，以及「文件片段N：」後面緊接的那段（到第一個逗號/句號為止）。
    純邏輯。敘述式的說明（「文件2提到企業永遠不遺忘」）不是引文、不抽。"""
    segments = list(_QUOTED_RE.findall(evidence))
    for part in _EVIDENCE_SPLIT_RE.split(evidence):
        m = _CITED_AFTER_LABEL_RE.match(part.strip())
        if m:
            segments.append(m.group(1))
    return [s for s in segments if _norm_for_quote(s)]


_HEDGED_EVIDENCE_RE = re.compile(r"間接|推論|推測|隱含|暗示|可延伸|可理解為|形成.{0,4}對應")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
# rag_answer.build_context() 幫每個片段加的編號標籤，本身不是來源內容——不先
# 拿掉的話，1～8 這幾個數字永遠「出現在來源裡」，捏造的個位數會直接過關。
_FRAGMENT_LABEL_RE = re.compile(r"\[文件片段 \d+\]")


def apply_deterministic_guards(claims, context: str, copy_text: str = "") -> list:
    """在模型判斷之後再套幾道機械檢查，專門擋模型判得太寬的方向。純邏輯。

    A. 被歸成 rhetoric、但含比較級/最高級/規模用詞/數字/模糊量詞的句子，
       強制改成無依據的 fact。實測模型把「…的終極方案」這種標題歸成修辭，
       等於最高級主張完全沒經過查核——prompt 明文寫了最高級屬於 fact 還是會漏。
    B. 被判 supported 的 fact，裡面每個阿拉伯數字都必須以完整數字（不是子
       字串，避免 70 被 1970 蓋過）出現在來源文件片段裡；「數萬」「上千」
       這類模糊量詞也必須逐字出現在來源裡。否則降級成無依據。數字是行銷
       文案最常見的捏造形式，模型會因為句子其他部分有依據就整句放行。
       「七成」「百家」這類精確中文數字不在 B 的範圍內（A 會攔修辭裡的，
       fact 裡的靠模型判斷）。
    C. 被判 supported 的 fact 裡的比較級/最高級/規模用詞，那個詞本身必須
       出現在來源文件片段裡，否則降級。實測模型會拿標題裡其他部分的依據，
       把「…的終極方案」整句判成有依據。「最」單字太常見（來源裡就有「最終」
       「最重要」），所以連同後面一個字一起比對（「最完整」看「最完」）。
       這只是字面檢查：來源裡有「全球」（願景句）時，文案把願景寫成現況的
       「正協助全球企業」仍然會過——那種語意上的誇大這裡擋不到。
    D. 被判 supported、但 evidence 裡模型自己寫了「間接」「推論」這類避險
       用詞的，降級。實測「正協助全球企業…」三次裡有兩次 evidence 寫
       「間接對應」；另一次沒寫，所以這道只是部分緩解，不是解決。
    E. 被判 supported 的 fact，evidence 裡當成原文引用的片段（見
       cited_segments()）必須出現在來源文件片段裡。實測模型會捏造引文當
       依據：「文件片段7：無需自建複雜系統」，來源裡根本沒有「自建」兩個字。
       找不到的片段如果全都出現在文案自己的文字裡（claim 或 copy_text），
       是循環引用——依據無法自動核實，標黃；只要有一段連文案裡都沒有，
       就是捏造的引文，標紅。

    A～D 一律標紅。"""
    source_text = _FRAGMENT_LABEL_RE.sub("", context)
    normalized_source = _norm_for_quote(source_text)
    source_numbers = set(_NUMBER_RE.findall(source_text))
    source_numbers |= _arabic_equivalents(source_text)
    guarded = []
    for c in claims:
        if c["kind"] == "rhetoric" and _RISKY_RHETORIC_RE.search(c["claim"]):
            guarded.append(_unsupported(c["claim"], RECLASSIFIED_REASON))
            continue
        if c["kind"] == "fact" and c["supported"] is True:
            guarded.append(_guard_supported_fact(
                c, source_text, normalized_source, source_numbers, copy_text))
            continue
        guarded.append(c)
    return guarded


def _guard_supported_fact(c, source_text, normalized_source, source_numbers, copy_text):
    hedge = _HEDGED_EVIDENCE_RE.search(c["evidence"])
    if hedge:
        return _unsupported(c["claim"], (
            f"模型判定有依據，但它自己的依據說明寫了「{hedge.group(0)}」，"
            f"代表依據不直接——視同無依據，需人工核對。模型原本的依據：{c['evidence']}"
        ))
    unbacked = [p for s in cited_segments(c["evidence"])
                for p in unbacked_quote_pieces(s, normalized_source)]
    if unbacked:
        own_text = _norm_for_quote(c["claim"] + copy_text)
        fabricated = [p for p in unbacked if _norm_for_quote(p) not in own_text]
        if fabricated:
            return _unsupported(c["claim"], (
                f"模型當成依據引用的文字 {fabricated} 在來源文件片段裡找不到，文案裡也"
                "沒有——可能是捏造的引文，視同無依據"
            ))
        return _unsupported(c["claim"], (
            f"模型當成依據引用的文字 {unbacked} 是文案自己的措辭，不是來源原文"
            "（循環引用）——依據無法自動核實，需人工對照來源確認"
        ), alert=ALERT_YELLOW)
    missing = [n for n in _NUMBER_RE.findall(c["claim"]) if n not in source_numbers]
    missing += [q for q in _VAGUE_QUANTITY_RE.findall(c["claim"]) if q not in source_text]
    if missing:
        return _unsupported(c["claim"], (
            f"模型判定有依據，但數字/量詞 {missing} 在來源文件片段裡找不到——"
            "視同無依據，需人工核對"
        ))
    unbacked_words = [w for w in _superlative_tokens(c["claim"]) if w not in source_text]
    if unbacked_words:
        return _unsupported(c["claim"], (
            f"模型判定有依據，但比較級/最高級/規模用詞 {unbacked_words} 在來源文件片段"
            "裡找不到——視同無依據，需人工核對"
        ))
    return c


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9}
_CN_NUMBER_RE = re.compile(r"[零一二兩三四五六七八九十]+")


def _cn_to_int(token: str):
    """一～九十九的中文數字轉整數，其他形式回傳 None。純邏輯。"""
    if "十" not in token:
        return _CN_DIGITS[token] if len(token) == 1 and token in _CN_DIGITS else None
    tens, _, ones = token.partition("十")
    if len(tens) > 1 or len(ones) > 1 or "十" in ones:
        return None
    t = _CN_DIGITS.get(tens, None) if tens else 1
    o = _CN_DIGITS.get(ones, None) if ones else 0
    if t is None or o is None:
        return None
    return t * 10 + o


def _arabic_equivalents(text: str) -> set:
    """來源裡寫成中文數字的（「五大產品矩陣」「八大基礎矩陣」），補上對應的
    阿拉伯數字，讓 B 不會把「5 大產品矩陣」誤判成來源找不到。實測數據導向型
    樣板會自然把中文數字改寫成阿拉伯數字，沒有這一步幾乎每份都會誤報。"""
    values = (_cn_to_int(t) for t in _CN_NUMBER_RE.findall(text))
    return {str(v) for v in values if v is not None}


def _superlative_tokens(claim: str) -> list:
    tokens = []
    for m in _SUPERLATIVE_RE.finditer(claim):
        token = m.group(0)
        if token == "最":
            token = claim[m.start():m.start() + 2]
        tokens.append(token)
    return tokens


def alert_levels(result: dict) -> set:
    """這份查核結果裡出現了哪幾級警示。查核失敗、有損壞項目、有整行漏檢，
    都算紅——那些是「根本沒檢查到」，不是循環引用。純邏輯。"""
    if result["parse_error"] is not None or result["malformed_count"] or result["uncovered_lines"]:
        levels = {ALERT_RED}
    else:
        levels = set()
    levels |= {c["alert"] for c in result["claims"] or [] if c.get("alert")}
    return levels


def needs_attention(result: dict) -> bool:
    """這份文案能不能當成「查核過、沒問題」交出去。只要有任何一條事實主張
    無依據、有無法顯示的損壞項目、有整行沒被檢查到的內容、或整批查核失敗，
    就回傳 True。純邏輯。"""
    if result["parse_error"] is not None:
        return True
    if result["malformed_count"] or result["uncovered_lines"]:
        return True
    return any(c["kind"] == "fact" and c["supported"] is False for c in result["claims"])


def check_marketing_claims(copy_text, context, chat_url, chat_model,
                           max_tokens=8192, timeout=600):
    """對一份生成好的行銷文案全文，比對檢索到的來源文件片段全文(context)，
    逐條判斷賣點主張是否有依據支撐。永遠回傳結構一致的 dict（"claims"/
    "malformed_count"/"parse_error"），不拋例外。

    max_tokens 預設 8192：逐句拆開+分類的推理很長，實測一份6條重點、每條
    帶好幾個模組名稱的文案，2048 全部花在思考上、還沒輸出 JSON 就被截斷。
    截斷要單獨回報，不要偽裝成「JSON 解析失敗」。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"行銷文案全文：\n{copy_text}\n\n"
                    f"---\n\n來源文件片段全文：\n{context}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return _failed_claims(f"呼叫忠實度檢查模型失敗 (HTTP {status}): {resp}")
    try:
        choice = resp["choices"][0]
        raw = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return _failed_claims(f"忠實度檢查模型回應格式不如預期: {resp}")
    if choice.get("finish_reason") == "length":
        return _failed_claims(
            f"忠實度檢查被 max_tokens（{max_tokens}）截斷，推理還沒結束就用完額度，"
            f"沒有產出查核結果"
        )

    result = parse_faithfulness_response(raw)
    if result["parse_error"] is None:
        result["claims"] = apply_deterministic_guards(result["claims"], context, copy_text)
        result["uncovered_lines"] = find_uncovered_lines(copy_text, result["claims"])
    return result
