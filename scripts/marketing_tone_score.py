#!/usr/bin/env python3
"""
marketing_tone_score.py

行銷文案的說服力/語氣評分——全新的判斷維度，跟九格/深科技那種投資評估
評分不是同一件事：**這裡沒有客觀對錯，只定義產生分數的機制，不解決「這個
評分本身準不準」這個驗證方法論問題**（那個問題留待另外討論，見規格文件
第七節，不在這裡自行假設一套驗證標準當作定案）。

四個子項目（見規格文件 3.2 節，草案、開放調整）：開頭吸引力、具體性、
行動呼籲、語氣適配度。每項 1-5 分 + 一句話理由，仿照 scoring.py 的
parse_scoring_response() 輸出格式，但這裡的 evidence 是「文案裡具體對應
到這項評分的段落」，不是外部查核依據——跟忠實度檢查（marketing_
faithfulness.py）的 evidence 意義不同，不要混用。

四個子項目一次評分（同一次 LLM 呼叫），任何一項缺漏/不合法就整批視為
失敗，不做部分成功——說服力評分是要整體判斷這份文案的呈現效果，缺一項
子分數比 scoring.py 缺一個評估維度風險更高（那裡可以先算出已成功的維度、
留空缺的維度標記失敗），這裡沒有「先出一部分結果」的實際用途，不宜半殘
顯示。
"""
import json

from rag_common import JSON_OUTPUT_REMINDER, http_json
from marketing_faithfulness import load_first_json_object

TONE_SUBITEMS = ("開頭吸引力", "具體性", "行動呼籲", "語氣適配度")

TONE_SCORE_SYSTEM_PROMPT = (
    "你是資深行銷文案編輯。你會看到一份行銷文案全文，針對下面四個子項目"
    "各給 1-5 分的整數評分（5分最好），沒有客觀對錯，憑你的專業判斷：\n"
    "1. 開頭吸引力：開頭有沒有在前一兩句抓住注意力，不是流水帳式的公司"
    "介紹\n"
    "2. 具體性：有沒有具體數據/案例支撐賣點，不是空泛形容詞堆疊\n"
    "3. 行動呼籲：結尾有沒有明確的下一步呼籲（聯絡/預約/試用等）\n"
    "4. 語氣適配度：語氣是否適合對外行銷情境，不是內部報告式的中性敘述\n"
    "每個子項目都要給 rationale（50字以內，說明為什麼給這個分數）跟"
    "evidence（具體引用文案裡對應到這項評分的段落，不能寫「整體來看」"
    "這種空泛的話）。\n"
    f"{JSON_OUTPUT_REMINDER}\n"
    "請只輸出一個 JSON 物件，格式為：\n"
    '{"scores": {"開頭吸引力": {"score": <1-5的整數>, "rationale": "<字串>", '
    '"evidence": "<字串>"}, "具體性": {...}, "行動呼籲": {...}, '
    '"語氣適配度": {...}}}\n'
    "四個子項目都要有，不要有任何其他文字、不要用 markdown code fence。"
)

_VALID_SCORES = (1, 2, 3, 4, 5)


def _failed_tone_score(parse_error):
    return {"scores": None, "parse_error": parse_error}


def parse_tone_score_response(raw: str) -> dict:
    """把模型輸出解析成說服力/語氣評分結果。四個子項目（TONE_SUBITEMS）
    都必須存在、且各自的 score/rationale/evidence 都合法，才算成功——
    任何一項缺漏或 score 不是 1-5 的整數，整批視為抽取失敗（scores=None，
    parse_error 有值），不半殘顯示只評了一部分子項目的結果。"""
    try:
        parsed = load_first_json_object(raw)
        scores = parsed.get("scores")
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return _failed_tone_score(f"說服力評分結果無法解析成 JSON，原始輸出前 300 字：{raw[:300]}")

    if not isinstance(scores, dict):
        return _failed_tone_score(f"scores 不是物件：{parsed}")

    cleaned_scores = {}
    for subitem in TONE_SUBITEMS:
        entry = scores.get(subitem)
        if not isinstance(entry, dict):
            return _failed_tone_score(f"缺少子項目「{subitem}」或格式不對：{parsed}")
        score = entry.get("score")
        rationale = entry.get("rationale")
        evidence = entry.get("evidence")
        if score not in _VALID_SCORES or not rationale or not evidence:
            return _failed_tone_score(
                f"子項目「{subitem}」欄位不完整或 score 不是 1-5 的合法值：{entry}"
            )
        cleaned_scores[subitem] = {"score": score, "rationale": rationale, "evidence": evidence}

    return {"scores": cleaned_scores, "parse_error": None}


def score_marketing_tone(copy_text, chat_url, chat_model, max_tokens=8192, timeout=600):
    """對一份生成好的行銷文案全文評分說服力/語氣。永遠回傳結構一致的
    dict（"scores"/"parse_error"），不拋例外。max_tokens 預設 8192 的理由
    跟 marketing_faithfulness.check_marketing_claims() 一樣：實測一份四段
    敘事文案，2048 在思考階段就用完。"""
    payload = {
        "model": chat_model,
        "messages": [
            {"role": "system", "content": TONE_SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": f"行銷文案全文：\n{copy_text}"},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    status, resp = http_json("POST", chat_url, payload, timeout=timeout)
    if status != 200:
        return _failed_tone_score(f"呼叫說服力評分模型失敗 (HTTP {status}): {resp}")
    try:
        choice = resp["choices"][0]
        raw = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return _failed_tone_score(f"說服力評分模型回應格式不如預期: {resp}")
    if choice.get("finish_reason") == "length":
        return _failed_tone_score(
            f"說服力評分被 max_tokens（{max_tokens}）截斷，推理還沒結束就用完額度"
        )

    return parse_tone_score_response(raw)
