"""
測試重點：
- compute_blocking_signature()：驗證相同range_str與findings（順序不同）產生相同hash，不同range_str產生不同hash；任何findings欄位差異導致hash不同
- read_valid_override()：驗證檔案不存在/無匹配token/過期token/有效token/無效JSON行的處理邏輯
"""

import sys
import os
import hashlib
import json
import datetime
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from code_review_agent import (
    compute_blocking_signature,
    override_log_path,
    read_valid_override,
)
# 這裡刻意直接 import 真正的 override_log_path()，不要自己重寫一份同名的
# 假 helper——草稿原本自己定義了一個回傳 "override_log.jsonl" 的版本，跟
# 真正的函式回傳的 ".code_review_overrides.jsonl"（OVERRIDE_LOG_FILENAME）
# 完全是兩個不同檔名。這造成一個嚴重的假陽性：test_read_valid_override_
# token_not_found／test_read_valid_override_token_expired 這兩條測試把
# mock 檔案寫到錯的路徑，read_valid_override() 內部用真正的路徑去找，
# 根本找不到檔案，永遠回傳 None——斷言 result is None 剛好「通過」，但
# 通過的原因是「檔案根本沒被讀到」，不是「token 真的比對出不符合／過期」，
# 測試等於沒測到自己聲稱要測的東西。跟批次A抓到的
# test_render_consistency_section_consistent 是同一種假陽性模式。

# compute_blocking_signature()：相同range_str與findings（順序不同）產生相同hash
@pytest.mark.parametrize("range_str, findings1, findings2", [
    ("file1.py:10-20", [{"file": "file1.py", "line": 10, "message": "error1"}, {"file": "file2.py", "line": 20, "message": "error2"}], [{"file": "file2.py", "line": 20, "message": "error2"}, {"file": "file1.py", "line": 10, "message": "error1"}]),
])
def test_compute_blocking_signature_same_findings_order(range_str, findings1, findings2):
    hash1 = compute_blocking_signature(range_str, findings1)
    hash2 = compute_blocking_signature(range_str, findings2)
    assert hash1 == hash2, "相同range_str與findings（順序不同）應產生相同hash"


# compute_blocking_signature()：不同findings欄位導致hash不同
@pytest.mark.parametrize("range_str, findings1, findings2", [
    ("file1.py:10-20", [{"file": "file1.py", "line": 10, "message": "error1"}, {"file": "file2.py", "line": 20, "message": "error2"}], [{"file": "file1.py", "line": 10, "message": "error1"}, {"file": "file2.py", "line": 20, "message": "error3"}]),
])
def test_compute_blocking_signature_different_findings(range_str, findings1, findings2):
    hash1 = compute_blocking_signature(range_str, findings1)
    hash2 = compute_blocking_signature(range_str, findings2)
    assert hash1 != hash2, "不同findings欄位應導致hash不同"


# compute_blocking_signature()：不同range_str產生不同hash
@pytest.mark.parametrize("range_str1, range_str2, findings", [
    ("file1.py:10-20", "file1.py:10-21", [{"file": "file1.py", "line": 10, "message": "error1"}, {"file": "file2.py", "line": 20, "message": "error2"}]),
])
def test_compute_blocking_signature_different_range_str(range_str1, range_str2, findings):
    hash1 = compute_blocking_signature(range_str1, findings)
    hash2 = compute_blocking_signature(range_str2, findings)
    assert hash1 != hash2, "不同range_str應產生不同hash"


# read_valid_override()：檔案不存在時回傳None
def test_read_valid_override_file_not_found(tmp_path):
    repo_root = tmp_path
    token = "test_token"
    result = read_valid_override(repo_root, token)
    assert result is None, "檔案不存在時應回傳None"


# read_valid_override()：token不匹配時回傳None
def test_read_valid_override_token_not_found(tmp_path):
    repo_root = tmp_path
    token = "test_token"
    path = override_log_path(repo_root)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"token": "other_token", "expires_at": "2023-01-01T00:00:00Z"}))
    result = read_valid_override(repo_root, token)
    assert result is None, "token不匹配時應回傳None"


# read_valid_override()：token匹配但過期時回傳None
def test_read_valid_override_token_expired(tmp_path):
    repo_root = tmp_path
    token = "test_token"
    path = override_log_path(repo_root)
    expires_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"token": token, "expires_at": expires_at.isoformat()}))
    result = read_valid_override(repo_root, token)
    assert result is None, "token過期時應回傳None"


# read_valid_override()：token匹配且未過期時回傳完整紀錄
def test_read_valid_override_token_valid(tmp_path):
    repo_root = tmp_path
    token = "test_token"
    path = override_log_path(repo_root)
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"token": token, "expires_at": expires_at.isoformat(), "other": "data"}))
    result = read_valid_override(repo_root, token)
    assert result == {"token": token, "expires_at": expires_at.isoformat(), "other": "data"}, "token有效時應回傳完整紀錄"


# read_valid_override()：檔案有無效JSON行時跳過該行，繼續讀下一行
def test_read_valid_override_invalid_json(tmp_path):
    repo_root = tmp_path
    token = "test_token"
    path = override_log_path(repo_root)
    # 原本這裡的 expires_at 寫死 "2023-01-01T00:00:00Z"——這個日期已經過去
    # 了（現在是2026年），就算孤立標籤/路徑問題都修好，這筆紀錄本身也會
    # 因為已過期被判定不合法，回傳 None，跟這條測試想驗證的「無效JSON
    # 行有沒有被正確跳過、繼續讀到後面合法的那一行」完全是两件事，斷言
    # 會失敗但失敗理由跟測試名稱說的完全不一樣。改成用未來的到期時間。
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    valid_record = {"token": token, "expires_at": expires_at.isoformat()}
    with open(path, "w", encoding="utf-8") as f:
        f.write("invalid json\n")
        f.write(json.dumps(valid_record) + "\n")
    result = read_valid_override(repo_root, token)
    assert result == valid_record, "無效JSON行應被跳過，繼續讀到後面合法且未過期的紀錄"