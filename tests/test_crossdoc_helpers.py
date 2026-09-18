"""
測試重點：
- parse_source_tag()：測試有無 source 標籤、空值處理
- distinct_sources()：空列表、過濾 None/空字串、去重、排序（不 strip 內容，
  只濾掉 falsy 值——這是刻意的：呼叫端如果傳進未 strip 的 source，寧可讓
  查詢明確查不到、报錯提示，也不要靜默把使用者的參數丟掉當作沒指定）
- missing_sources()：空字典、全存在、混合存在、全不存在
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from rag_common import distinct_sources, missing_sources
from generate_report import parse_source_tag


# parse_source_tag()：有[source: xxx]前綴時，回傳(xxx去頭尾空白, 前綴後剩下的問題文字去頭尾空白)
@pytest.mark.parametrize("line,expected_source,expected_rest", [
    ("[source: file1.py] question text", "file1.py", "question text"),
    ("[source:   file2.py   ] another question", "file2.py", "another question"),
    ("[source: ] empty_source", "", "empty_source"),
])
def test_parse_source_tag_with_prefix(line, expected_source, expected_rest):
    source, rest = parse_source_tag(line)
    assert source == expected_source, f"原始字串: {line}"
    assert rest == expected_rest, f"原始字串: {line}"


# parse_source_tag()：沒有[source: ...]前綴時，回傳(None, 原字串不變，不去空白)
@pytest.mark.parametrize("line,expected_source,expected_rest", [
    ("normal question", None, "normal question"),
    ("   leading spaces", None, "   leading spaces"),
    ("trailing spaces   ", None, "trailing spaces   "),
])
def test_parse_source_tag_without_prefix(line, expected_source, expected_rest):
    source, rest = parse_source_tag(line)
    assert source == expected_source
    assert rest == expected_rest


# parse_source_tag()：[source: ]裡面只有空白字元時，source應該是空字串
@pytest.mark.parametrize("line,expected_source", [
    ("[source: ]", ""),
    ("[source:   ]", ""),
    ("[source: \t\n]", ""),
])
def test_parse_source_tag_empty_source(line, expected_source):
    source, _ = parse_source_tag(line)
    assert source == expected_source


# parse_source_tag()：前綴不是在行首時不會被解析
@pytest.mark.parametrize("line", [
    "this is [source: file.py] text",
    "prefix[source: file.py]suffix",
])
def test_parse_source_tag_not_at_start(line):
    source, rest = parse_source_tag(line)
    assert source is None
    assert rest == line


# distinct_sources()：空list回傳空list
def test_distinct_sources_empty_list():
    assert distinct_sources([]) == []


# distinct_sources()：有None或空字串混在裡面時要濾掉，但不會 strip 內容本身
@pytest.mark.parametrize("sources,expected", [
    ([None, "", "a", "a", "b"], ["a", "b"]),
    ([None, "  ", "  c  ", "d", None], ["  ", "  c  ", "d"]),
])
def test_distinct_sources_filter_none_and_empty(sources, expected):
    assert distinct_sources(sources) == expected


# distinct_sources()：重複的值只留一個，結果排序
@pytest.mark.parametrize("sources,expected", [
    (["a", "b", "a", "c"], ["a", "b", "c"]),
    (["branes_deck_excerpt.md", "branes_dd_report.md", "branes_deck_excerpt.md"],
     ["branes_dd_report.md", "branes_deck_excerpt.md"]),
    (["z", "a", "z", "b"], ["a", "b", "z"]),
])
def test_distinct_sources_deduplicate_and_sort(sources, expected):
    assert distinct_sources(sources) == expected


# missing_sources()：空dict回傳空list
def test_missing_sources_empty_dict():
    assert missing_sources({}) == []


# missing_sources()：全部都True(存在)時回傳空list
@pytest.mark.parametrize("source_exists", [
    {"a": True},
    {"x": True, "y": True},
])
def test_missing_sources_all_exist(source_exists):
    assert missing_sources(source_exists) == []


# missing_sources()：混合True/False時只回傳False的那些鍵，且排序過
@pytest.mark.parametrize("source_exists,expected", [
    ({"a": False, "b": True, "c": False}, ["a", "c"]),
    ({"z": True, "x": False, "y": False}, ["x", "y"]),
])
def test_missing_sources_mixed(source_exists, expected):
    assert missing_sources(source_exists) == expected


# missing_sources()：全部False時回傳全部鍵，排序過
@pytest.mark.parametrize("source_exists,expected", [
    ({"a": False}, ["a"]),
    ({"x": False, "y": False, "z": False}, ["x", "y", "z"]),
    ({"b": False, "a": False}, ["a", "b"]),
])
def test_missing_sources_all_missing(source_exists, expected):
    assert missing_sources(source_exists) == expected
