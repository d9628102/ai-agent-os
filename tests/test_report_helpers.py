"""
測試重點：
- filter_citations() 測試相關度門檻值0.6的邊界情況與空列表情境
- slugify_heading() 測試中文/特殊符號處理與slug衝突現象
- 皆依照現有測試檔案的參數化寫法與斷言風格
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import pytest

from generate_report import slugify_heading, filter_citations

MIN_CITATION_SCORE = 0.6


# filter_citations() 測試案例
@pytest.mark.parametrize("hits,expected", [
    # 剛好等於門檻值的邊界情況
    ([{"score": 0.6}], [{"score": 0.6}]),
    # 過濾後清單為空的情境
    ([{"score": 0.5}, {"score": 0.59}], []),
    # 正常過濾情境
    ([{"score": 0.6}, {"score": 0.7}], [{"score": 0.6}, {"score": 0.7}]),
    # 零個項目的情境
    ([], []),
])
def test_filter_citations_boundary_and_empty(hits, expected):
    result = filter_citations(hits)
    assert result == expected, (
        f"輸入 {hits} 應返回 {expected}，實際得到 {result}"
    )


# slugify_heading() 測試案例
@pytest.mark.parametrize("text,expected_slug", [
    # 中文處理測試
    ("這是一個標題", "這是一個標題"),
    # 特殊符號處理
    ("Hello! World®?", "hello-world"),
    # 這是與 GitHub 渲染器行為一致的已知 upstream 限制（句點被直接刪除、
    # 非轉為連字符），非本函式缺陷——GitHub 自己的錨點演算法也是直接刪句
    # 點，所以「3.10」跟「3.11」在真實 GitHub 上一樣會撞出同一個錨點
    # 「python-310-vs-311」。曾評估改成把句點轉成連字符，但那樣會讓
    # slug 跟真實 GitHub/VS Code 產生的錨點不一致，違背這個函式想要
    # 「貼合真實渲染器行為」的設計目的，所以維持現況、不修這個技術債。
    ("Python 3.10 vs 3.11", "python-310-vs-311"),
    # 空白處理
    ("   多個   空格   ", "多個-空格"),
    # 混合情境
    ("Section #1: Introduction", "section-1-introduction"),
])
def test_slugify_heading_special_characters(text, expected_slug):
    result = slugify_heading(text)
    assert result == expected_slug, (
        f"「{text}」應轉換為 {expected_slug}，實際得到 {result}"
    )


def test_slugify_heading_collision():
    # 已知限制：不同標題可能撞出相同 slug，slugify_heading 目前沒有防碰撞
    # 機制。這裡用「空格」跟「已有的連字符」都會被轉成 "-" 的事實構造一組
    # 真的會碰撞的例子——不是假設,是這兩個輸入實際上就是會撞。
    slug1 = slugify_heading("Hello World")
    slug2 = slugify_heading("Hello-World")
    assert slug1 == slug2 == "hello-world", (
        f"「Hello World」與「Hello-World」預期撞出相同 slug "
        f"（已知限制,目前沒有防碰撞機制），實際得到 {slug1!r} 跟 {slug2!r}"
    )