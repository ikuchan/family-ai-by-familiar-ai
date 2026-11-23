"""設計ドキュメントの、機械で確かめられる約束（2026-09-10 新設・2026-11-23 改め）。

2026-11-23 から、文書の変わり目は git の履歴で追う。ファイル名と題に版を付けず、`## 更新履歴` の節も持たない
（版上げのたびに改名とリンクの付け替えが要り、更新履歴は文書の 20% を占めて git と二重だった）。

**これは文章の良し悪しを見るテストではない。** 見るのは、ファイル名・題・節・索引のリンクという、
**答えが1つに決まるもの**だけである。中身が実物と合っているかは、人が読んで確かめるしかない。
"""

from __future__ import annotations

import pathlib
import re
import urllib.parse

import pytest

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "NewModesDocs"
_ALL = sorted(p for p in _DOCS.glob("*.md") if p.name != "README.md")


def test_there_are_documents():
    """**反証側**：拾えていなければ、以下がすべて空振りで通ってしまう。"""
    assert len(_ALL) >= 20


@pytest.mark.parametrize("doc", _ALL, ids=lambda p: p.name)
def test_the_file_name_carries_no_version(doc: pathlib.Path):
    """版はファイル名に付けない（変わり目は git で追う）。付けると版上げのたびに改名とリンクの付け替えが要る。"""
    assert not re.search(r"_v\d+_\d+\.md$", doc.name)


@pytest.mark.parametrize("doc", _ALL, ids=lambda p: p.name)
def test_the_title_carries_no_version(doc: pathlib.Path):
    head = doc.read_text(encoding="utf-8").split("\n", 1)[0]
    assert not re.search(r"[（(]v\d+\.\d+", head), f"題に版がある：{head}"


@pytest.mark.parametrize("doc", _ALL, ids=lambda p: p.name)
def test_there_is_no_history_section(doc: pathlib.Path):
    """更新履歴は git が持つ。文書に足し続けると、本文より長くなる（課題8 は 22%）。"""
    assert not re.search(r"^## (更新履歴|改訂履歴)", doc.read_text(encoding="utf-8"), re.M)


def _linked(index: str) -> "list[str]":
    """索引の本文にある md へのリンク先（file 名）。"""
    return [urllib.parse.unquote(m.group(1)) for m in re.finditer(r"\]\(([^)]+\.md)\)", index)]


def _unlinked(index: str) -> "list[str]":
    """README 以外の md のうち、索引からリンクされていないもの。"""
    linked = set(_linked(index))
    return sorted(
        p.name for p in _DOCS.glob("*.md") if p.name != "README.md" and p.name not in linked
    )


def test_the_index_links_resolve():
    """索引から引けない資料は、無いのと同じである。"""
    index = (_DOCS / "README.md").read_text(encoding="utf-8")
    dead = [t for t in _linked(index) if not (_DOCS / t).exists()]
    assert dead == [], f"索引のリンク切れ：{dead}"


@pytest.mark.parametrize("doc", _ALL, ids=lambda p: p.name)
def test_numbered_headings_do_not_repeat(doc: pathlib.Path):
    """`## 4.` が2つあると、参照が指す先が決まらない。"""
    nums = re.findall(r"^## (\d+)\.", doc.read_text(encoding="utf-8"), re.M)
    dup = sorted({n for n in nums if nums.count(n) > 1})
    assert dup == [], f"見出し番号の重複：{dup}"


def test_every_document_is_in_the_index():
    """逆向き（資料→索引）。リンク切れだけを見ていると、索引が空になっても切れたリンクが 0 本で緑になる
    （`0b64d35`・2026-10-11 に README の 195 行が消えて緑だった）。"""
    assert _unlinked((_DOCS / "README.md").read_text(encoding="utf-8")) == []


def test_an_emptied_or_trimmed_index_is_caught():
    """**反証側**：空の索引と 1 行欠けた索引で、リンクの無い資料が出ること。"""
    index = (_DOCS / "README.md").read_text(encoding="utf-8")
    everything = sorted(p.name for p in _DOCS.glob("*.md") if p.name != "README.md")
    assert _unlinked("") == everything
    trimmed = "\n".join(line for line in index.splitlines() if "Notionとの接続" not in line)
    assert [n for n in _unlinked(trimmed)] == [n for n in everything if "Notionとの接続" in n]
