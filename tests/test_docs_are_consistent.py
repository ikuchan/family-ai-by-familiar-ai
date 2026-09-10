"""設計ドキュメントの、機械で確かめられる約束（2026-09-10 新設）。

点検で**同じ抜けが 10 件**出た——版番号を上げたのに、タイトルを直し忘れる／更新履歴に
その版の項を足し忘れる。人が気をつけても抜けるので、機械に見させる。

索引が定める約束は次のとおり。

> 各資料は版番号で管理し、**ファイル名の版＝タイトルの版**で揃える。本文（最新の状態）が
> 先、`## 更新履歴` は末尾に置く。

**これは文章の良し悪しを見るテストではない。** 見るのは、file 名・タイトル・履歴・索引の
リンクという、**答えが1つに決まるもの**だけである。中身が実物と合っているかは、人が読んで
確かめるしかない（そちらは群ごとの点検で行う）。
"""

from __future__ import annotations

import pathlib
import re
import urllib.parse

import pytest

_DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "NewModesDocs"
_VERSIONED = sorted(p for p in _DOCS.glob("*.md") if re.search(r"_v0_\d+\.md$", p.name))


def _version_of(path: pathlib.Path) -> str:
    return "0." + re.search(r"_v0_(\d+)\.md$", path.name).group(1)


def test_there_are_versioned_documents():
    """**反証側**：拾えていなければ、以下がすべて空振りで通ってしまう。"""
    assert len(_VERSIONED) >= 20


@pytest.mark.parametrize("doc", _VERSIONED, ids=lambda p: p.name)
def test_the_title_carries_the_same_version_as_the_file_name(doc: pathlib.Path):
    """ファイル名の版＝タイトルの版。片方だけ上げると、どちらが本当か分からなくなる。"""
    head = doc.read_text(encoding="utf-8").split("\n", 1)[0]
    found = re.search(r"v(\d+\.\d+)", head)
    assert found, f"タイトルに版が無い：{head}"
    assert found.group(1) == _version_of(doc), f"タイトル v{found.group(1)}"


@pytest.mark.parametrize("doc", _VERSIONED, ids=lambda p: p.name)
def test_the_history_has_an_entry_for_this_version(doc: pathlib.Path):
    """**版を上げたら、何を変えたかを1項書く。**

    書かなければ、あとから git を掘るしかない。実際に 10 件が掘り直しになった。
    """
    text = doc.read_text(encoding="utf-8")
    head = re.search(r"^## (更新履歴|改訂履歴)", text, re.M)
    assert head, "履歴の節が無い"
    ver = _version_of(doc)
    assert re.search(rf"^> v{re.escape(ver)}[：:]", text[head.end() :], re.M), f"v{ver} の項が無い"


def test_the_index_links_resolve():
    """索引から引けない資料は、無いのと同じである。"""
    index = (_DOCS / "README.md").read_text(encoding="utf-8")
    dead = [
        t
        for m in re.finditer(r"\]\(([^)]+\.md)\)", index)
        if not (_DOCS / (t := urllib.parse.unquote(m.group(1)))).exists()
    ]
    assert dead == [], f"索引のリンク切れ：{dead}"


@pytest.mark.parametrize("doc", _VERSIONED, ids=lambda p: p.name)
def test_numbered_headings_do_not_repeat(doc: pathlib.Path):
    """`## 4.` が2つあると、参照が指す先が決まらない。"""
    nums = re.findall(r"^## (\d+)\.", doc.read_text(encoding="utf-8"), re.M)
    dup = sorted({n for n in nums if nums.count(n) > 1})
    assert dup == [], f"見出し番号の重複：{dup}"
