"""`ME.md` が持つ「名前として使える言葉」の一覧。

沈黙依頼は名前で呼ばれたときだけ受けるので、**どう呼ばれても通る**必要がある。呼び方は
一つとは限らない（愛称、呼び捨て）ので、`ME.md` に並べて書けるようにする。

名前の正本は `ME.md` だけである（`.env` の `AGENT_NAME` と設定画面の項目は撤去した）。
"""

from __future__ import annotations

from familiar_agent.core.parsing import parse_me_names


def test_a_single_name_is_a_list_of_one():
    assert parse_me_names("名前： パジュ") == ["パジュ"]


def test_several_names_separated_by_a_japanese_comma():
    assert parse_me_names("名前： パジュ、ぱじゅ、パジュちゃん") == ["パジュ", "ぱじゅ", "パジュちゃん"]


def test_a_western_comma_works_too():
    assert parse_me_names("名前： パジュ, Paju") == ["パジュ", "Paju"]


def test_surrounding_spaces_are_dropped():
    assert parse_me_names("名前：  パジュ 、 ぱじゅ  ") == ["パジュ", "ぱじゅ"]


def test_no_name_line_yields_nothing():
    assert parse_me_names("何も書いていない") == []


def test_empty_entries_are_skipped():
    assert parse_me_names("名前： パジュ、、ぱじゅ") == ["パジュ", "ぱじゅ"]
