"""REST の各層が、2 つに分けたり書き直したりした JSON の返りも読む（記-p・2026-10-10・本人の決定ア）。

畳み込みが「返りを読めなかった」で見送った日（6/14・6/28）の材料で同じ頼みを 18 回投げると、途切れは 0 回（どれも
`end_turn`・出力 352〜885 トークン）で、読めなかった 3 回は、JSON を 2 つに分けて書いた（`{"episode": …}` の次の行に
`{"persons": …}`）か、1 度書いてから「出力形式を誤りました。正しく出力します。」と書き直したものだった。読む側は最初の
`{` から最後の `}` までを 1 つとして読んでいたので `Extra data` で落ちた。同じ読み方が REST の 8 か所にあった。返りの中の
JSON の物体を前から順にすべて読み、後のもので上書きしながら 1 つにまとめる口（`structured_ask.read_json_merged`）に寄せる。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.structured_ask import read_json_merged

SPLIT = (
    '```json\n{"episode": "今日はパパとワールドカップの話をした。"}\n'
    '{"persons": {"パパ": "試合を楽しみにしている。"}}\n```'
)
REDONE = (
    '```json\n{"episode": "一度目の日記。"}\n```\n\n...出力形式を誤りました。正しく出力します。\n\n'
    '```json\n{"episode": "書き直した日記。", "persons": {"たいき": "眠った。"}}\n```'
)


def test_split_objects_are_merged():
    assert read_json_merged(SPLIT) == {
        "episode": "今日はパパとワールドカップの話をした。",
        "persons": {"パパ": "試合を楽しみにしている。"},
    }


def test_a_redone_answer_keeps_the_later_one():
    got = read_json_merged(REDONE)
    assert got == {"episode": "書き直した日記。", "persons": {"たいき": "眠った。"}}


def test_stray_braces_are_skipped_and_one_fenced_object_reads_as_before():
    assert read_json_merged('メモ {こういう括弧} のあと {"a": 1}') == {"a": 1}
    assert read_json_merged('```json\n{"a": {"b": [1, {"c": 2}]}}\n```') == {
        "a": {"b": [1, {"c": 2}]}
    }


@pytest.mark.parametrize("raw", ["", "JSON はありません", "[1, 2]", None])
def test_nothing_is_none(raw):
    assert read_json_merged(raw) is None


# ── 8 か所の読み口 ────────────────────────────────────────────────────────────


def _two(first: dict, second: dict) -> str:
    import json

    return json.dumps(first, ensure_ascii=False) + "\n" + json.dumps(second, ensure_ascii=False)


def test_fold_reads_a_split_answer():
    from familiar_agent.loop import rest_fold

    got = rest_fold._parse(SPLIT)
    assert got is not None and got.persons == {"パパ": "試合を楽しみにしている。"}


@pytest.mark.parametrize("module", ["rest_season", "rest_self_image"])
def test_dict_readers_read_a_split_answer(module):
    import importlib

    mod = importlib.import_module(f"familiar_agent.loop.{module}")
    assert mod._parse(_two({"a": "1"}, {"b": "2"})) == {"a": "1", "b": "2"}


def test_core_reads_a_split_answer():
    from familiar_agent.loop import rest_core

    got = rest_core._parse(_two({"text": "核"}, {"sources": ["s1"], "people": ["パパ"]}))
    assert got == ("核", ["s1"], ["パパ"])


def test_music_reads_a_split_answer():
    from familiar_agent.loop import rest_music

    got = rest_music._parse(_two({"title": "灯", "artist": "誰か"}, {"reason": "合う"}))
    assert got == {"title": "灯", "artist": "誰か", "reason": "合う"}


def test_family_now_and_calendar_read_a_redone_answer():
    from familiar_agent.loop import rest_calendar, rest_family_now

    assert rest_family_now._parse(_two({"now": "前"}, {"now": "いまの様子"})) == "いまの様子"
    assert rest_calendar._parse(_two({"summary": "前"}, {"summary": "月のまとめ"}), 100) == (
        "月のまとめ"
    )


def test_settings_reads_a_redone_answer():
    from familiar_agent.loop import rest_settings

    assert (
        rest_settings._proposed(_two({"arbiter_timeout_sec": 4.0}, {"arbiter_timeout_sec": 5.0}))
        == 5.0
    )
    assert rest_settings._proposed("提案なし") is None
