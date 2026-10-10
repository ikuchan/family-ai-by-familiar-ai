"""聞き違いでも、手元のプレイリストに読みで当てる（知-at・2026-10-10・実機 11:05・本人の決定ア）。

「プレイリストたいきをかけて」が「待機」「対キー」と書き起こされ、手元（`MUSIC.md`・自分のプレイリスト）は文字で照らすので
当たらず、Spotify 全体から別の曲をかけた。文字で当たらなかったときだけ、言われた名前と手元の名前をどちらも読み（カタカナ・
OpenJTalk）にして照らす。範囲は `MUSIC.md` と自分のプレイリストだけ（名前が少なく、誤りも速さも心配が少ない）。短い名前の
読みはほかの言葉の読みに入りやすいので、読みで当てるのは手元の名前の読みが 3 字以上のときだけ〔仮・本人の決定〕。
「プレイリスト」と言われたら、軽量LLM に種類（`kind`）を書かせる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.core import music_rules as mr
from familiar_agent.tools.music import MusicTool

TABLE = (("たいき", "spotify:playlist:taiki", True), ("あい", "spotify:playlist:ai", False))
CATALOG = mc.Catalog(
    playlists=[mc.Playlist("p1", "よるのうた", "spotify:playlist:yoru", [])],
    albums=[],
    tracks=[{"title": "よぞら", "artist": "誰か", "uri": "spotify:track:yozora"}],
)


@pytest.mark.parametrize("said", ["待機", "対キー"])
def test_a_misheard_name_is_found_by_its_reading(said):
    got = mr.find_local(said, TABLE, CATALOG)
    assert got is not None and got[2] == "spotify:playlist:taiki"
    assert got[0] == "読み:MUSIC.md"


def test_a_reading_that_differs_is_not_found():
    assert mr.find_local("帯域", TABLE, CATALOG) is None


def test_the_text_comes_first():
    table = (("待機", "spotify:playlist:taiki_kanji", False), *TABLE)
    got = mr.find_local("待機", table, CATALOG)
    assert got is not None and got[0] == "MUSIC.md" and got[2] == "spotify:playlist:taiki_kanji"


def test_a_short_reading_is_not_matched():
    """「あい」（読み 2 字）は、読みでは当てない（「愛」「藍」などに広く当たる）。"""
    assert mr.find_local("愛", (("あい", "spotify:playlist:ai", False),), CATALOG) is None


def test_own_playlists_are_also_read():
    got = mr.find_local("夜の歌", (), CATALOG)  # ヨルノウタ
    assert got is not None and got[0] == "読み:プレイリスト"


def test_the_library_is_not_matched_by_reading():
    assert mr.find_local("夜空", (), CATALOG) is None  # 「よぞら」（ヨゾラ）はライブラリの曲


def test_the_reply_says_it_was_read(monkeypatch):
    from familiar_agent.core import music_catalog

    monkeypatch.setattr(music_catalog, "stored", lambda: CATALOG)
    io = MagicMock()
    io.play = AsyncMock(return_value=True)
    io.set_shuffle = AsyncMock(return_value=True)
    tool = MusicTool(io=io, bus=lambda: MagicMock(), table=lambda: TABLE)
    text, ok = asyncio.run(tool.call("play_music", {"name": "待機"}))
    assert ok and "読み" in text and "たいき" in text and "待機" in text


def test_the_writer_is_told_to_write_the_kind_for_a_playlist():
    from familiar_agent.loop.arbiter import _EXTRA_ACTIONS

    note = _EXTRA_ACTIONS["play_music"][1]
    assert '"kind"' in note and "プレイリスト" in note
