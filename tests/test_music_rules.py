"""音楽の決まり（知-aa 段 1・2026-09-21）。純関数。

鳴らす先は `spotifyd`（MPRIS）。ここは **表の読み取り・通す言葉・寿命・減音**だけを持つ。
"""

from __future__ import annotations

from familiar_agent.core.music_rules import (
    DUCK_RATIO,
    MUSIC_MAX_SEC,
    RESTORE_AFTER_SEC,
    duck,
    expired,
    find_playlist,
    is_music_word,
    parse_music_md,
)

TABLE = (("朝の曲", "spotify:playlist:aaa"), ("ドライブ", "spotify:playlist:bbb"))


def test_the_table_is_read_from_the_file():
    text = "# 音楽\n\n朝の曲：spotify:playlist:aaa\nドライブ：spotify:playlist:bbb\n"
    assert parse_music_md(text) == TABLE


def test_lines_without_a_uri_are_skipped():
    assert parse_music_md("# 音楽\n\n（名前）：（URI）\nこわれた行\n") == ()
    assert parse_music_md("") == ()


def test_a_fenced_example_does_not_bleed_into_the_name():
    """雛形の例はコードブロックに入れてある。名前に改行が混じらないこと（2026-09-21）。"""
    text = "例：\n\n```\n朝の曲：spotify:playlist:aaa\n```\n"
    assert parse_music_md(text) == (("朝の曲", "spotify:playlist:aaa"),)


def test_a_playlist_is_found_by_name():
    assert find_playlist("朝の曲", TABLE) == "spotify:playlist:aaa"
    assert find_playlist(" ドライブ ", TABLE) == "spotify:playlist:bbb"
    assert find_playlist("夜の曲", TABLE) is None
    # 言い方に揺れがあっても、表の名前を含んでいれば当てる
    assert find_playlist("朝の曲かけて", TABLE) == "spotify:playlist:aaa"


def test_the_words_that_pass_while_music_plays():
    """鳴っているあいだは、音楽の操作とプレイリストの変更だけ通す（時間の道具は通さない）。"""
    for word in ("止めて", "ストップ", "音楽止めて", "次の曲", "もっと大きく", "小さくして"):
        assert is_music_word(word, TABLE), word
    assert is_music_word("朝の曲かけて", TABLE)  # プレイリストの変更
    assert not is_music_word("3 分測って", TABLE)
    assert not is_music_word("明日の天気は？", TABLE)
    assert not is_music_word("", TABLE)


def test_music_stops_after_thirty_minutes():
    assert MUSIC_MAX_SEC == 1800
    assert not expired(1000.0, now=1000.0 + 1799)
    assert expired(1000.0, now=1000.0 + 1800)


def test_the_volume_ducks_to_a_quarter_and_comes_back():
    assert DUCK_RATIO == 0.25
    assert RESTORE_AFTER_SEC == 10
    assert duck(0.8) == 0.2
    assert duck(0.0) == 0.0
    assert duck(1.0) == 0.25
