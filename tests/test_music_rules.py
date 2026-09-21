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
    wants_shuffle,
)

# 表は (名前, URI, ランダムか)。3 つ目は `MUSIC.md` の既定で、言葉で上書きできる（本人の決定）。
TABLE = (("朝の曲", "spotify:playlist:aaa", False), ("ドライブ", "spotify:playlist:bbb", True))


def test_the_table_is_read_from_the_file():
    text = "# 音楽\n\n朝の曲：spotify:playlist:aaa\nドライブ：spotify:playlist:bbb：ランダム\n"
    assert parse_music_md(text) == TABLE


def test_shuffle_is_off_unless_the_line_says_so():
    assert parse_music_md("夜：spotify:playlist:ccc") == (("夜", "spotify:playlist:ccc", False),)
    assert parse_music_md("夜：spotify:playlist:ccc：順番") == (
        ("夜", "spotify:playlist:ccc", False),
    )
    assert parse_music_md("夜：spotify:playlist:ccc：ランダム") == (
        ("夜", "spotify:playlist:ccc", True),
    )


def test_lines_without_a_uri_are_skipped():
    assert parse_music_md("# 音楽\n\n（名前）：（URI）\nこわれた行\n") == ()
    assert parse_music_md("") == ()


def test_a_fenced_example_does_not_bleed_into_the_name():
    """雛形の例はコードブロックに入れてある。名前に改行が混じらないこと（2026-09-21）。"""
    text = "例：\n\n```\n朝の曲：spotify:playlist:aaa\n```\n"
    assert parse_music_md(text) == (("朝の曲", "spotify:playlist:aaa", False),)


def test_a_playlist_is_found_by_name():
    assert find_playlist("朝の曲", TABLE) == TABLE[0]
    assert find_playlist(" ドライブ ", TABLE) == TABLE[1]
    assert find_playlist("夜の曲", TABLE) is None
    # 言い方に揺れがあっても、表の名前を含んでいれば当てる
    assert find_playlist("朝の曲かけて", TABLE) == TABLE[0]


def test_the_spoken_words_decide_the_order_over_the_table():
    """言葉で言われたら、表の既定を上書きする（本人の決定・2026-09-21）。"""
    assert wants_shuffle("ドライブをランダムでかけて", True) is True
    assert wants_shuffle("ドライブを順番にかけて", True) is False
    assert wants_shuffle("ドライブかけて", True) is True  # 言われなければ表のまま
    assert wants_shuffle("朝の曲をシャッフルで", False) is True
    assert wants_shuffle("朝の曲かけて", False) is False


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
