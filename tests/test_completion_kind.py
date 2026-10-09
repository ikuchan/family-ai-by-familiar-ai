"""完了（自分の動作の結果が届いた）を、機械が「何が起きたか」に分ける（出-ay 段 4-2・2026-10-09・`設計方針_判定の段` §2.2.5）。

道具の名前・失敗の印・結果の文・求めの起点から、種類と、2 回目に並べる動作を決める（本人の表）。選択肢が 1 つなら Jev に
聞かない。失敗で特別に扱うのは 2 つだけ：「別の曲が鳴っている」→ かけ直す、検索の結果が空 → 記憶か調べ直すかを Jev に聞く。
ほかの失敗（使えない・見つからない・時間切れ・知らない文）は、どれも軽く伝える。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.completion_kind import kind_of


@pytest.mark.parametrize(
    "action, origin, actions",
    [
        ("search_deferred", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("fetch_deferred", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("recall", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("recall_when", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("family_schedule", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("ask_vault_yusuke", "発話", ("reply_full", "reply_light", "search_deferred")),
        ("play_music", "発話", ("silent",)),
        ("stop_music", "発話", ("tell_light",)),
        ("next_track", "発話", ("silent",)),
        ("music_volume", "発話", ("silent",)),
        ("set_timer", "発話", ("tell_light",)),
        ("set_alarm", "発話", ("tell_light",)),
        ("cancel_timer", "発話", ("tell_light",)),
        ("pause_timer", "発話", ("tell_light",)),
        ("resume_timer", "発話", ("tell_light",)),
        ("cancel_alarm", "発話", ("tell_light",)),
        ("start_stopwatch", "発話", ("tell_light",)),
        ("stop_stopwatch", "発話", ("tell_light",)),
        ("confirm", "発話", ("tell_light",)),
        ("decline", "発話", ("tell_light",)),
        ("look", "発話", ("reply_light", "look")),
        ("see", "発話", ("reply_light", "look")),
        ("look", "情動", ("look", "talk_light", "silent")),
        ("see", "情動", ("look", "talk_light", "silent")),
        ("search_deferred", "情動", ("silent", "talk_light", "search_deferred")),
    ],
)
def test_a_success_is_sorted_by_the_tool_and_who_started_it(action, origin, actions):
    got = kind_of(action, failed=False, result="結果", origin=origin)
    assert got is not None and got[1] == actions


@pytest.mark.parametrize(
    "result",
    [
        "「ヨルシカ」は見つからなかったので、かけられない",
        "かけられなかった（音の出口が見つからない）",
        "MCP tool 'tavily-search' timed out after 30s",
        "なにか知らない失敗",
    ],
)
def test_other_failures_are_told_lightly(result):
    assert kind_of("play_music", failed=True, result=result, origin="発話")[1] == ("tell_light",)


def test_the_wrong_song_is_played_again():
    got = kind_of(
        "play_music",
        failed=True,
        result="「夜に駆ける」をかけようとしたが、別の曲が鳴っている",
        origin="発話",
    )
    assert got[1] == ("play_music",)


@pytest.mark.parametrize("result", ["", "   ", None])
def test_an_empty_search_asks_memory_or_search_again(result):
    got = kind_of("search_deferred", failed=False, result=result, origin="発話")
    assert got[1] == ("recall", "search_deferred")


def test_an_unknown_tool_is_left_to_the_old_judge():
    assert kind_of("music_suggestion_reply", failed=False, result="x", origin="発話") is None
