"""完了は機械で分け、選択肢が 1 つなら Jev に聞かない（出-ay 段 4-2・2026-10-09・`設計方針_判定の段` §2.2.5）。

結果が届いた反復では、返った道具（名前・失敗の印・結果の文）から `core/completion_kind.kind_of` が何が起きたかを決める。
選択肢が 1 つならそのまま（黙る→light・文なし、軽く伝える→light・軽量LLM の一言、道具→action）。2 つ以上なら Jev に
2 回目だけ聞き、確信度に関係なく 1 番を使う（本人：この領域では確信度を使わない）。結果が届いた反復で黙ると決まったら、
発話の求めでも主LLM を呼ばずに沈黙で閉じる。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.loop import iteration
from tests._arbiter_fakes import decide, jev_says, prompt_of, writer_says


def _jev_picks(choice: str, confidence: float = 0.2) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True, answers={"action": {"choice": choice, "confidence": confidence}}
        )
    )
    return jev


def _run(returned, *, jev=None, writer=None, origin="発話"):
    jev = jev or jev_says("full")
    writer = writer or writer_says(
        {"text": "掛けたよ", "query": "夜に駆ける", "tool_input": {"name": "夜に駆ける"}}
    )
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュ、音楽をかけて",
            origin=origin,
            returned=returned,
        )
    )
    return d, jev, writer


def test_music_played_is_silent_without_asking_jev():
    d, jev, writer = _run((("play_music", False, "「ケイマン」をランダムでかけ始めた"),))
    jev.ask.assert_not_awaited()
    writer.complete.assert_not_awaited()
    assert (d.branch, d.text) == ("light", "")


@pytest.mark.parametrize("action", ["set_timer", "stop_music", "start_stopwatch", "confirm"])
def test_a_finished_operation_is_told_lightly(action):
    d, jev, writer = _run(((action, False, "済んだ"),))
    jev.ask.assert_not_awaited()
    assert (d.branch, d.text) == ("light", "掛けたよ")
    assert "軽く伝える" in prompt_of(writer)


def test_a_failure_is_told_lightly():
    d, jev, _ = _run((("play_music", True, "「ヨルシカ」は見つからなかったので、かけられない"),))
    jev.ask.assert_not_awaited()
    assert d.branch == "light" and d.text == "掛けたよ"


def test_the_wrong_song_is_played_again():
    d, jev, _ = _run(
        (("play_music", True, "「夜に駆ける」をかけようとしたが、別の曲が鳴っている"),)
    )
    jev.ask.assert_not_awaited()
    assert (d.branch, d.action) == ("action", "play_music")


def test_a_research_answer_asks_jev_only_the_listed_actions_and_ignores_confidence():
    jev = _jev_picks("reply_light", confidence=0.1)
    d, jev, _ = _run((("search_deferred", False, "明日は晴れ"),), jev=jev)
    (_state, questions), _ = jev.ask.await_args
    assert set(questions) == {"action"}
    assert set(questions["action"]["criteria"]) == {"reply_full", "reply_light", "search_deferred"}
    assert (d.branch, d.text) == ("light", "掛けたよ")  # 確信度 0.1 でも 1 番を使う


def test_thinking_goes_to_the_main_llm():
    d, _, _ = _run((("search_deferred", False, "明日は晴れ"),), jev=_jev_picks("reply_full"))
    assert d.branch == "full"


def test_an_unknown_tool_keeps_the_old_judge():
    jev = jev_says("full")
    d, jev, _ = _run((("music_suggestion_reply", False, "x"),), jev=jev)
    jev.ask.assert_awaited()
    assert "branch" in jev.ask.await_args.args[1]


def test_a_quiet_decision_on_a_result_closes_silently_even_for_a_conversation():
    from familiar_agent.loop.arbiter import Decision

    quiet = Decision(branch="light", text="")
    assert iteration.closes_silently(quiet, trigger_kind="発話", returned=frozenset({"play_music"}))
    assert iteration.closes_silently(quiet, trigger_kind="情動", returned=frozenset())
    assert not iteration.closes_silently(quiet, trigger_kind="発話", returned=frozenset())


# ── 音楽の道具が調停で必ず倒れていた（2026-10-09・段 4-2 で一緒に直す・本人の決定ア）──────────────────────
#
# 動作には語（query）が要るが、音楽の道具は道具の入力（曲の名前）だけを受けるので語が空のまま残り、`assemble` が
# 「動作なのに語が無い」として None を返していた。Jev が play_music を選んでも必ず主LLM に倒れ、ログに調停が play_music
# を投げた跡は一度も無い。タイマーと同じく、道具の入力から語を作る（曲の名前、無ければ道具の名前）。


@pytest.mark.parametrize(
    "action, tool_input, query",
    [
        ("play_music", {"name": "ケイマン", "order": "ランダム"}, "ケイマン"),
        ("stop_music", {}, "stop_music"),
        ("next_track", {}, "next_track"),
        ("music_volume", {"value": 0.3}, "music_volume"),
    ],
)
def test_the_arbiter_can_dispatch_a_music_tool(action, tool_input, query):
    d = asyncio.run(
        decide(
            jev=jev_says("action", action=action),
            writer=writer_says({"tool_input": tool_input}),
            utterance="パジュ、音楽",
            extra_actions=(action,),
        )
    )
    assert (d.branch, d.action, d.query) == ("action", action, query)
    assert d.tool_input == tool_input
