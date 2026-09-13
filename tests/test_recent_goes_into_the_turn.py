"""直近のやりとりを、ターンの文脈へ渡す（段 4 → 記-h で W の枠に）。

会話履歴は W の先頭の枠として主LLM へ渡る。時系列で最新 n 往復（無条件）＋継起の鎖。
**逐語で出す。** 細部が要るからこの設計にしたので、ここで縮めると意味がない。
**判定を待たない。** 続き先の判定は継起の辺を書くだけで、載せる／載せないを決めない
（2026-09-13 に撤回。判定つきだと調停と主LLM で記憶が食い違い、話題が切り替わった直後は
直近が空になった）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends import ToolCall
from tests.test_event_loop import _agent, _run, _turn

_NOW = datetime(2026, 9, 7, 21, 14, tzinfo=timezone.utc)


def _rows():
    return [
        {
            "obs_id": "o1",
            "content": "明日の運動会って何時から？",
            "role": "起点",
            "direction": "発話",
            "timestamp": _NOW,
            "depth": 0,
        },
        {
            "obs_id": "o2",
            "content": "8時半に開会式だよ。",
            "role": "答え",
            "direction": "発話",
            "timestamp": _NOW,
            "depth": 0,
        },
    ]


def _with_recent(a, rows):
    a._memory.latest_exchange_origins = MagicMock(return_value=["o1"] if rows else [])
    a._memory.recent_exchanges = MagicMock(return_value=rows)


def test_the_recent_talk_reaches_the_system_text_without_a_judgement():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    a._evaluator.judge_follows = AsyncMock(return_value=None)  # 続きと判定されなくても
    _with_recent(a, _rows())
    _run(a, utterance="開会式って何時だっけ")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "直近のやりとり" in system
    assert "明日の運動会って何時から？" in system
    assert "8時半に開会式だよ。" in system


def test_the_verbatim_is_not_shortened():
    """W の過去の列は 120 字で切るが、直近の枠は切らない。切ると細部が消える。"""
    long = "あ" * 400
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _with_recent(
        a,
        [
            {
                "obs_id": "o9",
                "content": long,
                "role": "答え",
                "direction": "発話",
                "timestamp": _NOW,
                "depth": 0,
            }
        ],
    )
    _run(a, utterance="ねえ")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert long in system


def test_nothing_is_shown_when_there_is_nothing_recent():
    """空の見出しは「無い」ではなく「調べたが無い」と読まれる。出さない。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _with_recent(a, [])
    _run(a, utterance="ねえ")
    system = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
    assert "[直近のやりとり（" not in system  # 枠の見出し（規則文は枠の名を挙げるので語では見ない）


def test_the_recent_frame_is_read_by_time_not_by_a_cursor():
    """起点は時刻順に店から引く。装置は表示のカーソルを持たない（再起動直後でも同じ）。"""
    from familiar_agent.loop.event_loop import InformationProcessing

    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    _with_recent(a, _rows())
    _run(a, utterance="ねえ")
    assert a._memory.latest_exchange_origins.called
    assert not hasattr(InformationProcessing(a), "_recent_cursor")
