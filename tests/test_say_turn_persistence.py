"""Tests for persisting turns whose reply came through say()（永続化バグの止血）.

エージェントは本文テキストを書かず `say()` ツールだけで話すことがある。この
とき `result.text` は空で `final_text` が "(no response)" になり、永続化ブロック
（メンタル状態・post-response pipeline・その中の mood nudge）がまとめて飛ばされて
いた。会話しても観測・会話 summary・mood が一切書かれない状態である。

方針は「そのターンに自分がしたこととして、考えたこと（本文）と話したこと
（say）を区別せず記録する」。どちらか一方を選ばない。両方とも無いターン
（沈黙を選んだターン）だけ、従来どおり何も書かない。

粒度はターン単位のまま変えない（1周ごとの記録は新ループ [D-反復出力] の仕事）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from familiar_agent.backend import ToolCall

from tests.test_agent_react_loop import _make_agent, _patch_heavy, _turn


def _say(text: str) -> ToolCall:
    return ToolCall(id="t1", name="say", input={"text": text})


# パイプラインは背景タスクとして spawn されるので、テスト終了時点ではまだ
# await されていないことがある。呼ばれたかどうかは call_count で見る。
_PIPELINE = "familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline"


async def _run_turn(agent, pipeline: AsyncMock, user_input: str = "こんにちは") -> str:
    """パイプラインのモックを _patch_heavy の extra として渡す。

    _patch_heavy は同じ属性を patch するので、外側で patch すると後勝ちで
    上書きされ、こちらのモックが呼ばれない。extra へ渡して一本化する。
    """
    ps = _patch_heavy({_PIPELINE: pipeline})
    for p in ps:
        p.start()
    try:
        return await agent.run(user_input)
    finally:
        for p in ps:
            p.stop()


@pytest.mark.asyncio
async def test_say_only_turn_is_persisted() -> None:
    """say() だけで話したターンでも永続化が走る（本不具合の本体）。"""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", text="", tool_calls=[_say("サッカー楽しかったね")]), ""),
            (_turn("end_turn", text=""), ""),
        ]
    )

    pipeline = AsyncMock()
    await _run_turn(agent, pipeline)

    assert pipeline.call_count == 1, "say() で話したのに永続化が走っていない"


@pytest.mark.asyncio
async def test_say_text_reaches_persistence() -> None:
    """発話した内容が永続化へ渡る（記憶と実際に届いた発話が一致する）。"""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", text="", tool_calls=[_say("サッカー楽しかったね")]), ""),
            (_turn("end_turn", text=""), ""),
        ]
    )

    pipeline = AsyncMock()
    await _run_turn(agent, pipeline)

    assert "サッカー楽しかったね" in pipeline.call_args.kwargs["final_text"]


@pytest.mark.asyncio
async def test_thought_and_speech_are_both_persisted() -> None:
    """本文と発話の両方があれば、区別せず両方が残る（片方を捨てない）。"""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", text="勝った話だ、喜ぼう", tool_calls=[_say("よかったね")]), ""),
            (_turn("end_turn", text=""), ""),
        ]
    )

    pipeline = AsyncMock()
    await _run_turn(agent, pipeline)

    persisted = pipeline.call_args.kwargs["final_text"]
    assert "勝った話だ、喜ぼう" in persisted, "考えたことが落ちている"
    assert "よかったね" in persisted, "話したことが落ちている"


@pytest.mark.asyncio
async def test_only_the_delivered_say_is_persisted() -> None:
    """2回目の say() は音声も表示も抑制されるので、届いた最初の1回だけを残す。"""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (
                _turn("tool_use", text="", tool_calls=[_say("最初のひとこと")]),
                "",
            ),
            (
                _turn("tool_use", text="", tool_calls=[_say("届かない二度目")]),
                "",
            ),
            (_turn("end_turn", text=""), ""),
        ]
    )

    pipeline = AsyncMock()
    # 短い挨拶は brief reply 扱いでイテレーション上限が2になるため、
    # 3周する本ケースでは通常のターンになる入力を使う。
    await _run_turn(agent, pipeline, "今日のサッカーの試合はどうだった？詳しく教えて")

    persisted = pipeline.call_args.kwargs["final_text"]
    assert "最初のひとこと" in persisted
    assert "届かない二度目" not in persisted, "発話されなかった文字列が記憶に混ざっている"


@pytest.mark.asyncio
async def test_silent_turn_is_not_persisted() -> None:
    """考えも発話も無いターンは、従来どおり何も書かない（反証側）。"""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text=""), ""))

    pipeline = AsyncMock()
    await _run_turn(agent, pipeline)

    assert pipeline.call_count == 0, "沈黙のターンまで記録されている"
