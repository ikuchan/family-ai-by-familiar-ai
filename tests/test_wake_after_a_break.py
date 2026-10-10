"""名前が文の区切りの直後にあっても呼ばれたと分かる（出-bh・2026-10-11・本人の決定ア）。

10/11 08:18〜08:20、書き起こしが前の家族の会話や雑音と名前を 1 つにまとめ、「あ、ジェルと言うのもいい?ちょっとしていいの?
アパージュ、おはよう」「横横横横横横横 アージュおはよう」を窓の外として捨てた。名前は文頭だけを見る（2026-09-30）。文頭に
無ければ、区切り（「。？！?!.」と空白）の直後を見て、名前から後ろだけを先へ渡す。名前より前は落とす。「、」は区切りに
しない（「ねえ、パジュ」は通さない・2026-09-30 本人の決定「許さない」）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core.wake_window import from_name

NAMES = ["パジュ"]


@pytest.mark.parametrize(
    "heard, cut",
    [
        (
            "あ、ジェルと言うのもいい?ちょっとしていいの?アパージュ、おはよう",
            "アパージュ、おはよう",
        ),
        ("横横横横横横横 アージュおはよう", "アージュおはよう"),
        ("それでいいよ。パジュ、今日の予定は？", "パジュ、今日の予定は？"),
    ],
)
def test_the_name_after_a_break_starts_the_words(heard, cut):
    assert from_name(heard, NAMES) == cut


@pytest.mark.parametrize(
    "heard",
    [
        "パジュ、おはよう",  # 文頭にある
        "ねえ、パジュ、おはよう",  # 「、」の後ろは見ない
        "テレビでパジュって言ってた",  # 区切りの直後ではない
        "今日は晴れだね",  # 名前が無い
    ],
)
def test_otherwise_the_words_are_left_as_they_are(heard):
    assert from_name(heard, NAMES) == heard


def test_no_names_no_change():
    assert from_name("あ? パジュ", []) == "あ? パジュ"


def test_the_loop_gates_and_passes_the_words_from_the_name():
    from familiar_agent.core.wake_window import heard_name
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.agent_names = list(NAMES)
    ip = InformationProcessing(a)
    shown: list[tuple[str, bool]] = []
    ip._on_heard = lambda t, h: shown.append((t, h))  # type: ignore[attr-defined]
    ip._ensure_driver = lambda: None  # type: ignore[method-assign]
    gated: list[str] = []

    async def gate(trigger):
        gated.append(trigger.query)
        trigger.named = heard_name(trigger.query, NAMES)
        return not trigger.named

    ip._swallow_if_unheard = gate  # type: ignore[method-assign]

    async def go():
        task = asyncio.ensure_future(
            ip.push_utterance("ちょっとしていいの?パジュ、おはよう", source="voice")
        )
        await asyncio.sleep(0.05)
        queued = ip._triggers.get_nowait()
        task.cancel()
        await ip.close()
        return queued

    queued = asyncio.run(go())
    assert gated == ["パジュ、おはよう"]
    assert queued.named and queued.query == "パジュ、おはよう"
    assert shown == [("パジュ、おはよう", True)]
