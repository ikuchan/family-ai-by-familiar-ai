"""窓を開けた聞き違いの名前を、正しい名前に直して先へ渡す（知-ap・2026-10-10・本人の決定ア・画面はイ）。

窓の門は文頭の 1 字違いまで名前とみなす（`names_me`）が、文は書き起こしのまま O・調停・主LLM に渡っていた。
10/08 18:16「バージュ音楽をかけて」に、主LLM が「『バージュ』っていう曲かアーティストですか？」と聞き返し、
「パシュー！おはよう！」には「パシューって呼び方もいいですね！」と返した。門を通って名前で呼ばれたら、聞き違えた
部分だけ当たった名前に直す。画面には聞こえたままの形に、直した名前を括弧で添える（`バージュ（パジュ）音楽をかけて`）。
長音とかなの違いだけ（「パジュー」「ぱじゅ」）は呼び方の違いなので直さない。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core.wake_window import fix_name

NAMES = ["パジュ"]


@pytest.mark.parametrize(
    ("heard", "fixed", "shown"),
    [
        ("バージュ音楽をかけて", "パジュ音楽をかけて", "バージュ（パジュ）音楽をかけて"),
        ("パシュー！おはよう！", "パジュ！おはよう！", "パシュー（パジュ）！おはよう！"),
        ("、バジュ、止めて", "、パジュ、止めて", "、バジュ（パジュ）、止めて"),
    ],
)
def test_a_misheard_name_is_fixed(heard, fixed, shown):
    assert fix_name(heard, NAMES) == (fixed, shown)


@pytest.mark.parametrize(
    "heard",
    [
        "パジュ、音楽をかけて",
        "パジュー、明日の天気は?",  # 長音だけの違い
        "パージュ、明日の天気は?",  # 長音だけの違い（途中でも）
        "ぱじゅ、おはよう",  # かなの違いだけ
        "音楽をかけて",  # 文頭に名前が無い
        "今日はパジュと遊んだ",  # 文頭以外
    ],
)
def test_the_rest_is_left_as_is(heard):
    assert fix_name(heard, NAMES) == (heard, heard)


def test_no_names_no_change():
    assert fix_name("バージュ音楽をかけて", []) == ("バージュ音楽をかけて", "バージュ音楽をかけて")


def _push(text: str, *, named: bool):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    a.config.agent_names = list(NAMES)
    ip = InformationProcessing(a)
    shown: list[tuple[str, bool]] = []
    ip._on_heard = lambda t, h: shown.append((t, h))  # type: ignore[attr-defined]
    ip._ensure_driver = lambda: None  # type: ignore[method-assign]  # 積まれた入力をそのまま見る

    async def gate(trigger):
        trigger.named = named
        return False

    ip._swallow_if_unheard = gate  # type: ignore[method-assign]

    async def go():
        task = asyncio.ensure_future(ip.push_utterance(text, source="voice"))
        await asyncio.sleep(0.05)
        queued = ip._triggers.get_nowait()
        task.cancel()
        await ip.close()
        return queued.query

    return asyncio.run(go()), shown


def test_the_loop_passes_the_fixed_text_and_shows_both():
    query, shown = _push("バージュ音楽をかけて", named=True)
    assert query == "パジュ音楽をかけて"
    assert shown == [("バージュ（パジュ）音楽をかけて", True)]


def test_an_unnamed_input_is_not_touched():
    query, shown = _push("バージュ音楽をかけて", named=False)
    assert query == "バージュ音楽をかけて"
    assert shown == [("バージュ音楽をかけて", True)]
