"""求めの途中で生まれた確認待ちも、次の反復の W に載る（出-ag-ろ 穴 2・2026-09-25）。

実機 2026-09-21 17:34：「3分測って」→ 反復 1 で `set_timer` が確認待ちを置いた。ところが
`[確認待ち]` の枠は**求めを始めたときに 1 回だけ**写していたので、同じ求めの反復 2〜5 の W に
一度も載らなかった。問いが見えたのは、道具が返った直後の 1 反復（`[いま道具から返った]`）だけ。
上限の反復で調停は枠の無い W を見て、`light` で「タイマーをセットしました」と言った。

枠は反復ごとに写す。途中で生まれた預かりも、次の反復から W の最上部に載る。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

ASK = "3 分のタイマーね。その間は黙って聞かないよ、いい？"
#: 枠そのものの書き出し。候補の説明文（`confirm`／`decline`）にも「[確認待ち]」の語はあるので、
#: 語だけで見ると枠が無くても通ってしまう（下書きで実際に通った）。
FRAME_HEAD = "[確認待ち] タイマーを掛ける"


def _arbiter_prompts(*, timer_leaves_a_confirm: bool) -> "list[str]":
    """「3分測って」で求めを始め、調停へ渡したプロンプトを順に返す。

    反復 1 で調停が `set_timer` を選び、道具が（`timer_leaves_a_confirm` なら）確認待ちを置く。
    反復 2 の調停は `light` で閉じる。
    """
    a = _agent(stream_returns=[])
    a._timer_tool.get_tool_definitions = MagicMock(
        return_value=[
            {"name": n} for n in ("set_timer", "cancel_timer", "pause_timer", "resume_timer")
        ]
    )
    a._alarm_tool = None
    a._stopwatch_tool = None
    a._pending_confirm = None
    pc = SimpleNamespace(what="タイマーを掛ける「タイマー」（3 分）", text=ASK)
    a.confirm_frame = lambda: f"[確認待ち] {pc.what}：「{ASK}」" if a._pending_confirm else ""
    a.confirm_alive = lambda: a._pending_confirm is not None

    async def timer_call(action, tool_input, now=None, **kw):
        if timer_leaves_a_confirm:
            a._pending_confirm = pc
            return f"まだ掛けていない。本人に一度聞く：「{ASK}」", True
        return "3 分のタイマーを掛けた", True

    a._timer_tool.call = timer_call
    prompts: list[str] = []

    async def complete(prompt, *_a, **_k):
        if '"branch"' not in prompt:
            return "OK"  # 調停でない軽量LLM の仕事（申告など）
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                '{"branch":"action","action":"set_timer",'
                '"tool_input":{"after_minutes":3,"label":"タイマー"}}'
            )
        return '{"branch":"light","text":"3 分のタイマーね、いい？"}'

    a._utility_backend.complete = complete

    async def scenario():
        ip = InformationProcessing(a)
        await ip.push_utterance("3分測って", on_text=lambda _t: None)
        for _ in range(400):
            if len(prompts) >= 2:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        await ip.close()

    asyncio.run(scenario())
    return prompts


def test_the_confirm_made_in_iteration_one_is_in_iteration_twos_w():
    prompts = _arbiter_prompts(timer_leaves_a_confirm=True)
    assert len(prompts) >= 2, f"反復 2 の調停まで届いていない：{len(prompts)}"
    assert FRAME_HEAD not in prompts[0]  # 反復 1 ではまだ無い
    assert FRAME_HEAD in prompts[1], "途中で生まれた確認待ちが、次の反復の W に無い"


def test_no_frame_when_no_confirm_was_made():
    """反証：預かりが置かれなければ、枠は出ない。"""
    prompts = _arbiter_prompts(timer_leaves_a_confirm=False)
    assert len(prompts) >= 2
    assert all(FRAME_HEAD not in p for p in prompts)
