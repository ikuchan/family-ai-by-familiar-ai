"""軽量LLM の書き手に、窓の会話を渡す（環-ar 段 C・2026-10-25・本人の決定）。

窓の会話の中なら、書き手の文は [人の言葉] のすぐ前に窓の会話の全部を置き、W から [直近のやりとり] を外す。
[そのあいだに起きたこと] はいまどおり W に載る。記憶の列から外すのは、Jev と同じく窓の会話の行になった記録と
[そのあいだに起きたこと] の行だけ（本人の決定ア）。窓の外では、いまどおり。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent
from tests.test_jev_gets_a_measured_state import _ws
from tests.test_window_talk_goes_to_jev import TALK, _past, _ws_across_the_window

# ── 書き手の文 ─────────────────────────────────────────────────


def _written(window_ctx: str = "") -> str:
    w = MagicMock()
    w.complete = AsyncMock(return_value='{"text": "うん"}')
    inp = ArbiterInput(
        utterance="ごりら",
        workspace_ctx="（W）",
        present_ctx="（顔ぶれ）",
        now_ctx="（いま）",
        window_ctx=window_ctx,
    )
    asyncio.run(Arbiter(jev=None, writer=w, timeout=5.0)._write(inp, {"branch": "light"}))
    return w.complete.await_args.args[0]


def test_the_window_talk_comes_just_before_the_persons_words():
    prompt = _written(TALK)
    assert f"{TALK}\n\n[人の言葉]\nごりら" in prompt
    assert prompt.index("[いま誰が居るか]") < prompt.index(TALK)


def test_with_no_window_the_writer_prompt_is_as_before():
    """**反証の側**：窓の会話が無ければ、[いま誰が居るか] のすぐ後に [人の言葉] が来る。"""
    prompt = _written()
    assert "窓の会話" not in prompt
    assert "[いま誰が居るか]\n（顔ぶれ）\n\n[人の言葉]\nごりら" in prompt


# ── 書き手の W ─────────────────────────────────────────────────


def test_the_writer_view_drops_the_recent_frame_in_the_window():
    """窓の会話は文の別の場所に置くので、W には載せない（2 回載せない）。"""
    ws = _ws()
    ws.window_talk = TALK
    text = ws.for_arbiter
    assert "直近のひとこと" not in text and "窓の会話" not in text


def test_with_no_window_the_writer_view_keeps_the_recent_frame():
    """**反証の側**：窓の外（`window_talk` が None）なら、いまどおり直近 3 往復。"""
    assert "直近のひとこと" in _ws().for_arbiter


def test_the_writer_view_leaves_out_only_what_the_window_talk_shows():
    ws = _ws_across_the_window()
    text = ws.for_arbiter
    assert "窓の前のはなし" in _past(text)  # 窓が開く前のやりとりは記憶の列に戻る
    assert "窓の中のはなし" not in text  # 窓の会話の行は書き手の文に別に載る
    assert text.count("[タイマー] 3 分たった") == 1  # [そのあいだに起きたこと] にだけ


# ── ループが渡す ─────────────────────────────────────────────────


def _decided_input(monkeypatch, *, window: bool):
    from familiar_agent.loop.arbiter import Decision
    from tests._arbiter_fakes import jev_says, writer_says
    from tests.test_iteration_characterization import _run_until

    a = _agent(stream_returns=[])
    a._jev = jev_says("light")
    a._utility_backend = writer_says({"text": ""})
    got: list = []

    async def capture(self, inp, on_decided=None):
        got.append(inp)
        return Decision(branch="light", text="")

    monkeypatch.setattr(Arbiter, "decide", capture)
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: "パパ"
    if window:
        ip._wake_window().open(time.monotonic())
        ip._wake_window().until += 3600  # 反復が回るあいだ開けておく
        ip._talk_line("パパ", "しりとりしよう", obs_id="obs-1")
    _run_until(ip, lambda: got, push=lambda: ip.push_affect("CALM", "落ち着いている"))
    return got[0]


def test_the_iteration_gives_the_window_talk_to_the_writer(monkeypatch):
    inp = _decided_input(monkeypatch, window=True)
    assert inp.window_ctx.startswith("[窓の会話（")
    assert "パパ：しりとりしよう" in inp.window_ctx
    assert "パパ：しりとりしよう" not in inp.workspace_ctx


def test_outside_the_window_the_writer_gets_no_window_talk(monkeypatch):
    """**反証の側**：窓が開いていなければ、書き手に窓の会話は渡らない。"""
    assert _decided_input(monkeypatch, window=False).window_ctx == ""
