"""話しかけたら、聞こえた合図と反応の合図をすぐ流す（出-bg・2026-10-10・本人の決定）。

声で話しかけた言葉が門を通り、Jev が判断を始めたら機械音 A を鳴らし続ける。Jev が決めたら A を止め、決めた動作に合わせて
流す：黙る → なし・軽く返す／聞き返す／状態を伝える → 効果音 B・道具を使う → 作り置きの声（はい・うん・りょ）・考えて返す →
作り置きの声（んー・えっと・うーん）。キーボードの入力では鳴らさない。合図は軽量LLM が文を書く前に出す（遅くならない）。
作り置きの声を流した求めでは、最初のつなぎを流さない（二言目以降は流す）。返事の声は合図が終わってから流す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import reaction_cue as rc
from tests._arbiter_fakes import decide

# ── どの合図を流すか ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("final", "cue"),
    [
        ("silent", rc.NONE),
        ("ask_back", rc.ACK),
        ("reply_light", rc.ACK),
        ("state_light", rc.ACK),
        ("play_music", rc.TOOL),
        ("set_timer", rc.TOOL),
        ("search_deferred", rc.TOOL),
        ("music_suggestion_reply", rc.TOOL),
        ("reply_full", rc.THINK),
    ],
)
def test_each_decision_has_its_cue(final, cue):
    assert rc.cue_for(final) == cue


def test_the_sounds_are_in_the_package():
    files = rc.sound_files()
    assert len(files) == 8  # A・B・道具の声 3・考える声 3
    for path in files:
        assert path.exists(), path


# ── 調停は、軽量LLM より前に決まったことを知らせる ─────────────────────────────


def _jev(answers: dict, ok: bool = True) -> MagicMock:
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return jev


def _run(jev, *, writer_text=None):
    events: list[str] = []
    writer = MagicMock()

    async def complete(*a, **k):
        events.append("writer")
        return writer_text or '{"text": "はい", "tool_input": {"name": "ケイマン"}}'

    writer.complete = AsyncMock(side_effect=complete)
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュ、ケイマンかけて",
            origin="発話",
            extra_actions=("play_music",),
            on_decided=lambda final: events.append(f"decided:{final}"),
        )
    )
    return d, events


def _c(choice, conf=0.9):
    return {"choice": choice, "confidence": conf}


def test_the_decision_is_told_before_the_writer():
    _, events = _run(_jev({"meaning": _c("music"), "action_music": _c("play_music")}))
    assert events[0] == "decided:play_music"
    assert events.count("decided:play_music") == 1
    assert "writer" in events[1:]


def test_silence_is_told_and_nothing_is_written():
    _, events = _run(_jev({"meaning": _c("unformed", 0.2)}))
    assert events == ["decided:silent"]


def test_when_jev_does_not_answer_it_is_thinking():
    _, events = _run(_jev({}, ok=False))
    assert events == ["decided:reply_full"]


# ── 反復：声の発話でだけ A を鳴らし、決まったら止めて合図を流す ──────────────────────


class _FakeDIF:
    def __init__(self):
        self.events: list[str] = []

    def start_thinking(self):
        self.events.append("A 開始")
        return self

    def stop(self):
        self.events.append("A 停止")

    def play_cue(self, cue):
        self.events.append(f"合図 {cue}")


def _ip(*, source: str, kind: str = "発話"):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    fake = _FakeDIF()
    # 本物の声の出口に、鳴らす口だけを差し込む（ほかの口はそのまま使う）。
    ip._dif.start_thinking = fake.start_thinking  # type: ignore[method-assign]
    ip._dif.play_cue = fake.play_cue  # type: ignore[method-assign]
    ip.cue_events = fake.events  # type: ignore[attr-defined]
    ip._req.trigger_kind = kind
    ip._req.source = source
    return ip


def _decide_with(ip, arbiter_decide, **kw):
    from familiar_agent.loop import arbiter as arb

    async def fake(self, inp, on_decided=None):
        return await arbiter_decide(on_decided)

    orig = arb.Arbiter.decide
    arb.Arbiter.decide = fake  # type: ignore[method-assign]
    try:

        async def go():
            try:
                return await ip._decide(
                    utterance="パジュ、ケイマンかけて",
                    workspace_ctx="",
                    present_ctx="",
                    capped=False,
                    round_=1,
                    **kw,
                )
            finally:
                await ip.close()

        return asyncio.run(go())
    finally:
        arb.Arbiter.decide = orig  # type: ignore[method-assign]


def _deciding(final):
    from familiar_agent.loop.arbiter import Decision

    async def run(on_decided):
        if on_decided is not None:
            on_decided(final)
        return Decision(branch="action", action=final, query="ケイマン")

    return run


def test_a_spoken_request_rings_then_cues():
    ip = _ip(source="voice")
    _decide_with(ip, _deciding("play_music"))
    assert ip.cue_events == ["A 開始", "A 停止", f"合図 {rc.TOOL}", "A 停止"]


def test_a_light_reply_cues_the_sound_not_a_voice():
    ip = _ip(source="voice")
    _decide_with(ip, _deciding("reply_light"))
    assert f"合図 {rc.ACK}" in ip.cue_events


def test_silence_stops_the_sound_and_plays_nothing():
    ip = _ip(source="voice")
    _decide_with(ip, _deciding("silent"))
    assert ip.cue_events == ["A 開始", "A 停止", "A 停止"]


@pytest.mark.parametrize(("source", "kind"), [("keyboard", "発話"), ("", "情動"), ("", "機器")])
def test_no_sound_unless_spoken(source, kind):
    ip = _ip(source=source, kind=kind)
    _decide_with(ip, _deciding("play_music"))
    assert ip.cue_events == []


def test_no_sound_when_a_result_came_back():
    ip = _ip(source="voice")
    _decide_with(ip, _deciding("silent"), returned_lookups=(("play_music", False, "かけた"),))
    assert ip.cue_events == []


def test_the_sound_stops_even_when_the_arbiter_fails():
    ip = _ip(source="voice")

    async def boom(on_decided):
        raise RuntimeError("調停が落ちた")

    with pytest.raises(RuntimeError):
        _decide_with(ip, boom)
    assert ip.cue_events == ["A 開始", "A 停止"]


def test_the_sound_rings_only_once_per_request():
    ip = _ip(source="voice")
    _decide_with(ip, _deciding("reply_full"))
    ip.cue_events.clear()
    _decide_with(ip, _deciding("reply_full"))
    assert ip.cue_events == []


# ── つなぎ：作り置きの声のあとも最初のつなぎは省かない（出-bd ②で改めた）──────────────────────
#
# 出-bg では作り置きの声が最初のつなぎの役を済ませるとして省いていた。出-bd ②で、合図の声も音に数え、鳴り終わってから
# 3 秒黙ったらつなぎを出すことにしたので、省かない（`test_filler_gap_after_sound.py` が見る）。


# ── 声の出口：返事の声は合図が終わってから ───────────────────────────────────────


def test_a_reply_waits_for_the_cue(monkeypatch):
    from familiar_agent.io import dif as dif_mod

    order: list[str] = []

    async def slow_play(path, gain):
        order.append("合図 始め")
        await asyncio.sleep(0.05)
        order.append("合図 終わり")
        return True

    monkeypatch.setattr(dif_mod, "_play_wav", slow_play)
    tts = MagicMock()

    async def call(name, payload):
        order.append("返事")
        return "Said: x", None

    tts.call = AsyncMock(side_effect=call)
    d = dif_mod.DIF(tts=tts)

    async def go():
        d.play_cue(rc.ACK)
        await d.speak("はい、パパですね")

    asyncio.run(go())
    assert order == ["合図 始め", "合図 終わり", "返事"]
