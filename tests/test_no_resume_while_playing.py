"""鳴っているあいだに、曲名の無い「かける」を選ばない（知-bq・2026-11-02・本人の決定ア）。

11/02 12:52:48、音楽が鳴っているあいだの「パジュ、止めて」が「はじめまして」と書き起こされた。名前は 1 字違いとして拾えたが
「止めて」は「まして」になり、調停は意味を音楽（0.40）、動作をかける 0.56・止める 0.27 と読み、いちばん確率の高い
`play_music` を選んだ。曲名が無いので「止まっていた続きをかける」になり、鳴っている曲をかけ直した。鳴っているときには
意味の無い動作なので、選べる候補は絞らず、選ばれたときだけ `play_music` を外して決まり（`utterance_meaning.decide`）で
決め直す。外したあとは、いまの決まりのとおり割り直す（12:52 なら止める・本人の決定ア）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, writer_says
from tests.test_arbiter_utterance_by_meaning import _jev

MUSIC = ("play_music", "stop_music", "next_track", "music_volume")


def _p(**probs: float) -> dict:
    top = max(probs, key=lambda k: probs[k])
    return {"choice": top, "confidence": probs[top], "probabilities": probs}


#: 12:52 の Jev の答え（`arbiter_records`）。
MEANING_1252 = _p(
    music=0.40, accepted_check=0.17, off_context=0.13, speaking=0.12, unformed=0.05, other=0.04
)
MUSIC_1252 = dict(
    silent=0.04,
    ask_back=0.02,
    music_now=0.02,
    next_track=0.03,
    play_music=0.56,
    stop_music=0.27,
    music_volume=0.06,
)


def _run(monkeypatch, action: dict, *, playing: bool, name: str = "", seen=None):
    """`seen` を渡すと、記録に残した最終の動作（`final`）と、鳴らした合図の動作（`cue`）を入れる。"""
    kept: list = []
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: kept.append(kw["final"]))
    jev = _jev({"meaning": MEANING_1252, "action_music": action})
    writer = writer_says({"text": "止めますか？", "tool_input": {"name": name}})
    cues: list = []
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュまして",
            origin="発話",
            extra_actions=MUSIC,
            music_playing=playing,
            on_decided=cues.append,
        )
    )
    if seen is not None:
        seen["final"], seen["cue"] = kept, cues
    return d


def test_a_nameless_play_while_playing_is_read_again(monkeypatch):
    d = _run(monkeypatch, _p(**MUSIC_1252), playing=True)
    assert (d.branch, d.action) == ("action", "stop_music")


@pytest.mark.parametrize(
    "action",
    [
        _p(play_music=0.92, stop_music=0.05, silent=0.03),  # ほかに 0.1 以上が残らない
        _p(
            play_music=0.25, stop_music=0.20, music_volume=0.20, next_track=0.20, silent=0.15
        ),  # 割れて聞き返す
    ],
)
def test_the_heard_check_never_keeps_play(monkeypatch, action):
    """聞き返すとき、預ける見立てを play_music にしない（「はい」でかけ直さない）。"""
    d = _run(monkeypatch, action, playing=True)
    assert d.action != "play_music"
    assert d.text  # 聞き返している
    assert not d.heard_check or d.heard_check.get("action") != "play_music"


def test_scraps_are_not_kept_as_the_heard_action(monkeypatch):
    """ほかに 0.1 以上が残らなければ、見立て無しで聞き返す（0.05 の「止める」を「はい」で動かさない）。"""
    d = _run(monkeypatch, _p(play_music=0.92, stop_music=0.05, silent=0.03), playing=True)
    assert d.text and d.heard_check is None


def test_a_failed_writer_is_not_read_as_no_name(monkeypatch, caplog):
    """**反証の側**：書き手が失敗して主LLM へ倒れたときは、曲名が無いとは限らないので決め直さない。"""
    caplog.set_level("INFO", logger="familiar_agent.loop.arbiter")
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"meaning": MEANING_1252, "action_music": _p(**MUSIC_1252)})
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says(raises=RuntimeError("書けなかった")),
            utterance="パジュ、ある曲かけて",
            origin="発話",
            extra_actions=MUSIC,
            music_playing=True,
        )
    )
    assert d.branch == "full" and "読み直した" not in caplog.text


def test_without_removes_the_choice_and_its_confidence():
    from familiar_agent.core import utterance_meaning as um

    got = um.without(_p(play_music=0.56, stop_music=0.27), "play_music")
    assert got["choice"] == "stop_music" and got["confidence"] == 0.27
    assert "play_music" not in got["probabilities"]
    assert um.without(_p(play_music=1.0), "play_music") == {}


def test_the_cue_and_the_record_follow_the_reread(monkeypatch):
    """合図と記録は、決め直した動作で（かける合図を鳴らしてから止めない・記録に play_music と残さない）。"""
    seen: dict = {}
    _run(monkeypatch, _p(**MUSIC_1252), playing=True, seen=seen)
    assert seen["cue"] == ["stop_music"] and seen["final"] == ["stop_music"]


def test_a_named_play_still_gets_its_cue(monkeypatch):
    """**反証の側**：曲名があれば、待たせた合図をかけるで鳴らす。"""
    seen: dict = {}
    _run(monkeypatch, _p(**MUSIC_1252), playing=True, name="ある曲", seen=seen)
    assert seen["cue"] == ["play_music"] and seen["final"] == ["play_music"]


def test_not_playing_resumes_as_before(monkeypatch):
    """**反証の側**：鳴っていなければ、止まっていた続きをかける（いまどおり）。"""
    d = _run(monkeypatch, _p(**MUSIC_1252), playing=False)
    assert (d.branch, d.action) == ("action", "play_music")


def test_a_named_play_while_playing_still_plays(monkeypatch):
    """**反証の側**：曲名があれば、鳴っていてもかける（「〇〇かけて」）。"""
    d = _run(monkeypatch, _p(**MUSIC_1252), playing=True, name="ある曲")
    assert (d.branch, d.action, d.query) == ("action", "play_music", "ある曲")


def test_stop_chosen_first_is_unchanged(monkeypatch, caplog):
    """**反証の側**：初めから止めるが選ばれたときは、決め直さない（書き直しも起きない）。"""
    caplog.set_level("INFO", logger="familiar_agent.loop.arbiter")
    d = _run(monkeypatch, _p(stop_music=0.7, play_music=0.3), playing=True)
    assert (d.branch, d.action) == ("action", "stop_music")
    assert "読み直した" not in caplog.text


def test_the_loop_tells_the_arbiter_that_music_is_playing():
    from types import SimpleNamespace
    from unittest.mock import patch

    from familiar_agent.loop import arbiter
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests._arbiter_fakes import ws_of
    from tests.test_event_loop import _agent

    seen: list = []

    async def fake_decide(self, inp, on_decided=None):
        seen.append(inp.music_playing)
        return arbiter.Decision(branch="light", text="うん")

    for playing in (True, False):
        a = _agent(stream_returns=[])
        a._music_state = SimpleNamespace(playing=playing)
        ip = InformationProcessing(a)
        ip._req.trigger_kind = "発話"
        with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
            asyncio.run(
                ip._decide(utterance="止めて", ws=ws_of(), present_ctx="", capped=False, round_=1)
            )
    assert seen == [True, False]


# ── 道具の側でも止める（どの道から呼ばれても・2026-11-02・本人「そこまで実装してしまって」）──────────────
#
# 調停が意味で決めた道だけでなく、主LLM が考えて返す道や、聞き返しに「はい」と答えた道からも、鳴っているあいだに曲名の
# 無い play_music が来うる。道具が鳴っている印を見て、続きをかけ直さずに返す。完了の表で「かけた → 黙る」なので、成功で返す。


def _music_tool(playing: bool, *, spotify_playing: bool = True):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.tools.music import MusicTool

    web = MagicMock()
    web.resume = MagicMock(return_value=True)
    web.now_playing = MagicMock(return_value={"playing": spotify_playing})
    tool = MusicTool(
        io=MagicMock(play=AsyncMock(return_value=True)),
        bus=lambda: MagicMock(),
        table=lambda: (),
        state=SimpleNamespace(playing=playing, started_at=0.0),
        web=web,
        device_name="パジュ",
    )
    return tool, web


@pytest.mark.parametrize("said", ["", "音楽"])
def test_the_tool_does_not_resume_while_playing(said):
    tool, web = _music_tool(True)
    text, ok = asyncio.run(tool.call("play_music", {"name": said}))
    assert ok is True and "鳴っている" in text
    web.resume.assert_not_called()


def test_the_tool_resumes_when_not_playing():
    """**反証の側**：鳴っていなければ、いまどおり止まっていた続きをかける。"""
    tool, web = _music_tool(False)
    text, ok = asyncio.run(tool.call("play_music", {"name": ""}))
    assert ok is True and "続き" in text
    web.resume.assert_called_once()


def test_a_stale_flag_is_checked_with_spotify():
    """**反証の側**：印が古く（外で止めた直後など）Spotify では止まっていれば、いまどおり続きをかける。"""
    tool, web = _music_tool(True, spotify_playing=False)
    web.now_playing.side_effect = [{"playing": False}] + [{"playing": True}] * 20
    text, ok = asyncio.run(tool.call("play_music", {"name": ""}))
    assert ok is True and "続き" in text
    web.resume.assert_called_once()
