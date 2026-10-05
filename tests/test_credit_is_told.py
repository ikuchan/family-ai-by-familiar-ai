"""残高切れを、人が来たら録音済みの声で知らせる（環-z 段 3・2026-10-05・`設計方針_クレジット切れの知らせ` v0.1）。

**専用の情動**（知らせたい・`core/credit.pending`）が立っていて、**居る**（顔ぶれか在席）に気づいたら、その担い手の
録音（`sounds/credit_<担い手>.wav`）を鳴らす。LLM も TTS も通さない（切れているかもしれない）。

- 居ない → 居る に変わるたびに伝える。立った時点ですでに居れば、すぐ伝える。居続けているあいだは繰り返さない。
- 門は情動と同じ：「黙っていて」のあいだは控え、明けたら伝える。夜も鳴らす。
- 回復したもの（消えた知らせたい）は鳴らさない。録音が無くても落ちない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import credit


@pytest.fixture
def world(monkeypatch):
    w = {"alerts": {"gemini": "t"}, "here": True, "silenced": False}
    monkeypatch.setattr(credit, "pending", lambda: dict(w["alerts"]))
    monkeypatch.setattr("familiar_agent.loop.tonic._silenced_now", lambda: w["silenced"])
    return w


def _tonic(world):
    from familiar_agent.loop.tonic import Tonic

    agent = MagicMock()
    agent.config.tts_gain = 0.25
    t = Tonic(MagicMock(), agent=agent)
    t._someone_here = lambda: world["here"]  # type: ignore[method-assign]
    t._dif = MagicMock()
    t._dif.say_credit = AsyncMock(return_value=True)
    return t


def _tick(t):
    asyncio.run(t._maybe_tell_credit())


def test_it_is_told_at_once_when_someone_is_here(world):
    t = _tonic(world)
    _tick(t)
    t._dif.say_credit.assert_awaited_once_with("gemini", gain=0.25)


def test_it_is_not_repeated_while_they_stay(world):
    t = _tonic(world)
    _tick(t)
    _tick(t)
    _tick(t)
    assert t._dif.say_credit.await_count == 1


def test_it_waits_while_nobody_is_here_and_tells_whoever_comes(world):
    world["here"] = False
    t = _tonic(world)
    _tick(t)
    t._dif.say_credit.assert_not_awaited()
    world["here"] = True
    _tick(t)
    assert t._dif.say_credit.await_count == 1


def test_leaving_and_coming_back_tells_again(world):
    t = _tonic(world)
    _tick(t)
    world["here"] = False
    _tick(t)
    world["here"] = True
    _tick(t)
    assert t._dif.say_credit.await_count == 2


def test_it_holds_back_while_asked_to_be_quiet_and_tells_after(world):
    world["silenced"] = True
    t = _tonic(world)
    _tick(t)
    t._dif.say_credit.assert_not_awaited()
    world["silenced"] = False
    _tick(t)
    assert t._dif.say_credit.await_count == 1


def test_a_recovered_provider_is_not_told(world):
    world["alerts"] = {}
    t = _tonic(world)
    _tick(t)
    t._dif.say_credit.assert_not_awaited()


def test_each_provider_is_told_once(world):
    world["alerts"] = {"gemini": "t", "anthropic": "t"}
    t = _tonic(world)
    _tick(t)
    _tick(t)
    names = sorted(c.args[0] for c in t._dif.say_credit.await_args_list)
    assert names == ["anthropic", "gemini"]


def test_running_out_again_after_recovering_is_told_again(world):
    t = _tonic(world)
    _tick(t)
    world["alerts"] = {}
    _tick(t)
    world["alerts"] = {"gemini": "t2"}
    _tick(t)
    assert t._dif.say_credit.await_count == 2


def test_the_tick_tells(world):
    import inspect

    from familiar_agent.loop import tonic

    assert "_maybe_tell_credit()" in inspect.getsource(tonic.Tonic._run)


# ── 録音を鳴らす口 ────────────────────────────────────────────────────────────


def test_the_recording_is_played_by_provider_name(monkeypatch, tmp_path):
    from familiar_agent.io import dif

    (tmp_path / "credit_gemini.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(dif, "_SOUNDS", tmp_path)
    played = AsyncMock(return_value=True)
    monkeypatch.setattr(dif, "_play_wav", played)
    d = dif.DIF(ip=MagicMock())
    assert asyncio.run(d.say_credit("gemini", gain=0.25)) is True
    assert played.await_args.args == (tmp_path / "credit_gemini.wav", 0.25)
    assert asyncio.run(d.say_credit("kimi", gain=0.25)) is False  # 録音が無くても落ちない


# ── 録音（環-z 段 4）─────────────────────────────────────────────────────────


def test_every_provider_has_a_reading_and_a_recording():
    """担い手ごとに読み上げる名前があり、録音（`scripts/gen_credit_voices.py` が作る）がそろっている。"""
    import pathlib

    from familiar_agent.backends.anthropic import AnthropicBackend
    from familiar_agent.backends.gemini import GeminiBackend
    from familiar_agent.backends.glm import GLMBackend
    from familiar_agent.backends.kimi import KimiBackend
    from familiar_agent.backends.openai_compat import OpenAICompatibleBackend

    names = {
        b.credit_name
        for b in (AnthropicBackend, GeminiBackend, GLMBackend, KimiBackend, OpenAICompatibleBackend)
    } | {"jev"}
    assert set(credit.READINGS) == names
    assert credit.READINGS["anthropic"] == "クロード" and credit.READINGS["jev"] == "ジェブ"
    sounds = pathlib.Path(__file__).resolve().parents[1] / "src" / "familiar_agent" / "sounds"
    for name in names:
        wav = sounds / f"credit_{name}.wav"
        assert wav.exists() and wav.stat().st_size > 10_000, wav.name
        assert wav.read_bytes()[:4] == b"RIFF"


def test_the_sentence():
    assert credit.sentence("gemini") == "ジェミニのクレジットが足りなくなりました。"
