"""黙る依頼は窓の中で受け、原則 30 分・名前と「話していいよ」でだけ解く（出-as 段 5・2026-09-26・`設計方針_話していいかの決まり` §2.5）。

今朝の実機で「黙って」は 5 回、受け付けられなかった——名前が無い（「30分だまってて」）、話者が分からない
（「誰からか分からないので受けない」）。ウェイクワードの窓を入れたので、黙る依頼にかかっていた名前と話者の関門は
窓にまとめる。

- 受ける：窓の中の入力なら（入口で窓を見ている）、話者が分からなくても受ける。
- 長さ：言われなければ 30 分（いまの既定 60 分から変える）。
- 解く：**名前と「話していいよ」が同じ発話にあるときだけ**、誰でも。窓が開いているだけでは解かない。
- 誰も居なくなっても解けない（60 秒で解けていた）。
- タイマー由来の沈黙はいまのまま（鳴る・止めるまで。操作の言葉は通す）。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from familiar_agent.core.silence_hold import lifts
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.silence_state import SilenceRequest, is_silenced

from tests.test_event_loop import _agent

NAMES = ["パジュ"]


def _ip(speaker: str = ""):
    a = _agent(stream_returns=[])
    a.config.agent_names = NAMES
    a.config.silence_minutes = 30
    a.config.silence_max_minutes = 60
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: speaker
    return ip


def test_the_default_is_thirty_minutes(monkeypatch):
    from familiar_agent.config import AgentConfig

    monkeypatch.delenv("SILENCE_MINUTES", raising=False)
    assert AgentConfig().silence_minutes == 30


def test_a_request_is_taken_even_when_the_speaker_is_unknown(monkeypatch):
    saved = []
    monkeypatch.setattr("familiar_agent.silence_state.save_silence", saved.append)
    ip = _ip(speaker="")
    ip._apply_silence(SimpleNamespace(silence_minutes=-1, lift_silence=False), utterance="黙ってて")
    assert len(saved) == 1
    assert 29 * 60 < saved[0].until - time.time() <= 30 * 60


def test_a_request_without_the_name_is_taken_inside_the_window(monkeypatch):
    """名前の関門は窓にまとめた（キーボードの「30分だまってて」も受ける）。"""
    saved = []
    monkeypatch.setattr("familiar_agent.silence_state.save_silence", saved.append)
    ip = _ip(speaker="パパ")
    ip._apply_silence(
        SimpleNamespace(silence_minutes=30, lift_silence=False), utterance="30分だまってて"
    )
    assert len(saved) == 1 and saved[0].person == "パパ"


# ── 黙っているあいだに通す言葉 ─────────────────────────────────────────────


def test_the_name_with_a_release_lifts_for_anyone():
    assert lifts("会話入力", "パジュ、話していいよ", names=NAMES, reason="")


def test_a_release_without_the_name_does_not_lift():
    assert not lifts("会話入力", "話していいよ", names=NAMES, reason="")


def test_a_stop_word_no_longer_lifts_an_explicit_silence():
    """頼んだ本人の「止めて」を通す決まりは外した。解く言葉は名前と「話していいよ」だけ。"""
    assert not lifts("会話入力", "パジュ、止めて", names=NAMES, reason="")


def test_timer_rules_are_unchanged():
    assert lifts("機器", "タイマー", names=NAMES, reason="timer:3")
    assert lifts("会話入力", "止めて", names=NAMES, reason="timer:3")


# ── 解く ───────────────────────────────────────────────────────────────────


def test_anyone_can_lift_an_explicit_silence(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "familiar_agent.silence_state.load_silence",
        lambda: SilenceRequest(person="パパ", until=time.time() + 600),
    )
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: cleared.append(1))
    _ip(speaker="")._release_silence()
    assert cleared == [1]


def test_a_timer_silence_is_not_lifted_by_words(monkeypatch):
    cleared = []
    monkeypatch.setattr(
        "familiar_agent.silence_state.load_silence",
        lambda: SilenceRequest(person="パパ", until=time.time() + 600, reason="timer:3"),
    )
    monkeypatch.setattr("familiar_agent.silence_state.clear_silence", lambda: cleared.append(1))
    _ip(speaker="パパ")._release_silence()
    assert cleared == []


def test_nobody_around_does_not_lift_it():
    """居るかの材料を受ける口そのものを外した。解くのは期限と名前つきの「話していいよ」だけ。"""
    import inspect

    assert "nobody_since" not in inspect.signature(is_silenced).parameters
    now = time.time()
    assert is_silenced(SilenceRequest(person="パパ", until=now + 1800), now=now)
