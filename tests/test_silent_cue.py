"""返しようがないと決めて黙ったときに、決まった音を鳴らす（出-dc・2026-10-26・本人の決定）。

合図（返事の合図・「えっと」など）は動作が決まった時点で鳴る（出-bg・本人の決定イ）。その後で、軽量LLM の黙る判定
（出-db 段 E）か主LLM の黙る印（出-db 段 F）で黙ると、合図だけ鳴って何も言わなかった。黙ったその時に「黙る音」を 1 回
鳴らす（鳴らす道はイ：その 2 つすべて）。合図を鳴らしていない回（キーボードの言葉・Jev が最初から黙ると決めた回）では
鳴らさない。音は低い和音（ド・ミ・ソ）を 0.4 秒、出だしと終わりを丸くやわらかくしたもの（聞いて決めた・本人）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import reaction_cue as rc

# ── 音 ─────────────────────────────────────────────────


def test_the_silent_cue_has_its_own_sound():
    assert rc.sound_for(rc.SILENT) == rc.SILENT_WAV
    assert rc.SILENT_WAV.name == "react_silent.wav" and rc.SILENT_WAV.exists()


def test_the_sound_is_the_low_soft_chord(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "src"))
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import gen_reaction_sounds as g

    x = g.build_silent()
    assert abs(len(x) / g.RATE - 0.4) < 0.001
    assert abs(x).max() <= 0.35 + 1e-6
    assert abs(x[0]) < 0.01 and abs(x[-1]) < 0.01  # 出だしと終わりはやわらかい（ぶつっと鳴らない）


# ── 軽量LLM の黙る判定：印 ─────────────────────────────────────


def test_a_light_reply_judged_silent_carries_the_mark():
    from tests.test_light_reply_can_stay_silent import _LIGHT, _run, _writer

    d = _run(_LIGHT, _writer(judge="判定：黙る"))
    assert (d.branch, d.text, d.judged_silent) == ("light", "", True)


def test_a_silence_jev_chose_from_the_start_has_no_mark():
    """**反証の側**：Jev が最初から黙ると決めた回は、合図も鳴らしていないので印を立てない。"""
    from familiar_agent.loop.arbiter import Decision

    assert Decision(branch="light", text="").judged_silent is False


# ── 鳴らす ─────────────────────────────────────────────────


def _ip(*, played: bool):
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.cue_played = played
    ip._dif = MagicMock()
    return ip


def test_after_a_cue_the_silent_cue_rings():
    ip = _ip(played=True)
    ip._play_silent_cue("軽量LLM の判定")
    ip._dif.play_cue.assert_called_once_with(rc.SILENT)


def test_without_a_cue_nothing_rings():
    """**反証の側**：合図が鳴っていない回（キーボードの言葉・Jev が時間切れ）では鳴らさない。"""
    ip = _ip(played=False)
    ip._play_silent_cue("軽量LLM の判定")
    ip._dif.play_cue.assert_not_called()


def _closed_early(judged: bool):
    from familiar_agent.loop.arbiter import Decision
    from familiar_agent.loop.iteration import Iteration

    ip = MagicMock()
    ip._agent.config.event_max_iterations = 5
    ip._request_generation = 1
    ip._req.trigger_kind = "発話"
    ip._finish = AsyncMock()
    it = Iteration(ip)
    it.decision = Decision(branch="light", text="", judged_silent=judged)
    it.ws = SimpleNamespace(memories=[], returned_actions=frozenset())
    assert asyncio.run(it._close_early()) == ""
    return ip


def test_the_iteration_rings_when_the_writer_judged_silent():
    _closed_early(True)._play_silent_cue.assert_called_once()


def test_the_iteration_does_not_ring_when_jev_chose_silence():
    """**反証の側**：Jev が最初から黙ると決めて閉じる回は鳴らさない。"""
    _closed_early(False)._play_silent_cue.assert_not_called()


def test_the_main_llm_silence_mark_rings():
    """本物の判定を通す。主LLM の呼び出しは `lookups` に「主LLM」として積まれるが、道具ではないので鳴らす。"""
    from familiar_agent.loop.request import Lookup
    from tests.test_bare_text_is_sent_back_for_say import _act, _bare, _with_jev

    ip, _a = _with_jev("reply")
    ip._req.cue_played = True
    ip._req.lookups.append(Lookup(index=1, action="主LLM", query="主LLM1", generation=0))
    ip._dif = MagicMock()
    assert _act(ip, _bare("（黙る）")) == ""
    ip._dif.play_cue.assert_called_once_with(rc.SILENT)


def test_other_bare_text_does_not_ring():
    """**反証の側**：黙る印の無い素の文（差し戻し）では鳴らさない。"""
    from tests.test_bare_text_is_sent_back_for_say import _act, _bare, _with_jev

    ip, _a = _with_jev("reply")
    ip._play_silent_cue = MagicMock()
    _act(ip, _bare())
    ip._play_silent_cue.assert_not_called()


# --- 見直しの指摘（1 回目）---


def test_no_silent_cue_after_a_tool_was_used():
    """道具を使った求め（頼まれたことをやり終えて黙る）では鳴らさない。"""
    from familiar_agent.loop.request import Lookup

    ip = _ip(played=True)
    ip._req.lookups.append(Lookup(index=1, action="play_music", query="x", generation=0))
    ip._play_silent_cue("主LLM の印")
    ip._dif.play_cue.assert_not_called()


def test_the_waiting_beeps_stop_when_staying_silent():
    ip = _ip(played=False)
    ip._stop_beeps = MagicMock()
    ip._play_silent_cue("軽量LLM の判定")
    ip._stop_beeps.assert_called_once()


def test_the_played_mark_is_set_only_when_a_cue_really_rang():
    from familiar_agent.loop.event_loop import _ReactionCue

    ip = _ip(played=False)
    ip._start_beeps = MagicMock()
    ip._dif.play_cue = MagicMock(return_value=None)  # 鳴らせなかった
    _ReactionCue(ip, MagicMock()).decided("reply_light")
    assert ip._req.cue_played is False
    ip._dif.play_cue = MagicMock(return_value=MagicMock())
    _ReactionCue(ip, MagicMock()).decided("reply_light")
    assert ip._req.cue_played is True


def test_the_silent_sound_is_listed_with_the_others():
    assert rc.SILENT_WAV in rc.sound_files()


# --- 見直しの指摘（2 回目）---


def test_a_late_decision_of_an_old_request_does_not_mark_the_new_one():
    from familiar_agent.loop.event_loop import _ReactionCue

    ip = _ip(played=False)
    ip._start_beeps = MagicMock()
    cue = _ReactionCue(ip, MagicMock())
    ip._request_generation += 1  # 決まる前に打ち切られ、次の求めになった
    ip._dif.play_cue = MagicMock(return_value=MagicMock())
    cue.decided("reply_light")
    assert ip._req.cue_played is False


# --- 見直しの指摘（3 回目）---


def test_an_old_decision_plays_no_cue_and_no_beeps():
    from familiar_agent.loop.event_loop import _ReactionCue

    ip = _ip(played=False)
    ip._start_beeps = MagicMock()
    cue = _ReactionCue(ip, MagicMock())
    ip._request_generation += 1
    cue.decided("reply_light")
    ip._dif.play_cue.assert_not_called()
    ip._start_beeps.assert_not_called()


def test_a_cue_that_could_not_start_keeps_the_sounding_one():
    ip = _ip(played=True)
    ip._dif.play_cue = MagicMock(return_value=None)
    playing = MagicMock()
    ip._cue_sound = playing
    ip._play_silent_cue("軽量LLM の判定")
    assert ip._cue_sound is playing
