"""話者の指定に寿命を持たせる（知-t・2026-09-18・`知覚在席` §3-2b）。

`/speaker パパ` から 60 秒過ぎ、在席表にもその人が居なければ「誰か分からない」に戻す。
戻ると：想起は共通の面（パジュ自身）、system 文に `(present :speaker …)` を出さない（名前を
呼ばない）、面の材料にその人を立てない、話者ゲート・タイマーの `speaker` も空。
その人に返事するたび 60 秒延びる（会話中は切れない）。09-17 の「話者の指定は残す」は撤回。
実機 2026-09-18 12:34：`/speaker` から 2 分 22 秒・在席表は空なのに、想起をパパの面で引き
「パパ」と呼びかけ、版に「パパに聞かれ」と書いた（誰か分からない相手をパパとして扱った）。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.loop.tonic import Tonic


def _agent(*, set_at=None, confirmed_at=None, present_ids=(), speaker_id="id:パパ"):
    a = MagicMock(spec=Agent)
    a.config = MagicMock()
    a.config.presence_said_sec = 60.0
    a._pmm = MagicMock()
    a._pmm.current_speaker_id = speaker_id
    a._pmm.get_present_ids = MagicMock(return_value=list(present_ids))
    a._persons = MagicMock()
    a._persons.active_is_explicit = speaker_id is not None
    a._persons.active_name = "パパ"
    for attr, v in (("_speaker_set_at", set_at), ("_speaker_confirmed_at", confirmed_at)):
        if v is None:
            with __import__("contextlib").suppress(AttributeError):
                delattr(a, attr)
        else:
            setattr(a, attr, v)
    a.speaker_known = lambda: Agent.speaker_known(a)
    return a


# ── 切れる判定は 1 箇所 ───────────────────────────────────────────────────


def test_known_within_a_minute_of_speaker_command():
    assert _agent(set_at=time.time() - 30).speaker_known() is True
    assert _agent(set_at=time.time() - 70).speaker_known() is False


def test_replying_to_that_person_extends_it():
    assert _agent(set_at=time.time() - 300, confirmed_at=time.time() - 20).speaker_known() is True
    assert _agent(set_at=time.time() - 300, confirmed_at=time.time() - 90).speaker_known() is False


def test_a_matched_face_in_the_table_keeps_it_known():
    assert _agent(set_at=time.time() - 300, present_ids=("id:パパ",)).speaker_known() is True


def test_no_speaker_at_all_is_unknown():
    assert _agent(speaker_id=None, set_at=time.time()).speaker_known() is False


# ── 切れたら戻す（T の tick）────────────────────────────────────────────────


def test_the_tick_resets_an_expired_speaker():
    a = _agent(set_at=time.time() - 120)
    ip = MagicMock()
    t = Tonic(ip)
    t._agent = a
    t._expire_speaker()
    a._persons.reset_to_default.assert_called_once()
    a._pmm.clear_speaker.assert_called_once()


def test_the_tick_leaves_a_known_speaker_alone():
    a = _agent(set_at=time.time() - 10)
    t = Tonic(MagicMock())
    t._agent = a
    t._expire_speaker()
    a._persons.reset_to_default.assert_not_called()


# ── 読み手が同じ判定を通る ─────────────────────────────────────────────────


def test_recall_uses_the_shared_face_when_unknown():
    a = _agent(set_at=time.time() - 120)
    a._active_memory = lambda: Agent._active_memory(a)
    a._active_memory()
    a._pmm.get_agent_memory.assert_called_once()
    a._pmm.get_speaker_memory.assert_not_called()


def test_the_conversation_perspective_does_not_name_an_unknown_speaker():
    from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID

    a = _agent(set_at=time.time() - 120)
    a._conversation_perspective = lambda: Agent._conversation_perspective(a)
    assert a._conversation_perspective()["writer_id"] == DEFAULT_PERSON_ID


def test_the_system_prompt_stops_naming_an_unknown_speaker():
    from familiar_agent.loop.generator import _present_ctx

    a = _agent(set_at=time.time() - 120)
    a._pmm.presence_status = MagicMock(return_value=[])
    a._presence_sensor = None
    ctx = _present_ctx(a)
    assert "パパ" not in ctx
    b = _agent(set_at=time.time() - 10)
    b._pmm.presence_status = MagicMock(return_value=[])
    b._presence_sensor = None
    assert "パパ" in _present_ctx(b)
