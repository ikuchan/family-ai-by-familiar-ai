"""声で話者を決める——ループへの結線（知-ae 段 4・2026-10-02・`設計方針_声で話者を見分ける` v0.1）。

門を通った声の入力で、在席（カメラ）があれば照らし、`core/voice_speaker.decide` の結果を当てる。

- **付け替える**：`_set_speaker`（名乗りと同じ効き）。
- **分からない**：話者を既定の人に戻す（本人の決定イ・顔は使わない）。
- **続ける・照らせない**：何もしない。
- 厳しい閾値で当たった発話の声は今日の声に足す（登録の声には足さない）。
- 窓の外で捨てた声・在席の無いとき・キーボード・特徴の無い声では何も起きない。
- その発話の特徴は求めに持たせる（名乗りが話者に付いたときに使う・段 5）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from familiar_agent.config import RecognitionConfig
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent

FAMILY = (
    "## パパ\n- **名前**：雄輔\n- **呼び方**：パパ、ゆうすけ\n\n"
    "## たいき\n- **名前**：泰輝\n- **呼び方**：たいき\n"
)
ROWS = [{"id": "papa", "name": "雄輔"}, {"id": "taiki", "name": "泰輝"}]
PAPA = np.asarray([1.0, 0.0], dtype=np.float32)
TAIKI = np.asarray([0.0, 1.0], dtype=np.float32)


class _Store:
    def __init__(self, registered=None, today=None):
        self.registered = registered or {}
        self.today = today or {}
        self.added: list = []

    def centroids(self, kind, origin, *, day=None):
        assert kind == "voice"
        return dict(self.registered if origin == "registered" else self.today)

    def add(self, person_id, kind, origin, vec, *, cap, day=None, now=None):
        self.added.append((person_id, kind, origin, cap))

    def drop_old_today(self, today):
        return 0


def _ip(*, present: float = 1.0, speaker: "str | None" = "パパ", store=None):
    a = _agent(stream_returns=[])
    a._family_md = FAMILY
    a.config.recognition = RecognitionConfig()  # 閾値 0.25・0.35・今日の声 10
    a._occupancy = MagicMock(return_value=present)
    a._persons.active_name = speaker or "誰か"
    a._persons.active_is_explicit = speaker is not None
    a._sync_pmm_speaker = AsyncMock()
    a._pmm = MagicMock()
    a._pmm.list_persons = MagicMock(return_value=list(ROWS))
    a._pmm.find_person_id_by_name = MagicMock(
        side_effect=lambda n: {"パパ": "papa", "たいき": "taiki"}.get(n)
    )
    ip = InformationProcessing(a)
    s = store if store is not None else _Store(registered={"papa": PAPA, "taiki": TAIKI})
    ip._voice_store = lambda: s  # type: ignore[method-assign]
    return ip, a, s


def test_a_clear_other_voice_takes_over_and_feeds_today():
    ip, a, s = _ip()
    asyncio.run(ip._match_voice(TAIKI))
    a._persons.set_active.assert_called_once_with("たいき")  # 呼びかけ名で付ける
    a._sync_pmm_speaker.assert_awaited_once_with("たいき")
    assert s.added == [("taiki", "voice", "today", 10)]


def test_the_same_voice_keeps_and_a_clear_match_feeds_today():
    ip, a, s = _ip()
    asyncio.run(ip._match_voice(PAPA))
    a._persons.set_active.assert_not_called()
    a._persons.reset_to_default.assert_not_called()
    assert s.added == [("papa", "voice", "today", 10)]


def test_a_loose_match_keeps_without_feeding_today():
    ip, a, s = _ip()
    loose = np.asarray([0.3, 0.954], dtype=np.float32)  # パパに 0.30（緩い）・たいきに 0.954 だが…
    s.registered = {"papa": PAPA}  # たいきの基準は無い
    asyncio.run(ip._match_voice(loose))
    a._persons.set_active.assert_not_called()
    a._persons.reset_to_default.assert_not_called()
    assert s.added == []


def test_an_unknown_voice_returns_to_the_default_person():
    ip, a, s = _ip(store=_Store(registered={"papa": PAPA}))
    asyncio.run(ip._match_voice(np.asarray([-1.0, 0.0], dtype=np.float32)))
    a._persons.reset_to_default.assert_called_once()
    a._pmm.clear_speaker.assert_called_once()
    assert s.added == []


def test_nothing_happens_without_presence_or_a_voice():
    ip, a, s = _ip(present=0.0)
    asyncio.run(ip._match_voice(TAIKI))
    ip2, a2, s2 = _ip()
    asyncio.run(ip2._match_voice(None))
    for agent, store in ((a, s), (a2, s2)):
        agent._persons.set_active.assert_not_called()
        agent._persons.reset_to_default.assert_not_called()
        assert store.added == []


def test_no_voices_at_all_change_nothing():
    ip, a, s = _ip(store=_Store())
    asyncio.run(ip._match_voice(TAIKI))
    a._persons.set_active.assert_not_called()
    a._persons.reset_to_default.assert_not_called()


def test_from_the_default_person_a_clear_voice_names_someone():
    ip, a, s = _ip(speaker=None)
    asyncio.run(ip._match_voice(PAPA))
    a._persons.set_active.assert_called_once_with("パパ")


# ── push_utterance：門を通った声だけを照らし、特徴を求めへ運ぶ ────────────────


def _pushed(ip, *, swallowed: bool, voice):
    ip._swallow_if_unheard = AsyncMock(return_value=swallowed)  # type: ignore[method-assign]
    ip._match_voice = AsyncMock()  # type: ignore[method-assign]
    ip._ensure_driver = lambda: None  # type: ignore[method-assign]
    seen: list = []

    def put(trigger):
        seen.append(trigger)
        trigger.future.set_result("")

    ip._triggers = MagicMock()
    ip._triggers.put_nowait = put
    asyncio.run(ip.push_utterance("パパだよ", source="voice", voice=voice))
    return seen


def test_a_swallowed_voice_is_not_judged():
    ip, _, _ = _ip()
    assert _pushed(ip, swallowed=True, voice=PAPA) == []
    ip._match_voice.assert_not_awaited()


def test_an_admitted_voice_is_judged_and_rides_the_trigger():
    ip, _, _ = _ip()
    seen = _pushed(ip, swallowed=False, voice=PAPA)
    ip._match_voice.assert_awaited_once()
    assert np.allclose(ip._match_voice.await_args.args[0], PAPA)
    assert np.allclose(seen[0].voice, PAPA)


def test_the_request_keeps_the_voice_of_its_utterance():
    ip, _, _ = _ip()
    ip._write_origin = AsyncMock(return_value="obs-1")  # type: ignore[method-assign]
    ip._note_origin = MagicMock()  # type: ignore[method-assign]
    ip._notify_request_state = MagicMock()  # type: ignore[method-assign]
    ip._pending_voice = PAPA  # `_utterance_iteration` が置く
    asyncio.run(ip._begin_request(kind="発話", text="パパだよ", utterance="パパだよ"))
    assert np.allclose(ip._req.voice, PAPA)
    asyncio.run(ip._begin_request(kind="情動", text="x"))
    assert ip._req.voice is None


def test_run_hands_the_voice_to_the_loop():
    from familiar_agent.agent import EmbodiedAgent as Agent
    from familiar_agent.core.wake_window import VoiceText
    from tests.test_input_commands_before_loop_branch import _agent as _run_agent

    a = _run_agent()
    asyncio.run(Agent.run(a, VoiceText("パパだよ", voice=PAPA)))
    assert np.allclose(a._info_processing.push_utterance.await_args.kwargs["voice"], PAPA)
