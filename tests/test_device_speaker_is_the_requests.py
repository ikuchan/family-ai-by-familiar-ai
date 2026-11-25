"""端-b 段 1-ろ-3：端末で選んだ話者は、その求めの話者にし、家のそばに居る人には数えない（課題9 b3）。

- 端末の求めでは、話者を求め（`Request.speaker`）が持つ。想起の面、会話の記録の書き手、呼びかけの名前、口調（大人か）、
  道具の `speaker` は、その人で決める。求めのあいだは、60 秒の寿命で切れない。
- 家の話者の状態（顔ぶれの表、明示した話者、話者の時刻）には書かない。家の求めは、いままでどおり家の話者を読む。
- 家族に無い名前や、名前の無い端末の求めは「誰か分からない」にする（家の話者を借りない）。
- 端末の言葉で「〇〇じゃない」と言われても、家の顔ぶれを変えない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent as Agent
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.generator import _present_ctx

pytestmark = pytest.mark.real_window

TAB = "tablet-1"
FAMILY = """## ママ
- **名前**：ママ
- **呼び方**：ママ
- **関係**：母（大人）
## むすめ
- **名前**：むすめ
- **呼び方**：むすめ
- **関係**：娘（子ども）
"""


def _agent():
    a = MagicMock()
    a._speaker_confirmed_at = 0.0
    a.config = MagicMock()
    a.config.agent_names = ["パジュ"]
    a.config.speaker_hold_sec = 60.0
    a._family_md = FAMILY
    a._pmm = MagicMock()
    a._pmm.current_speaker_id = "id:パパ"  # 家の話者（家で /speaker したばかり）
    a._pmm.get_present_ids = MagicMock(return_value=["id:パパ"])
    a._pmm.presence_status = MagicMock(
        return_value=[{"person_id": "id:パパ", "name": "パパ", "is_speaker": True}]
    )
    ids = {"ママ": "id:ママ", "むすめ": "id:むすめ"}
    a._pmm.find_person_id_by_name = MagicMock(side_effect=lambda n: ids.get(n))
    a._pmm.get_person_name = MagicMock(side_effect=lambda pid: pid.split(":", 1)[1])
    a._pmm.get_memory_for = MagicMock(side_effect=lambda pid: f"mem:{pid}")
    a._pmm.get_speaker_memory = MagicMock(return_value="mem:id:パパ")
    a._pmm.get_agent_memory = MagicMock(return_value="mem:self")
    a._persons = MagicMock()
    a._persons.active_is_explicit = True
    a._persons.active_name = "パパ"
    a._speaker_set_at = time.time()
    for name in (
        "speaker_known",
        "home_speaker_known",
        "_speaker_name_if_known",
        "_active_memory",
        "_conversation_perspective",
    ):
        setattr(a, name, getattr(Agent, name).__get__(a))
    ip = InformationProcessing(a)
    a._info_processing = ip
    ip._req.trigger_kind = "発話"
    return a, ip


def _device(ip, speaker="ママ"):
    ip._req.device = TAB
    ip._req.speaker = ip._resolve_device_speaker(speaker)


def test_the_device_speaker_is_the_speaker_of_its_request():
    a, ip = _agent()
    _device(ip)
    assert a.speaker_known() is True
    assert a._speaker_name_if_known() == "ママ"
    assert a._active_memory() == "mem:id:ママ"
    assert ip._current_speaker_name() == "ママ"
    assert ip._addressee() == ("id:ママ", "ママ")
    assert ip._speaker_id() == "id:ママ"


def test_a_home_request_still_reads_the_home_speaker():
    """**反証の側**：家の求めは、いままでどおり家の話者（顔ぶれと /speaker）を読む。"""
    a, ip = _agent()
    ip._req.device = ""
    ip._req.speaker = ("id:ママ", "ママ")  # 家の求めでは使わない
    assert a._speaker_name_if_known() == "パパ"
    assert a._active_memory() == "mem:id:パパ"
    assert ip._current_speaker_name() == "パパ"
    assert ip._speaker_id() == "id:パパ"


def test_the_device_speaker_does_not_expire_during_its_request():
    a, ip = _agent()
    a._speaker_set_at = time.time() - 600  # 家の話者の寿命は切れている
    a._pmm.get_present_ids = MagicMock(return_value=[])
    assert a.speaker_known() is False
    _device(ip)
    assert a.speaker_known() is True


def test_an_unknown_or_missing_device_speaker_is_unknown_not_the_home_one():
    a, ip = _agent()
    for name in ("", "となりの人"):
        _device(ip, speaker=name)
        assert a.speaker_known() is False
        assert a._speaker_name_if_known() == ""
        assert a._active_memory() == "mem:self"
        assert ip._current_speaker_name() == ""
        assert ip._addressee() == (None, "")


def test_the_device_conversation_is_written_by_its_speaker_without_home_people():
    a, ip = _agent()
    _device(ip)
    view = a._conversation_perspective()
    assert view["writer_id"] == "id:ママ"
    assert view["participants"] == []  # 家のそばに居る人は、端末の会話を聞いていない（仮）


def test_the_device_speakers_tone_follows_the_device_speaker():
    a, ip = _agent()
    _device(ip, speaker="むすめ")
    assert ip._speaker_adult() is False
    _device(ip, speaker="ママ")
    assert ip._speaker_adult() is True


def test_the_present_context_names_the_device_speaker_as_away_from_home():
    a, ip = _agent()
    _device(ip)
    ctx = _present_ctx(a)
    assert '"ママ"' in ctx and "端末" in ctx
    assert "パパ" not in ctx
    _device(ip, speaker="")
    ctx = _present_ctx(a)
    assert "ママ" not in ctx and "パパ" not in ctx and "端末" in ctx


def test_a_device_word_denying_a_person_does_not_change_home_presence():
    a, ip = _agent()
    _device(ip)
    a._pmm.person_left = AsyncMock()
    asyncio.run(ip._apply_not_person("むすめ"))
    a._pmm.person_left.assert_not_called()
    a._persons.reset_to_default.assert_not_called()


def test_the_speaker_from_the_device_rides_on_its_request():
    """押した話者は、言葉と一緒に届き、その求めに載る（家の話者には書かない）。"""
    a, ip = _agent()
    ip._open_request = AsyncMock()
    ip._write_origin = AsyncMock(return_value="obs-1")
    ip._note_origin = MagicMock()
    ip._confirm_frame = MagicMock(return_value="")
    ip._paju().window_owner_at = MagicMock(return_value=TAB)

    def begin(device, speaker, kind="発話"):
        t = Trigger(kind="会話入力", query="明日の天気は？", device=device, speaker=speaker)
        ip._pending_device, ip._pending_speaker = t.device, t.speaker
        asyncio.run(ip._begin_request(kind=kind, text=t.query, utterance=t.query))
        return ip._req.speaker

    assert begin(TAB, "ママ") == ("id:ママ", "ママ")
    assert ip._pending_speaker == ""
    assert begin(TAB, "となりの人") == (None, "")
    assert begin("", "ママ") is None  # 家の言葉は端末の話者を持たない
    assert begin(TAB, "ママ", kind="情動") is None


def test_a_device_request_leaves_the_home_speaker_state_alone():
    """**反証の側**：端末の話者を家の顔ぶれや明示の話者に書かない（b3）。"""
    a, ip = _agent()
    a._pmm.set_speaker = AsyncMock()
    a._pmm.person_arrived = AsyncMock()
    set_at = a._speaker_set_at
    _device(ip)
    a.speaker_known()
    ip._current_speaker_name()
    a._conversation_perspective()
    a._pmm.set_speaker.assert_not_called()
    a._pmm.person_arrived.assert_not_called()
    a._persons.set_active.assert_not_called()
    assert a._speaker_set_at == set_at and a._pmm.current_speaker_id == "id:パパ"


def test_recall_for_a_device_request_searches_the_device_speaker_not_home_people():
    from familiar_agent.tools.memory import MemoryTool

    mem = {pid: MagicMock() for pid in ("self", "id:パパ", "id:ママ")}
    for pid, m in mem.items():
        m.recall_async = AsyncMock(return_value=[])
    pmm = MagicMock()
    pmm.get_agent_memory = MagicMock(return_value=mem["self"])
    pmm.get_all_present_memories = MagicMock(return_value=[("id:パパ", mem["id:パパ"])])
    pmm.get_memory_for = MagicMock(side_effect=lambda pid: mem[pid])
    tool = MemoryTool(pmm)
    asyncio.run(tool.call("recall", {"query": "傘"}, people=["id:ママ"]))
    mem["id:ママ"].recall_async.assert_awaited_once()
    mem["id:パパ"].recall_async.assert_not_called()
    asyncio.run(tool.call("recall", {"query": "傘"}))  # 家の求め（いままでどおり顔ぶれ）
    mem["id:パパ"].recall_async.assert_awaited_once()


def test_the_lookup_fixes_whose_memories_to_search_when_it_is_thrown():
    """記憶を引く人は投げたときの求めで決める（裏で走るあいだに次の求めが始まっても取り違えない）。"""
    a, ip = _agent()
    a._memory_tool = MagicMock()
    a._memory_tool.call = AsyncMock(return_value=("なし", None))
    _device(ip)

    async def go():
        ip._dispatch_lookup("recall", {"query": "傘"}, "傘", None)
        ip._req.device, ip._req.speaker = "", None  # 走るまえに家の求めへ替わった
        await asyncio.gather(*list(ip._background_tasks))

    asyncio.run(go())
    assert a._memory_tool.call.await_args.kwargs["people"] == ["id:ママ"]
    assert ip._recall_people() is None  # 家の求めは顔ぶれ
    _device(ip, speaker="")
    assert (
        ip._recall_people() == []
    )  # 誰か分からない端末の話者は、誰の記憶も引かない（自分の記憶だけ）


def test_after_the_device_request_closes_home_words_are_not_the_device_speakers():
    a, ip = _agent()
    _device(ip)
    ip._req.request_id = "obs-1"
    ip._to_state = AsyncMock()
    ip._close_final = AsyncMock(return_value=None)
    ip._close_exchange = MagicMock(return_value=[])
    ip._background_tasks.add(MagicMock())
    asyncio.run(ip._abort_lookups())
    assert ip._req.request_id is None
    assert ip._req.device == TAB  # 口は次の求めまで残す（1-ろ-2）
    assert ip._device_speaker() is None
    assert ip._current_speaker_name() == "パパ"
    assert a._conversation_perspective()["writer_id"] == "id:パパ"


def test_a_request_that_ends_without_closing_keeps_its_device_speaker():
    """閉じる所に届かずに終わっても求めは開いたまま（続きの反復がある）なので、端末の話者は外さない。"""
    a, ip = _agent()
    _device(ip)
    ip._req.request_id = "obs-1"
    ip._to_state = AsyncMock()
    asyncio.run(ip._hear_again())
    assert ip._current_speaker_name() == "ママ"


def test_personal_tools_do_not_open_for_a_device_speaker():
    """端末で押した話者は確かめていないので、本人の道具（vault・journal）は開けない（仮）。"""
    a, ip = _agent()
    seen = []
    import familiar_agent.loop.event_loop as ev

    real = ev.gate_personal_tools

    def spy(defs, *, speaker, members):
        seen.append(speaker)
        return real(defs, speaker=speaker, members=members)

    ev.gate_personal_tools = spy
    try:
        _device(ip)
        ip._gated([])
        ip._req.device = ""
        ip._gated([])
    finally:
        ev.gate_personal_tools = real
    assert seen == ["", "パパ"]


def test_home_words_heard_while_silent_are_the_home_speakers_during_a_device_request():
    a, ip = _agent()
    _device(ip)
    a._oif = MagicMock()
    a._oif.write = AsyncMock(return_value="obs-9")
    asyncio.run(ip._note_muted(Trigger(kind="会話入力", query="おなかすいた"), 1.0))
    assert ip._muted[-1].who == "パパ"
    assert a._oif.write.await_args.kwargs["writer_id"] == "id:パパ"


def test_the_device_speaker_is_called_by_the_same_name_as_at_home():
    a, ip = _agent()
    a._pmm.find_person_id_by_name = MagicMock(return_value="id:ママ")
    a._pmm.get_person_name = MagicMock(return_value="ママ")
    assert ip._resolve_device_speaker("べつのよびな") == ("id:ママ", "ママ")


def test_the_summary_is_written_by_the_speaker_of_the_closed_request():
    """会話の要約は裏で書くので、閉じる前に決めた書き手を使う（裏のあいだに次の求めが始まっても）。"""
    from familiar_agent.agent import EmbodiedAgent

    a, ip = _agent()
    a._oif = MagicMock()
    a._oif.write = AsyncMock(return_value=1)
    a._summarize_exchange = AsyncMock(return_value="要約")
    a._emotion_for_turn = AsyncMock(return_value=(None, 0.0, "中立"))
    a._maybe_discharge_satisfied_drives = AsyncMock()
    pipeline = EmbodiedAgent._run_post_response_pipeline.__get__(a)
    view = {"writer_id": "id:ママ", "participants": []}
    with __import__("contextlib").suppress(Exception):
        asyncio.run(
            pipeline(user_input="傘は？", final_text="要るよ", camera_used=False, perspective=view)
        )
    kwargs = a._oif.write.await_args.kwargs
    assert kwargs["writer_id"] == "id:ママ" and kwargs["participants"] == []


def test_the_home_speakers_lifetime_ignores_the_device_request():
    """家の話者の寿命を切る見回りは、端末の求めのあいだも家の印だけで決める（端末の話者で切りも延ばしもしない）。"""
    from familiar_agent.loop.tonic import Tonic

    a, ip = _agent()
    t = Tonic.__new__(Tonic)
    t._agent = a
    _device(ip, speaker="")  # 端末の話者は分からない
    t._expire_speaker()
    a._pmm.clear_speaker.assert_not_called()  # 家の話者はまだ生きている
    a._speaker_set_at = time.time() - 600
    a._pmm.get_present_ids = MagicMock(return_value=[])
    _device(ip)  # 端末の話者は分かっている
    t._expire_speaker()
    a._pmm.clear_speaker.assert_called_once()  # 家の話者は切れた


def test_home_words_heard_while_silent_keep_the_home_speaker_when_the_device_speaker_is_unknown():
    """家の側の読み口は、家の話者の寿命だけを見る（端末の話者が分からなくても、家の話者は生きている）。"""
    a, ip = _agent()
    a._pmm.presence_status = MagicMock(
        return_value=[]
    )  # 顔では分かっていない（/speaker の明示だけ）
    _device(ip, speaker="")
    a._oif = MagicMock()
    a._oif.write = AsyncMock(return_value="obs-9")
    asyncio.run(ip._note_muted(Trigger(kind="会話入力", query="おなかすいた"), 1.0))
    assert ip._muted[-1].who == "パパ"
    assert a._oif.write.await_args.kwargs["writer_id"] == "id:パパ"
