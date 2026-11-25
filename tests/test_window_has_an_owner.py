"""端-b 段 1-ろ：窓は持ち主（家か、どの端末か）を 1 つ持ち、別の口の言葉は断る（課題9 §7.4）。

- 家で名前を呼ばれて開いた窓は家のもの、端末から話しかけて開いた窓はその端末のもの。
- 窓が開いていて持ち主が別の口なら、その口の言葉は断る（窓を開け直さない・延ばさない・持ち主を変えない）。
- 端末の言葉は、端末の中で名前の合図を聞いてあるので、名前で呼ばれた言葉として受ける。家で「黙って」と頼まれて
  いても受け（b4）、黙るのは解かず、黙っていたあいだに控えた家の言葉も渡さない。
- 口の違う言葉は、いまの求めに添えず、確認の答えにも使わない（§7.4 の帰結）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core.wake_window import WakeWindow
from familiar_agent.loop.event_loop import InformationProcessing, Trigger

pytestmark = pytest.mark.real_window

TAB = "tablet-1"


def test_another_mouth_cannot_take_an_open_window():
    w = WakeWindow()
    w.open(100.0, owner=TAB)
    until, session = w.until, w.session
    w.open(105.0)  # 家の名前（持ち主が違う）
    assert w.owner == TAB and w.until == until and w.session == session
    w.extend(105.0)  # 家の窓の中の言葉としては延ばさない
    assert w.until == until
    w.extend(105.0, owner=TAB)  # 持ち主の言葉なら延ばす
    assert w.until > until


def test_a_closed_window_has_no_owner_and_the_next_opener_owns_it():
    w = WakeWindow()
    w.open(100.0, owner=TAB)
    assert w.owner_at(101.0) == TAB
    assert w.owner_at(200.0) is None  # 閉じた窓は誰のものでもない
    w.open(200.0)
    assert w.owner_at(201.0) == "" and w.session == 2  # 家が開けた新しい窓の会話


def _ip(*, silenced: bool = False):
    a = MagicMock()
    a.config.agent_names = ["パジュ"]
    a._pmm.presence_status = MagicMock(return_value=[])
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    a._occupancy = MagicMock(return_value=1.0)
    a._someone_here = MagicMock(return_value=True)
    a._nudge_drive = AsyncMock()
    a._timer_tool.frame = MagicMock(return_value="")
    a._dif.ringing = False
    ip = InformationProcessing(a)
    if silenced:
        from familiar_agent.silence_state import SilenceRequest

        ip._paju().silence = SilenceRequest(person="ママ", until=time.time() + 600)
    return ip


def _push(ip, text, *, source="voice", device=""):
    async def go():
        fut = asyncio.get_running_loop().create_future()
        t = Trigger(kind="会話入力", query=text, future=fut, source=source, device=device)
        swallowed = await ip._swallow_if_unheard(t)
        return not swallowed, t

    return asyncio.run(asyncio.wait_for(go(), timeout=2.0))


def test_a_device_word_opens_a_window_owned_by_that_device():
    ip = _ip()
    heard, t = _push(ip, "明日の天気は？", source="device", device=TAB)  # 名前は端末が聞いた
    assert heard and t.named
    assert ip._paju().wake.owner_at(time.monotonic()) == TAB
    assert t.window_session == ip._paju().wake.session


def test_home_words_are_refused_while_a_device_owns_the_window():
    ip = _ip()
    _push(ip, "明日の天気は？", source="device", device=TAB)
    assert _push(ip, "パジュ、おはよう")[0] is False  # 名前つきの家の声も断る
    assert _push(ip, "パジュ、おはよう", source="keyboard")[0] is False
    assert ip._paju().wake.owner_at(time.monotonic()) == TAB  # 持ち主は変わらない


def test_device_words_are_refused_while_home_owns_the_window():
    ip = _ip()
    assert _push(ip, "パジュ、おはよう")[0] is True
    assert _push(ip, "明日の天気は？", source="device", device=TAB)[0] is False
    assert _push(ip, "明日の天気は？", source="device", device="phone-2")[0] is False
    assert ip._paju().wake.owner_at(time.monotonic()) == ""


def test_the_owner_device_keeps_talking_without_the_name():
    ip = _ip()
    _push(ip, "明日の天気は？", source="device", device=TAB)
    assert _push(ip, "じゃあ傘は？", source="device", device=TAB)[0] is True


def test_a_device_is_answered_while_home_is_silenced_and_silence_stays():
    """**反証の側**：家の「黙って」は解かず、控えた家の言葉も端末の求めへ渡さない（b4）。"""
    ip = _ip(silenced=True)
    ip._muted = ["家で控えた言葉"]
    heard, _ = _push(ip, "明日の天気は？", source="device", device=TAB)
    assert heard
    assert ip._paju().silence is not None  # 黙っているまま
    assert ip._muted == ["家で控えた言葉"] and not getattr(ip, "_pending_heard", None)


def test_a_word_from_another_mouth_does_not_join_the_open_request():
    """口の違う言葉は、いまの求めの言い足しにも確認の答えにもしない。"""
    ip = _ip()
    ip._req.trigger_kind = "発話"
    ip._req.device = ""
    home = Trigger(kind="会話入力", query="あと", device="")
    tab = Trigger(kind="会話入力", query="はい", device=TAB)
    assert ip._same_mouth(home) and not ip._same_mouth(tab)
    ip._held = [home, tab]
    assert ip._waiting_words() == [home]  # 考え直しに見せるのも同じ口の言葉だけ


def test_a_device_request_does_not_read_a_home_confirmation():
    """家で聞いた確認（タイマーを掛ける前など）を、端末の言葉の答えに読まない。確認待ちそのものは生きたまま。"""
    ip = _ip()
    ip._paju().confirm = MagicMock(return_value=MagicMock(text="掛けていい？"))
    ip._paju().confirm_prompted_at = MagicMock(return_value=0.0)
    ip._req.device = TAB
    assert ip._confirming()  # 情動を止めるなど、家の決まりは確認待ちを見たまま
    assert ip._confirm_inputs() == {}
    ip._req.device = ""
    assert ip._confirm_inputs()


def test_a_device_request_neither_sees_nor_changes_home_silence():
    """**反証の側**：端末の「話していいよ」で家の「黙って」を解かず、調停にも黙っていると伝えない（b4）。"""
    ip = _ip(silenced=True)
    ip._req.device = TAB
    assert ip._silence_note() == ""
    decision = MagicMock(silence_minutes=0, lift_silence=True)
    assert asyncio.run(ip._apply_silence(decision, utterance="話していいよ")) is None
    assert ip._paju().silence is not None
    ip._req.device = ""
    assert ip._silence_note() != ""


def test_a_waited_word_is_refused_if_another_mouth_now_owns_the_window():
    ip = _ip()
    _push(ip, "明日の天気は？", source="device", device=TAB)
    home = Trigger(kind="会話入力", query="あと", device="")
    home.future = None
    assert ip._owned_by_another(home)
    tab = Trigger(kind="会話入力", query="傘は？", device=TAB)
    assert not ip._owned_by_another(tab)


def test_a_device_source_without_a_device_name_is_not_taken():
    ip = _ip()
    got = asyncio.run(ip.push_utterance("明日の天気は？", source="device", device=""))
    assert got == ""
    assert ip._paju().wake.owner_at(time.monotonic()) is None


def test_the_resumed_window_of_another_mouth_is_a_new_conversation():
    w = WakeWindow()
    w.open(100.0, owner=TAB)
    w.open(200.0, resume=True)  # 家の返事の話し終わり（端末の窓は閉じている）
    assert w.owner == "" and w.session == 2


def test_only_the_owners_request_holds_the_window():
    """**反証の側**：端末の窓のあいだに家の求めが考えていても、端末の窓を保たない（家を長く締め出さない）。"""
    from familiar_agent.loop.paju_state import PajuState

    sm = PajuState()
    sm.head_state = MagicMock(return_value="thinking")
    sm._in = MagicMock(return_value=False)
    sm.wake.open(time.monotonic(), owner=TAB)
    sm.request_mouth = ""
    assert not sm._request_holds()
    sm.request_mouth = TAB
    assert sm._request_holds()


def test_a_word_begun_before_the_window_was_taken_had_no_owner():
    w = WakeWindow()
    w.open(101.0, owner=TAB)
    assert w.owner_at(100.0) is None  # 端末が開ける前に話し始めた家の声
    assert w.owner_at(102.0) == TAB


def test_a_fresh_word_is_refused_at_take_time_if_another_mouth_took_the_window():
    """門を通ったあと、列で待つあいだに別の口が窓を取ったら断る（門の中の Jev 待ちのあいだも同じ）。"""
    ip = _ip()
    home = Trigger(kind="会話入力", query="パジュ、おはよう", device="")

    async def go():
        await ip._paju().open_window(time.monotonic(), owner=TAB)
        ip._triggers.put_nowait(home)
        ip._triggers.put_nowait(Trigger(kind="機器", query="x"))
        return await ip._take_trigger()

    got = asyncio.run(asyncio.wait_for(go(), timeout=2.0))
    assert got is not home
