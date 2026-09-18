"""沈黙は入口で止め、明けたら 1 つの求めにまとめる（情-h・2026-09-16）。

黙っているあいだ、会話入力・機器・情動は O に残すだけで求めを立てない（誰の声でも同じ・
案イ撤回）。通すのは解くきっかけだけ：タイマーが鳴る・本人の「話していい」・本人の止める
頼み。明けた瞬間の 1 つの求めに、溜めたものを W の作業状態の枠へ列挙（字数上限・新しい
ものから・溢れは「ほか N 件」・機器と情動は件数）し、主LLM が 1 回で答える。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core.silence_hold import Heard, lifts, render
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.silence_state import SilenceRequest

# ── 純関数 ────────────────────────────────────────────────────────────────


def test_what_lifts_the_silence():
    assert lifts("機器", "タイマー", speaker="", asker="パパ")
    assert lifts("会話入力", "もう話していいよ", speaker="パパ", asker="パパ")
    assert lifts("会話入力", "タイマー止めて", speaker="パパ", asker="パパ")
    assert not lifts("会話入力", "話していいよ", speaker="たいきくん", asker="パパ")  # 本人でない
    assert not lifts("会話入力", "明日の予定は？", speaker="パパ", asker="パパ")
    assert not lifts("機器", "入室", speaker="", asker="パパ")
    assert not lifts("情動", "SEEKING", speaker="", asker="パパ")


def _t(h, m):
    return time.mktime(time.strptime(f"2026-09-16 {h:02d}:{m:02d}", "%Y-%m-%d %H:%M"))


def test_render_lists_talks_verbatim_and_counts_the_rest():
    items = [
        Heard("会話入力", "明日の予定は？", who="パパ", at=_t(17, 1)),
        Heard("機器", "入室（たいきくん が来た）", at=_t(17, 2)),
        Heard("情動", "SEEKING", at=_t(17, 2)),
        Heard("情動", "SEEKING", at=_t(17, 3)),
        Heard("会話入力", "ただいま", who="たいきくん", at=_t(17, 3)),
    ]
    text = render(items, since=_t(17, 0), until=_t(17, 4), max_chars=4000)
    assert text.startswith("黙っていたあいだ（17:00〜17:04）")
    assert "聞いたこと 2 件・起きたこと 1 件・湧いたこと 2 件" in text
    assert "- 17:01 パパ：「明日の予定は？」" in text
    assert "- 17:03 たいきくん：「ただいま」" in text
    assert "起きたこと：入室（たいきくん が来た）" in text and "SEEKING ×2" in text
    assert text.index("パパ") < text.index("たいきくん")  # 時系列


def test_render_keeps_the_newest_within_the_limit():
    items = [
        Heard("会話入力", f"話 {i} " + "あ" * 30, who="パパ", at=_t(17, i)) for i in range(1, 30)
    ]
    text = render(items, since=_t(17, 0), until=_t(18, 0), max_chars=600)
    assert "話 29 " in text and "話 1 " not in text
    assert "ほか" in text and "記憶にある" in text
    assert len(text) <= 700  # 見出しと「ほか」の行ぶんの余裕


def test_render_is_empty_when_nothing_was_heard():
    assert render([], since=0, until=1, max_chars=100) == ""


# ── ループの入口 ──────────────────────────────────────────────────────────


def _ip(speaker="パパ", *, silenced_for="パパ", others=()):
    a = MagicMock()
    rows = [{"name": speaker, "is_speaker": True, "confidence": 1.0}]
    rows += [{"name": n, "is_speaker": False, "confidence": 1.0} for n in others]
    a._pmm.presence_status = MagicMock(return_value=rows)
    a._oif.write = AsyncMock(return_value="obs-1")
    a._observation_perspective = MagicMock(return_value={})
    a._conversation_perspective = MagicMock(return_value={})
    ip = InformationProcessing(a)
    req = SilenceRequest(person=silenced_for, until=time.time() + 600) if silenced_for else None
    ip._load_silence = lambda: req
    return ip, a


def _run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, timeout=2.0)

    return asyncio.run(bounded())


def test_an_utterance_during_silence_is_heard_but_not_answered():
    ip, a = _ip()

    async def scenario():
        fut = asyncio.get_running_loop().create_future()
        swallowed = await ip._swallow_if_unheard(
            Trigger(kind="会話入力", query="明日の予定は？", future=fut)
        )
        return swallowed, fut

    swallowed, fut = _run(scenario())
    assert swallowed is True
    assert fut.done() and fut.result() == ""  # GUI は吹き出しだけ
    a._oif.write.assert_awaited()  # O には残る
    assert [h.kind for h in ip._muted] == ["会話入力"]


def test_someone_else_is_not_answered_either():
    """案イ（他人への返事は通す）は撤回。黙っていてと頼まれたら、誰の声でも黙る。"""
    ip, _ = _ip(speaker="たいきくん", silenced_for="パパ", others=("パパ",))  # パパは居る

    async def scenario():
        return await ip._swallow_if_unheard(Trigger(kind="会話入力", query="ねえパジュ"))

    assert _run(scenario()) is True


def test_devices_and_urges_are_noted_not_acted_on():
    ip, _ = _ip()

    async def scenario():
        d = await ip._swallow_if_unheard(
            Trigger(kind="機器", query="入室", result="たいきくん が来た")
        )
        u = await ip._swallow_if_unheard(Trigger(kind="情動", query="SEEKING", result="探索したい"))
        return d, u

    assert _run(scenario()) == (True, True)
    assert [h.kind for h in ip._muted] == ["機器", "情動"]


def test_a_ringing_timer_and_the_release_phrase_pass():
    ip, _ = _ip()

    async def scenario():
        t = await ip._swallow_if_unheard(Trigger(kind="機器", query="タイマー", result="時間"))
        r = await ip._swallow_if_unheard(Trigger(kind="会話入力", query="もう話していいよ"))
        return t, r

    assert _run(scenario()) == (False, False)


def test_nothing_is_swallowed_when_not_silenced():
    ip, _ = _ip(silenced_for=None)

    async def scenario():
        return await ip._swallow_if_unheard(Trigger(kind="会話入力", query="おはよう"))

    assert _run(scenario()) is False


def test_when_silence_ends_the_heard_things_ride_the_next_request():
    ip, a = _ip(silenced_for=None)
    ip._muted = [Heard("会話入力", "明日の予定は？", who="パパ", at=time.time())]
    ip._muted_since = time.time() - 60

    async def scenario():
        assert await ip._swallow_if_unheard(Trigger(kind="会話入力", query="話していいよ")) is False
        await ip._begin_request(kind="発話", text="話していいよ", utterance="話していいよ")
        return ip._req.heard_while_silent, ip._muted

    heard, left = _run(scenario())
    assert [h.text for h in heard] == ["明日の予定は？"] and left == []


# ── W の作業状態の枠 ───────────────────────────────────────────────────────


def test_the_heard_things_ride_the_workspace_and_leave_the_recall_column():
    """列挙は作業状態の枠（想起を経ず全部）、同じ記録は想起の 7 件から除く。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request
    from tests.test_workspace_is_the_core import _rec

    oif = MagicMock(actors=MagicMock(return_value={}), roles=MagicMock(return_value={}))
    req = Request()
    now = time.time()
    req.heard_while_silent = [
        Heard("会話入力", "明日の予定は？", who="パパ", obs_id="hhhhhhhh-1", at=now),
        Heard("情動", "SEEKING", at=now),
    ]
    memories = [
        _rec("hhhhhhhh-1", "（黙っていたあいだに聞いた）明日の予定は？"),
        _rec("m2", "昔の話"),
    ]
    text, _ = workspace.compose(oif, memories, req)
    assert "黙っていたあいだ" in text and "「明日の予定は？」" in text and "SEEKING ×1" in text
    assert text.count("明日の予定は？") == 1  # 想起の列には重ねて出ない
    assert "昔の話" in text


# ── タイマー由来の沈黙では、操作の言葉は誰の言葉でも通す（情-m・2026-09-18）────────
#
# 沈黙の「本人か」は会話の寿命（話者の指定 60 秒・知-t）で決めるが、タイマーは何分も返事をしない
# 状態を設計として作る。掛けて 60 秒後には話者が「分からない」になり、本人の「一時停止」が
# `黙っているので聞くだけ` に落ちた（コードで確認・実機 会話 4b の予想）。声の門（`TIMER_MIC_CLOSE`）
# は既に「操作の言葉だけ・誰でも」なので、入口も同じにする。人の依頼の沈黙は今までどおり本人だけ。


def test_a_timer_silence_lets_control_words_through_from_anyone():
    assert lifts("会話入力", "一時停止", speaker="", asker="パパ", reason="timer:1")
    assert lifts("会話入力", "止めて", speaker="たいきくん", asker="パパ", reason="timer:1")
    assert not lifts("会話入力", "こんにちは", speaker="", asker="パパ", reason="timer:1")
    assert not lifts(
        "会話入力",
        "テレビの中で「もう止めてくれ」と叫んでいた",
        speaker="",
        asker="パパ",
        reason="timer:1",
    )


def test_a_human_silence_still_needs_the_asker():
    assert not lifts("会話入力", "止めて", speaker="たいきくん", asker="パパ", reason="")
    assert not lifts("会話入力", "一時停止", speaker="", asker="パパ")


def test_the_entrance_passes_a_pause_during_a_timer_when_the_speaker_has_expired():
    ip, a = _ip(speaker="", silenced_for=None)
    a._pmm.presence_status = MagicMock(return_value=[])  # 話者は切れている
    ip._load_silence = lambda: SilenceRequest(
        person="パパ", until=time.time() + 120, reason="timer:1"
    )
    ip._swallowed = MagicMock(return_value=True)
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="一時停止"))) is False
    ip._swallowed.assert_not_called()
    assert _run(ip._swallow_if_unheard(Trigger(kind="会話入力", query="こんにちは"))) is True
