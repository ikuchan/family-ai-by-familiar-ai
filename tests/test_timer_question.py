"""タイマー中の「パジュ、あと何分？」に答える（ウの 3・2026-11-13 と、その改め 2026-11-14・本人の決定・`設計方針_タイマー` §9）。

声の門（書き起こしの側）をなくし、耳の入口をループの入口の 1 か所にした。入口は耳の状態（音楽の言葉だけ通す／タイマーの
言葉だけ通す）を 1 回だけ読み、要るときだけ Jev に 1 回で聞く（宛先とタイマーへの問い）。長さでは切らない。答えの中身は
調停と主LLM が `[タイマー]` の枠を見て決める。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges

# --- 1. 入口の判定は 1 回で聞く（ウの 3 の改め・2026-11-14・本人「1回で複数の質問をすれば済むはず」） ---


def _both(robot: "float | None", timer: "str | None", tconf: float = 0.9, *, ok: bool = True):
    answers: dict = {}
    if robot is not None:
        rest = (1.0 - robot) / 3
        probs = {"robot": robot, "family": rest, "nowhere": rest, "unknown": rest}
        top = max(probs, key=lambda k: probs[k])
        answers["to_whom"] = {"choice": top, "confidence": probs[top], "probabilities": probs}
    if timer is not None:
        answers["timer"] = {"choice": timer, "confidence": tconf}
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return c


def _ask_entrance(client, *, addressee: bool, timer: bool, **kw):
    args = dict(
        text="あと何分？",
        recent="",
        present="パパ",
        names=["パジュ"],
        voice="",
        addressee=addressee,
        timer=timer,
        min_conf=0.6,
    )
    args.update(kw)
    return asyncio.run(jev_judges.judge_entrance(client, **args))


def test_the_entrance_asks_both_questions_in_one_call():
    c = _both(0.9, jev_judges.TIMER_QUESTION)
    got = _ask_entrance(c, addressee=True, timer=True)
    assert got.to_paju == jev_judges.TO_PAJU and got.timer_question is True
    assert c.ask.await_count == 1
    state, questions = c.ask.await_args.args
    assert set(questions) == {"to_whom", "timer"}
    # 宛先の状態に「タイマーが動いている」の 1 行を足す（本人の決定・案 A）
    assert "[ロボットの呼び名]\nパジュ" in state and "タイマーが動いている" in state


def test_only_the_needed_questions_are_asked():
    c = _both(None, jev_judges.TIMER_QUESTION)
    got = _ask_entrance(c, addressee=False, timer=True)
    assert set(c.ask.await_args.args[1]) == {"timer"}
    assert got.to_paju is None and got.timer_question is True
    c = _both(0.1, None)
    got = _ask_entrance(c, addressee=True, timer=False)
    state, questions = c.ask.await_args.args
    assert set(questions) == {"to_whom"} and "タイマーが動いている" not in state
    assert got.to_paju == jev_judges.NOT_TO_PAJU and got.timer_question is None


def test_no_answer_drops_the_timer_question_and_follows_the_addressee_rule():
    """答えなし：タイマーへの問いは「いいえ」（捨てる側）。宛先は `drop_unanswered` に従う（いままでどおり）。"""
    c = _both(None, None, ok=False)
    got = _ask_entrance(c, addressee=True, timer=True)
    assert got.timer_question is False and got.to_paju == jev_judges.TO_PAJU
    got = _ask_entrance(c, addressee=True, timer=True, drop_unanswered=True)
    assert got.to_paju == jev_judges.NOT_TO_PAJU
    assert _ask_entrance(None, addressee=False, timer=True).timer_question is False
    low = _both(None, jev_judges.TIMER_QUESTION, tconf=0.59)
    assert _ask_entrance(low, addressee=False, timer=True).timer_question is False


def test_nothing_is_asked_when_no_question_is_needed():
    c = _both(0.9, jev_judges.TIMER_QUESTION)
    got = _ask_entrance(c, addressee=False, timer=False)
    assert c.ask.await_count == 0 and got.to_paju is None and got.timer_question is None


# --- 2. 耳の入口（声の門をなくし、ループの入口の 1 か所で見る・ウの 3 の改め・2026-11-14・本人の決定 案 2） ---

_PLAYLISTS = (("朝の曲", "spotify:playlist:aaa", False),)


def _ear(reason: str, *, window: bool = False):
    from tests.test_silence_hold import _ip

    ip, a = _ip(speaker="", silenced_for=None)
    a.mic_gate_reason = MagicMock(return_value=reason)
    a._music_state.playing = reason == "音楽が鳴っている"
    a._music_table = MagicMock(return_value=_PLAYLISTS)
    ip._voice_stranger = MagicMock(return_value=False)
    ip._nudge_by_voice = AsyncMock()
    if window:
        asyncio.run(ip._paju().open_window(time.monotonic()))
    return ip, a


def _voice(ip, query, source="voice"):
    from familiar_agent.loop.event_loop import Trigger
    from tests.test_silence_hold import _run

    t = Trigger(kind="会話入力", query=query, source=source, arrived=time.monotonic())
    return _run(ip._swallow_if_unheard(t)), t


def test_music_drops_unnamed_non_music_words_first_and_quietly():
    """歌詞やテレビの声は、知らない声の判定・情動の押し上げ・記録より前に、黙って捨てる（いまの声の門と同じ）。"""
    ip, a = _ear("音楽が鳴っている", window=True)  # 窓の中でも捨てる
    swallowed, t = _voice(ip, "あしたの空へ飛んでいこう")
    assert swallowed is True and t.ear_dropped is True
    assert _voice(ip, "3 分測って、パジュ")[1].ear_dropped is True  # 名前は文頭だけ
    ip._voice_stranger.assert_not_called()
    ip._nudge_by_voice.assert_not_awaited()
    a._oif.write.assert_not_awaited()


def test_music_lets_named_and_music_words_through_the_ear():
    ip, _ = _ear("音楽が鳴っている")
    _, t = _voice(ip, "音楽止めてくれないかなって思ってるんだけど")  # 長さでは切らない
    assert t.ear_dropped is False
    _, t = _voice(ip, "朝の曲かけて")
    assert t.ear_dropped is False
    _, t = _voice(ip, "パジュ、天気は？")
    assert t.ear_dropped is False


def test_the_timer_drops_unnamed_voices_outside_the_window_quietly():
    ip, a = _ear("タイマー「パスタ」")
    swallowed, t = _voice(ip, "きょうのニュースです")
    assert swallowed is True and t.ear_dropped is True
    ip._nudge_by_voice.assert_not_awaited()


def test_the_timer_lets_voices_in_the_window_and_named_voices_reach_the_rest():
    """**反証の側**：窓の中の名前の無い声と、名前つきの声は、耳では捨てない（その先の決まりが決める）。"""
    ip, _ = _ear("タイマー「パスタ」", window=True)
    ip._between_family = AsyncMock(return_value=True)
    _, t = _voice(ip, "止めて")
    assert t.ear_dropped is False
    ip, _ = _ear("タイマー「パスタ」")
    _, t = _voice(ip, "パジュ、止めて")
    assert t.ear_dropped is False


def test_the_ear_does_not_touch_the_keyboard():
    ip, _ = _ear("音楽が鳴っている")
    _, t = _voice(ip, "あしたの空へ飛んでいこう", source="keyboard")
    assert t.ear_dropped is False


def test_a_quiet_drop_is_not_shown_nor_measured():
    """耳で捨てた声は画面に知らせず（本人の決定）、捨てた声の点数も測らない。"""
    from tests.test_event_loop import _agent

    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._notify_heard = MagicMock()
    ip._log_swallowed_voice = MagicMock()
    ip._ensure_driver = MagicMock()

    async def ear_drop(trigger):
        trigger.ear_dropped = True
        return ip._swallowed(trigger)

    ip._swallow_if_unheard = ear_drop
    assert asyncio.run(ip.push_utterance("あしたの空へ", source="voice")) == ""
    ip._notify_heard.assert_not_called()
    ip._log_swallowed_voice.assert_not_called()


# --- 3. タイマーの入口（耳が閉じている／口が黙っている・Jev は 1 回・ウの 3 の改め・2026-11-14・本人の決定） ---


def _gate(
    *,
    robot: "float | None" = 0.9,
    timer: "str | None" = None,
    ear="タイマー「パスタ」",
    reason="timer:1",
    window=False,
):
    """既定の設定（`TIMER_MIC_CLOSE`・`TIMER_SILENCE` が真）：耳はタイマーで閉じ、口はタイマーで黙っている。"""
    from familiar_agent.core.silence_hold import Heard
    from familiar_agent.silence_state import SilenceRequest
    from tests.conftest import silence

    ip, a = _ear(ear, window=window)
    if reason is not None:
        silence(ip, SilenceRequest(person="パパ", until=time.time() + 120, reason=reason))
    a.config.jev_confidence_min = 0.6
    a._jev = _both(robot, timer)
    ip._muted.append(Heard(kind="会話入力", text="こんにちは", at=time.time()))
    return ip, a


def _asked(a) -> "set[str]":
    assert a._jev.ask.await_count == 1, a._jev.ask.await_count  # 調停より前の Jev は 1 回だけ
    return set(a._jev.ask.await_args.args[1])


def test_a_named_question_passes_with_one_jev_call_and_keeps_what_was_heard():
    ip, a = _gate(robot=None, timer=jev_judges.TIMER_QUESTION)
    swallowed, t = _voice(ip, "パジュ、あと何分？")
    assert swallowed is False
    assert _asked(a) == {"timer"}  # 名前がそのまま当たれば宛先は聞かない
    assert [h.text for h in ip._muted] == ["こんにちは"]  # 溜めたものは渡さない
    assert ip._paju().window_open_at(time.monotonic())  # 返事を声にするため窓を開ける


def test_a_named_non_question_is_dropped_without_a_record_while_the_ear_is_closed():
    ip, a = _gate(robot=None, timer=jev_judges.NOT_TIMER_QUESTION)
    swallowed, t = _voice(ip, "パジュ、電気消して")
    assert swallowed is True and t.ear_dropped is True
    assert [h.text for h in ip._muted] == ["こんにちは"]  # 控えない
    a._oif.write.assert_not_awaited()  # 記録しない


def test_a_named_non_question_is_kept_while_only_the_mouth_is_closed():
    """**反証の側**：耳が開いている（`TIMER_MIC_CLOSE` が偽）なら、問いでない声は控える（いまどおり）。"""
    ip, a = _gate(robot=None, timer=jev_judges.NOT_TIMER_QUESTION, ear="")
    swallowed, t = _voice(ip, "パジュ、電気消して")
    assert swallowed is True and t.ear_dropped is False
    assert [h.text for h in ip._muted] == ["こんにちは", "パジュ、電気消して"]


def test_the_name_alone_opens_the_window_quietly_even_while_silent():
    """「パジュ」と呼んで間をおいて問う（10 秒・窓と同じ）。名前だけの声は控えず、黙って窓を開ける。"""
    ip, a = _gate(timer=jev_judges.TIMER_QUESTION)
    swallowed, t = _voice(ip, "パジュ")
    assert swallowed is True and t.quiet_open is True
    assert a._jev.ask.await_count == 0
    assert [h.text for h in ip._muted] == ["こんにちは"]
    swallowed, t = _voice(ip, "あと何分？")  # 窓の中の名前の無い声
    assert swallowed is False
    assert _asked(a) == {"to_whom", "timer"}  # 宛先も同じ 1 回で
    state = a._jev.ask.await_args.args[0]
    assert "タイマーが動いている" in state


def test_a_voice_in_the_window_not_for_paju_is_dropped():
    ip, a = _gate(robot=0.1, timer=jev_judges.TIMER_QUESTION, window=True)
    swallowed, t = _voice(ip, "あと何分？")
    assert swallowed is True and t.ear_dropped is True
    assert _asked(a) == {"to_whom", "timer"}


def test_the_rest_of_a_tentative_call_is_asked_once_with_the_name_put_in_front():
    ip, a = _gate(timer=jev_judges.TIMER_QUESTION)
    asyncio.run(ip._paju().call_tentatively(time.monotonic()))
    swallowed, _ = _voice(ip, "あと何分？")
    assert swallowed is False
    assert _asked(a) == {"to_whom", "timer"}
    assert "パジュ、あと何分？" in a._jev.ask.await_args.args[0]


def test_a_near_name_is_asked_once_for_both():
    """窓が閉じているときの 1 字違いの名前：宛先とタイマーへの問いを同じ 1 回で聞く（2 回聞いていた）。"""
    ip, a = _gate(timer=jev_judges.TIMER_QUESTION)
    ip._wake_admits = MagicMock(side_effect=lambda t, n, near_ok=False: near_ok)
    swallowed, _ = _voice(ip, "パジャ、あと何分？")
    assert swallowed is False
    assert _asked(a) == {"to_whom", "timer"}


def test_a_named_control_word_lifts_the_timer_silence_without_jev():
    ip, a = _gate()
    swallowed, _ = _voice(ip, "パジュ、止めて")
    assert swallowed is False
    assert a._jev.ask.await_count == 0
    assert ip._pending_heard and ip._muted == []  # 解く言葉の求めがまとめの求めになる


def test_a_named_control_word_with_only_the_ear_closed_takes_the_usual_way():
    ip, a = _gate(reason=None)
    swallowed, _ = _voice(ip, "パジュ、止めて")
    assert swallowed is False and a._jev.ask.await_count == 0


def test_a_human_silence_never_asks_about_the_timer():
    ip, a = _gate(timer=jev_judges.TIMER_QUESTION, ear="", reason="")
    swallowed, _ = _voice(ip, "パジュ、あと何分？")
    assert swallowed is True and a._jev.ask.await_count == 0
    assert [h.text for h in ip._muted] == ["こんにちは", "パジュ、あと何分？"]


def test_no_answer_drops_while_the_ear_is_closed():
    ip, a = _gate(robot=None, timer=None)
    a._jev.ask = AsyncMock(return_value=JevAnswer(ok=False, error="時間切れ"))
    swallowed, t = _voice(ip, "パジュ、あと何分？")
    assert swallowed is True and t.ear_dropped is True


# --- 4. 軽く返すときも `[タイマー]` の枠を見る ---

_TIMER_FRAME = "[タイマー]\n- id=1 パスタ 残り 3:20（12:05 に鳴る）"


def _written(timer_ctx: str) -> "list[str]":
    from tests._arbiter_fakes import decide, jev_says, writer_says

    writer = writer_says({"text": "あと 3 分 20 秒だよ"})
    asyncio.run(
        decide(
            jev=jev_says("light"),
            writer=writer,
            utterance="パジュ、あと何分？",
            named=True,
            timer_ctx=timer_ctx,
        )
    )
    return [c.args[0] for c in writer.complete.await_args_list]


def test_a_light_reply_sees_the_timer_frame():
    """軽量LLM は残り時間を知らずに答えていた（枠は主LLM にしか載らなかった）。"""
    assert any(_TIMER_FRAME in p for p in _written(_TIMER_FRAME))


def test_no_timer_frame_when_no_timer_runs():
    assert not any("[タイマー]" in p for p in _written(""))


def test_the_loop_hands_the_timer_frame_to_the_arbiter(monkeypatch):
    from familiar_agent.loop import event_loop
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    monkeypatch.setattr(event_loop, "_timer_frame", lambda agent: _TIMER_FRAME)
    assert (
        ip._arbiter_input(utterance="x", workspace_ctx="", present_ctx="").timer_ctx == _TIMER_FRAME
    )


# --- 5. お知らせの文言 ---


def test_the_frame_says_a_named_question_also_reaches(monkeypatch):
    """「止めて・一時停止・再開の言葉だけ届く」は、名前が要ることも時間を聞けることも書いていなかった。"""
    from tests.test_timer_tool import _tool

    monkeypatch.setenv("TIMER_CONFIRM", "false")
    monkeypatch.setenv("TIMER_MIC_CLOSE", "true")

    t, _, _ = _tool()
    asyncio.run(t.call("set_timer", {"after_minutes": 3, "label": "パスタ"}))
    frame = t.frame()
    assert "名前を呼んだときの、止めて・一時停止・再開の言葉と、タイマーへの問いだけ届く" in frame


# --- 6. 見直しの指摘（2026-11-14・high 1 回目） ---


def test_a_human_silence_is_kept_while_the_ear_is_closed_by_the_timer():
    """人に頼まれて黙っているときは、いまのまま（本人の決定）。耳がタイマーで閉じていても、タイマーの決まりで通さない（指摘 1）。"""
    ip, a = _gate(timer=jev_judges.TIMER_QUESTION, reason="")
    for said in ("パジュ、あと何分？", "パジュ、止めて", "パジュ"):
        swallowed, t = _voice(ip, said)
        assert swallowed is True and t.ear_dropped is False, (
            said
        )  # 耳で捨てず、黙っている枝で控える
    assert [h.text for h in ip._muted] == [
        "こんにちは",
        "パジュ、あと何分？",
        "パジュ、止めて",
        "パジュ",
    ]
    assert a._jev.ask.await_count == 0
    assert not ip._paju().window_open_at(time.monotonic())


@pytest.mark.real_window  # 窓の外の決まりは本物の判定で見る
def test_an_unnamed_control_word_outside_the_window_takes_the_usual_way():
    """名前の無い操作の言葉は、いままでどおり窓の外の決まりで捨てる（情動を押し上げ、状態の行に出す・指摘 6）。"""
    ip, _ = _ear("タイマー「パスタ」")
    swallowed, t = _voice(ip, "ストップ")
    assert swallowed is True and t.ear_dropped is False
    ip._nudge_by_voice.assert_awaited_once()
