"""ゲームのおすすめ（環-ba・2026-11-15・本人の決定）。

親愛や承認が溜まって話しかける発火（bond・esteem）のときに、その日まだ誘っていなければ、主LLM と軽量LLM の文に
`[遊びのおすすめ]` を載せ、「しりとりしない？」と誘ってよいことにする。誘うかは書く側が決める。数え方は音楽の
おすすめと同じで、声にした文に遊びの名前が入っていた日だけ「誘った」と数える（1 日 1 回まで）。
"""

from __future__ import annotations


import pytest

from familiar_agent.core import game_suggestion as gs
from familiar_agent.core import music_suggestion as ms
from familiar_agent.core.skills import SKILLS_DIR, load_skills
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

NAMES = ["しりとり", "マジカルバナナ"]
TODAY = "2026-11-15"


def test_offers_the_games_unless_already_today():
    got = gs.frame(NAMES, today=TODAY, last="2026-11-14")
    assert got.startswith(gs.HEADING)
    assert "しりとり、マジカルバナナ" in got
    assert "1 回の声で 1 つまで" in got


def test_no_second_offer_on_the_same_day():
    """**反証の側**：その日もう誘っていたら載せない。"""
    assert gs.frame(NAMES, today=TODAY, last=TODAY) == ""


def test_no_offer_without_games():
    """**反証の側**：遊びが 1 つも無いときは載せない。"""
    assert gs.frame([], today=TODAY, last="") == ""


def test_said_counts_only_with_a_game_name():
    assert gs.offered("ねえ、しりとりしない？", NAMES)
    assert not gs.offered("今日もいい天気だね", NAMES)  # **反証の側**


def _ip(
    monkeypatch, *, axis: str, last: str = "", saves: bool = True
) -> tuple[InformationProcessing, list]:
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._agent._music_tool = None  # 音楽の道具が無くても、遊びの枠は出す
    ip._skills = tuple(
        load_skills(SKILLS_DIR)
    )  # 話しかける発火の始めに `_load_skills` で読んだもの
    ip._req.trigger_kind = "情動"
    ip._req.fired_axis = axis
    saved: list = []
    monkeypatch.setattr(ms, "today", lambda: TODAY)
    monkeypatch.setattr(gs, "last_offered_on", lambda: last)
    monkeypatch.setattr(gs, "store_offered_on", lambda day: saved.append(day) or saves)
    return ip, saved


def test_the_loop_frame_carries_the_games_without_a_music_tool(monkeypatch):
    ip, _ = _ip(monkeypatch, axis="bond")
    assert gs.HEADING in ip._suggestion_frame()


def test_the_loop_frame_skips_the_games_for_a_quiet_firing(monkeypatch):
    """**反証の側**：seeking の発火では載せない。"""
    ip, _ = _ip(monkeypatch, axis="seeking")
    assert gs.HEADING not in ip._suggestion_frame()


def test_a_talking_firing_loads_the_skills_first(monkeypatch):
    """枠と数える口がループの上で skills/ を読まないよう、発火の始めにループの外で読んでおく。"""
    import asyncio
    from unittest.mock import AsyncMock

    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._write_origin = AsyncMock(return_value=1)  # type: ignore[method-assign]
    ip._iterate = AsyncMock(return_value="")  # type: ignore[method-assign]
    ip._start_beeps = lambda: None  # type: ignore[method-assign]
    ip._load_skills = AsyncMock(return_value=())  # type: ignore[method-assign]

    async def go(axis):
        await ip._begin_affect(axis, "促し")

    asyncio.run(go("bond"))
    assert ip._load_skills.await_count == 1
    asyncio.run(go("seeking"))  # **反証の側**：話しかけない発火では読まない
    assert ip._load_skills.await_count == 1
    asyncio.run(ip.close())


def test_saying_a_game_name_records_the_day(monkeypatch):
    ip, saved = _ip(monkeypatch, axis="bond")
    ip._note_suggestion_offered("ねえ、しりとりしない？")
    assert saved == [TODAY]


def test_a_voice_without_a_game_name_records_nothing(monkeypatch):
    """**反証の側**：遊びの名前が無い声と、その日もう誘っていた日は数えない。"""
    ip, saved = _ip(monkeypatch, axis="bond")
    ip._note_suggestion_offered("今日もいい天気だね")
    ip2, saved2 = _ip(monkeypatch, axis="bond", last=TODAY)
    ip2._note_suggestion_offered("ねえ、しりとりしない？")
    assert saved == [] and saved2 == []


def test_a_failed_save_is_not_logged_as_invited(monkeypatch, caplog):
    """保存できなかったら「誘った」と残さない（返りを見る）。"""
    ip, saved = _ip(monkeypatch, axis="bond", saves=False)
    with caplog.at_level("INFO"):
        ip._note_suggestion_offered("ねえ、しりとりしない？")
    assert saved == [TODAY]
    assert "event-loop 遊びに誘った" not in [r.getMessage() for r in caplog.records]
    assert "遊びに誘った日を残せなかった" in caplog.text


def test_the_games_know_paju_may_invite():
    """パジュが誘って相手が乗ったときも、Jev がやり方を選べるように「いつ要る」に書いてある。"""
    whens = {s.name: s.when for s in load_skills(SKILLS_DIR)}
    for name in NAMES:
        assert "パジュが誘って" in whens[name]


# ── 話しかける発火の声のあとに窓を開ける（2026-11-15 本人の決定） ──────────────


def test_a_talking_firing_opens_the_window_after_its_voice(monkeypatch):
    """「しりとりしない？」への名前の無い「いいよ」を受けられるよう、言い終えたところから窓を開ける。"""
    import asyncio

    from tests.test_a_named_request_reopens_the_window import _closed_ip

    ip = _closed_ip(monkeypatch, named=False, kind="情動")
    ip._req.fired_axis = "bond"
    asyncio.run(ip._speak("ねえ、しりとりしない？"))
    assert ip._wake_window().is_open(1006.0 + 9.0)
    assert not ip._wake_window().is_open(1006.0 + 10.0)
    assert ip._in_window()  # 言い終えたあとに足す窓の会話の行（`_talk_line`）は、窓の中なら入る


def test_a_quiet_or_device_voice_leaves_the_window_closed(monkeypatch):
    """**反証の側**：話しかける発火でない声（軸の無い情動、機器の知らせ）では、いままでどおり開けない。"""
    import asyncio

    from tests.test_a_named_request_reopens_the_window import _closed_ip

    for kind, axis in (
        ("情動", "seeking"),
        ("情動", "safety"),
        ("情動", "rest"),
        ("情動", ""),
        ("機器", ""),
    ):
        ip = _closed_ip(monkeypatch, named=False, kind=kind)
        ip._req.fired_axis = axis
        asyncio.run(ip._speak("お知らせです"))
        assert not ip._wake_window().is_open(1006.0), (kind, axis)


def test_a_talking_firing_without_a_voice_leaves_the_window_closed(monkeypatch):
    """**反証の側**：声が出なかったら（再生が落ちた）、パジュは何も言っていないので開けない。"""
    import asyncio
    from unittest.mock import AsyncMock

    from tests.test_a_named_request_reopens_the_window import _closed_ip

    ip = _closed_ip(monkeypatch, named=False, kind="情動")
    ip._req.fired_axis = "bond"
    ip._dif.speak = AsyncMock(side_effect=RuntimeError("鳴らせない"))
    with pytest.raises(RuntimeError):  # 再生の例外は、いままでどおり呼び手へ出る
        asyncio.run(ip._speak("ねえ、しりとりしない？"))
    assert not ip._wake_window().is_open(1000.0)


def test_the_invite_is_the_window_talk_and_a_plain_yes_is_heard(monkeypatch):
    """声にして後始末（`_finish`）まで通すと、誘いが窓の会話の行になり、数秒後の名前の無い「いいよ」を窓で受ける。"""
    import asyncio

    from familiar_agent.loop.event_loop import Trigger

    from tests.test_a_named_request_reopens_the_window import _closed_ip

    ip = _closed_ip(monkeypatch, named=False, kind="情動")
    ip._req.fired_axis = "bond"

    async def go():
        spoken, outcome = await ip._speak("ねえ、しりとりしない？")
        await ip._finish(spoken, [], outcome)
        yes = Trigger(kind="会話入力", query="いいよ", source="voice", arrived=1009.0)
        return outcome, await ip._wake_window_admits(yes)

    outcome, admitted = asyncio.run(go())
    assert outcome == "発話"
    assert "ねえ、しりとりしない？" in ip._window_talk().render()
    assert admitted
