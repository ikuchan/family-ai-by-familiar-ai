"""窓が開いていれば、Jev と判定の材料は窓の会話の全部（環-ar 段 B・2026-10-25・本人の決定）。

宛先の判定・返すか黙るかの判定（出-db 段 E）・意味と動作の判定（調停の Jev）は、直近 3 往復（5 分まで）の代わりに、
窓の会話の全部を見る。Jev の字数が足りないときは記憶から外し、窓の会話は外さない。窓が閉じていれば、いまどおり直近。
書き手（軽量LLM）と主LLM の W は、この段では変えない（段 C・段 D）。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from tests.test_event_loop import _agent
from tests.test_jev_gets_a_measured_state import _ws
from tests.test_workspace_is_the_core import _rec

TALK = "[窓の会話（19:00:00 から・古い順・全部）]\n- 19:00:00 パパ：しりとりしよう\n- 19:00:02 わたし（パパへ）：りんご！"


# ── Jev へ送る W ─────────────────────────────────────────────────


def test_the_jev_view_has_the_window_talk_instead_of_the_recent_frame():
    ws = _ws()
    ws.window_talk = TALK
    text = ws.for_jev_arbiter(28_000)
    assert TALK in text
    assert "直近のひとこと" not in text


def test_with_no_window_the_jev_view_keeps_the_recent_frame():
    """**反証の側**：窓の会話が無ければ（窓が閉じている）、いまどおり直近 3 往復。"""
    text = _ws().for_jev_arbiter(28_000)
    assert "直近のひとこと" in text and "窓の会話" not in text


def test_the_happenings_stay_beside_the_window_talk():
    """窓の会話はやりとりの代わり。タイマーや入室など [そのあいだに起きたこと] は、いまどおり載せる。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request
    from tests.test_recent_window_is_a_workspace_frame import _oif, _said

    oif = _oif(
        ["q1", "q2"],
        {
            "q1": [_said("q1", "直近のひとこと", "起点", 1)],
            "q2": [_said("q2", "[タイマー] 3 分たった", "起点", 0, direction="機器")],
        },
    )
    oif.roles = MagicMock(return_value={})
    oif.actors = MagicMock(return_value={})
    ws = workspace.Workspace.build(oif, [], Request(), n_arbiter=2, n_main=2)
    ws.window_talk = TALK
    text = ws.for_jev_arbiter(28_000)
    assert TALK in text and "[タイマー] 3 分たった" in text
    assert "直近のひとこと" not in text


def test_the_main_view_does_not_change_in_this_stage():
    """書き手の W は段 C で変わる（`test_window_talk_goes_to_writer`）。主LLM の W は段 D まで、いまどおり。"""
    ws = _ws()
    ws.window_talk = TALK
    text = ws.for_main
    assert "直近のひとこと" in text and "窓の会話" not in text


def test_memories_go_before_the_window_talk():
    big = "長い記憶の中身。" * 250  # 2,000 字
    ws = _ws([_rec("m1", "高い" + big, fit=0.9), _rec("m2", "低い" + big, fit=0.1)])
    ws.window_talk = TALK
    limit = len(ws.for_jev_arbiter(1_000_000)) - 1_000
    text = ws.for_jev_arbiter(limit)
    assert text is not None and TALK in text
    assert "低い" + big not in text


def test_a_window_talk_too_long_to_fit_is_not_cut():
    """窓の会話は削らない。記憶を全部外しても収まらなければ、送らない（None）。"""
    ws = _ws([_rec("m1", "記憶", fit=0.5)])
    ws.window_talk = TALK + "\n- 19:00:04 パパ：" + "ご" * 2_000
    assert ws.for_jev_arbiter(1_500) is None


# ── 判定に渡す直前のやりとり ───────────────────────────────────────


def _ip():
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._current_speaker_name = lambda: "パパ"
    return ip


def _in_window_request(ip):
    """窓を開け、その窓の中で始まった求めにする。窓の会話の番号を返す。"""
    ip._wake_window().open(time.monotonic())
    ip._req.window_session = ip._wake_window().session
    return ip._req.window_session


def _recent_is(monkeypatch, text, happened=""):
    from familiar_agent.loop import workspace

    monkeypatch.setattr(workspace, "recent_chains", lambda *a, **k: [])
    monkeypatch.setattr(
        workspace,
        "render_recent",
        lambda *a, talk=True, **k: ([], text if talk else happened, {}),
    )


def test_the_judges_get_the_window_talk_in_a_request_of_the_window(monkeypatch):
    ip = _ip()
    _in_window_request(ip)
    ip._talk_line("パパ", "しりとりしよう")
    _recent_is(monkeypatch, "直近 3 往復", happened="[そのあいだに起きたこと]\n- タイマー")
    got = ip._recent_for_judges()
    assert "窓の会話" in got and "パパ：しりとりしよう" in got
    assert "直近 3 往復" not in got
    assert "[そのあいだに起きたこと]" in got  # Jev の W と同じく、起きたことは添える


def test_an_open_window_with_no_lines_yet_gives_nothing_old(monkeypatch):
    """窓が開いたばかりで行がまだ無ければ、前の窓の話を直前として載せない。"""
    ip = _ip()
    _in_window_request(ip)
    _recent_is(monkeypatch, "前の窓の話")
    assert ip._recent_for_judges() == ""


def test_a_request_outside_the_window_gives_the_recent_exchanges(monkeypatch):
    """**反証の側**：窓の外で始まった求めは、いまどおり直近 3 往復（5 分まで）。"""
    ip = _ip()
    _in_window_request(ip)
    ip._talk_line("パパ", "しりとりしよう")
    ip._wake_window().close()
    ip._req.window_session = 0
    _recent_is(monkeypatch, "直近 3 往復")
    assert ip._recent_for_judges() == "直近 3 往復"


def test_a_request_begun_in_the_window_keeps_its_talk_after_the_window_closes(monkeypatch):
    """窓の中で始まった求め（調べものの返りなど）は、窓が切れても同じ窓の会話を見る（段 A の行の足し方と同じ）。"""
    ip = _ip()
    _in_window_request(ip)
    ip._talk_line("パパ", "明日の天気は？")
    ip._wake_window().close()
    _recent_is(monkeypatch, "直近 3 往復")
    assert "パパ：明日の天気は？" in ip._recent_for_judges()


def test_the_addressee_judge_reads_the_talk_of_the_window_the_voice_was_taken_in(monkeypatch):
    """門で受けたときの番号で見る。判定するときに窓が切れていても、その窓の会話で判定する。"""
    ip = _ip()
    session = _in_window_request(ip)
    ip._talk_line("パパ", "しりとりしよう")
    ip._wake_window().close()
    ip._req.window_session = 0
    _recent_is(monkeypatch, "直近 3 往復")
    ip._ask_addressee = AsyncMock(return_value="to_paju")
    asyncio.run(
        ip._between_family(Trigger(kind="会話入力", query="ごりら", window_session=session))
    )
    assert "パパ：しりとりしよう" in ip._ask_addressee.await_args.kwargs["recent"]


def test_a_voice_of_an_old_window_does_not_read_the_new_talk(monkeypatch):
    """**反証の側**：前の窓で受けた声を、新しい窓が開いてから判定するなら、新しい窓の会話は見せない（直近）。"""
    ip = _ip()
    old = _in_window_request(ip)
    ip._wake_window().close()
    _in_window_request(ip)
    ip._talk_line("パパ", "新しい窓の話")
    _recent_is(monkeypatch, "直近 3 往復")
    ip._ask_addressee = AsyncMock(return_value="to_paju")
    asyncio.run(ip._between_family(Trigger(kind="会話入力", query="ごりら", window_session=old)))
    assert ip._ask_addressee.await_args.kwargs["recent"] == "直近 3 往復"


def test_the_silence_judge_reads_the_window_talk():
    """返すか黙るかの判定（出-db 段 E）の材料も窓の会話。"""
    ip = _ip()
    _in_window_request(ip)
    ip._talk_line("パパ", "しりとりしよう")
    ip._req.trigger_kind = "発話"
    inp = ip._arbiter_input(utterance="ごりら", workspace_ctx="", present_ctx="")
    assert "パパ：しりとりしよう" in inp.recent_ctx


def test_the_iteration_gives_the_window_talk_to_the_jev_view(monkeypatch):
    from familiar_agent.loop.arbiter import Arbiter, Decision
    from tests._arbiter_fakes import jev_says, writer_says
    from tests.test_iteration_characterization import _run_until

    a = _agent(stream_returns=[])
    a._jev = jev_says("light")
    a._utility_backend = writer_says({"text": ""})
    got: list = []

    async def capture(self, inp, on_decided=None):
        got.append(inp)
        return Decision(branch="light", text="")

    monkeypatch.setattr(Arbiter, "decide", capture)
    ip = InformationProcessing(a)
    ip._current_speaker_name = lambda: "パパ"
    ip._wake_window().open(time.monotonic())
    ip._wake_window().until += 3600  # 反復が回るあいだ開けておく
    ip._talk_line("パパ", "しりとりしよう", obs_id="obs-1")
    _run_until(ip, lambda: got, push=lambda: ip.push_affect("CALM", "落ち着いている"))
    assert "パパ：しりとりしよう" in got[0].jev_workspace(28_000)
    assert "obs-1" in got[0].jev_workspace.__self__.window_ids


# --- 見直しの指摘（2 回目・3 回目）---


def test_an_open_window_with_no_lines_gives_jev_no_old_talk():
    """窓の中だが行がまだ無いとき（`window_talk` が空文字）、Jev へも前の窓の話を直近として送らない。判定と揃える。"""
    ws = _ws()
    ws.window_talk = ""
    assert "直近のひとこと" not in ws.for_jev_arbiter(28_000)
    ws.window_talk = None  # 窓の外
    assert "直近のひとこと" in ws.for_jev_arbiter(28_000)


def test_the_gate_stamps_the_window_it_took_the_words_in():
    ip = _ip()
    t = Trigger(kind="会話入力", query="しりとりしよう", source="keyboard", named=True)
    assert asyncio.run(ip._swallow_if_unheard(t)) is False
    assert t.window_session == ip._wake_window().session != 0


def test_a_request_takes_the_window_the_gate_gave():
    """門で受けた言葉の求めは、始めるまでに窓が切れても、その窓の求め（返すか黙るかも Jev も窓の会話）。"""
    ip = _ip()
    ip._wake_window().open(time.monotonic())
    session = ip._wake_window().session
    ip._wake_window().close()
    ip._write_origin = AsyncMock(return_value="obs-1")
    ip._pending_window_session = session
    asyncio.run(ip._begin_request(kind="発話", text="ごりら", utterance="ごりら"))
    assert ip._req.window_session == session


def test_a_word_outside_the_window_is_not_added_to_the_last_talk():
    """前の求めが窓の中で始まっていても、窓の外で受けた言葉（名前の無い音楽の言葉など）は閉じた会話に足さない。"""
    ip = _ip()
    _in_window_request(ip)
    ip._wake_window().close()
    ip._iterate = AsyncMock(return_value="")
    ip._write_origin = AsyncMock(return_value="obs-1")
    asyncio.run(ip._utterance_iteration("つぎの曲"))
    assert ip._window_talk().lines == []


# --- 窓が開く前のやりとりは、点が付けば思い出した記憶の列に出る（2026-10-25・本人の決定ア）---


def _ws_across_the_window():
    """窓が開く前の往復・機器の出来事・窓の中の往復（窓の会話の行）。3 つとも想起された。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request
    from tests.test_recent_window_is_a_workspace_frame import _oif, _said

    oif = _oif(
        ["q3", "q2", "q1"],
        {
            "q1": [_said("q1", "窓の前のはなし", "起点", 0)],
            "q2": [_said("q2", "[タイマー] 3 分たった", "起点", 1, direction="機器")],
            "q3": [_said("q3", "窓の中のはなし", "起点", 5)],
        },
    )
    oif.roles = MagicMock(return_value={})
    oif.actors = MagicMock(return_value={})
    recalled = [
        _rec("q1", "窓の前のはなし", fit=0.9),
        _rec("q2", "[タイマー] 3 分たった", fit=0.8),
        _rec("q3", "窓の中のはなし", fit=0.7),
    ]
    ws = workspace.Workspace.build(oif, recalled, Request(), n_arbiter=3, n_main=3)
    ws.window_talk = TALK
    ws.window_ids = frozenset({"q3"})  # 窓の会話の行になった記録
    return ws


def _past(text: str) -> str:
    return text[text.index("[過去の記憶") :] if "[過去の記憶" in text else ""


def test_an_exchange_from_before_the_window_comes_back_as_a_memory():
    text = _ws_across_the_window().for_jev_arbiter(28_000)
    assert "窓の前のはなし" in _past(text)


def test_an_exchange_inside_the_window_is_not_shown_twice():
    """**反証の側**：窓が開いた後の行は窓の会話に載っているので、想起されても記憶の列に出さない。"""
    text = _ws_across_the_window().for_jev_arbiter(28_000)
    assert "窓の中のはなし" not in _past(text)


def test_a_happening_shown_in_its_frame_is_not_shown_twice():
    """**反証の側**：[そのあいだに起きたこと] に載った行も、記憶の列に出さない。"""
    text = _ws_across_the_window().for_jev_arbiter(28_000)
    assert text.count("[タイマー] 3 分たった") == 1
    assert "[タイマー] 3 分たった" not in _past(text)


# --- 見直しの指摘（窓が開く前のやりとり）---


def test_an_old_line_of_a_long_window_is_not_shown_twice():
    """長い窓で直近の数を越えた行も、窓の会話の行になった記録なら、想起されても記憶の列に出さない。"""
    ws = _ws_across_the_window()
    ws.memories.append(_rec("q0", "窓の中の最初のはなし", fit=0.95))
    ws.window_ids = ws.window_ids | {"q0"}
    assert "窓の中の最初のはなし" not in _past(ws.for_jev_arbiter(28_000))


def test_records_of_the_window_that_are_not_lines_stay_as_memories():
    """**反証の側**：窓の中に書いた記録でも、窓の会話の行でないもの（いまの求めの版・知らせの答え）は外さない。

    版は調べた結果の全文を持つ（行は 80 字）。外すと Jev は全文を見られず、調べ直す。
    """
    ws = _ws_across_the_window()
    ws.memories.append(_rec("v-live", "「明日の天気は？」と聞かれ、結果の全文", fit=0.6))
    ws.memories.append(_rec("a-timer", "3 分たちました", fit=0.5))
    past = _past(ws.for_jev_arbiter(28_000))
    assert "結果の全文" in past and "3 分たちました" in past


def test_lines_keep_the_ids_of_their_records():
    ip = _ip()
    ip._wake_window().open(time.monotonic())
    ip._iterate = AsyncMock(return_value="")
    ip._write_origin = AsyncMock(return_value="obs-1")
    asyncio.run(ip._utterance_iteration("しりとりしよう"))
    assert "obs-1" in ip._window_talk().ids


def _asks(ip, kind):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    ip._req.trigger_kind = kind
    ip._dif = MagicMock(speak=AsyncMock())
    ip._confirming = lambda: True  # 問いを出す前に確認待ちが立っている（`_ask_confirm_if_fresh`）
    ip._delivery_block_reason = lambda: ""
    asyncio.run(ip._ask_confirm_aloud(SimpleNamespace(text="5 分のタイマーを掛けていい？")))
    return [line.split(" ", 2)[2] for line in ip._window_talk().lines]


def test_a_confirm_question_is_a_line_of_the_window_it_opens():
    """窓の外で始まった求め（タイマーなど）の確認の問いは、話し終わりに窓を開け、その窓の会話の行になる。宛先は書かない。"""
    ip = _ip()
    assert _asks(ip, "機器") == ["わたし：5 分のタイマーを掛けていい？"]


def test_a_confirm_question_to_a_person_names_them():
    ip = _ip()
    ip._wake_window().open(time.monotonic())
    ip._req.window_session = ip._wake_window().session
    assert _asks(ip, "発話") == ["わたし（パパへ）：5 分のタイマーを掛けていい？"]


def test_the_final_version_of_a_request_in_the_window_is_not_shown_twice():
    """窓の中の前の求めの最終版（問いと答えのまとめ）は、起点が窓の会話の行なので、記憶の列に出さない（記-t）。"""
    from familiar_agent.core import final_version

    ws = _ws_across_the_window()
    final = _rec("f3", final_version.MARK + "19:00〕窓の中のはなしと、その答え", fit=0.9)
    final.mi.parent_id = "q3"
    ws.memories.append(final)
    assert "その答え" not in _past(ws.for_jev_arbiter(28_000))
