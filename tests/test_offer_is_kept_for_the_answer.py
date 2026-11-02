"""パジュの申し出を預け、「はい」なら主LLM へ回す（出-dj・2026-11-02・本人の決定ア・b・a）。

11/02 12:54、主LLM は読んだページが途中までだったと伝え、続きを申し出た。12:55 の「調べて」は意味の道へ回り、申し出を
受ける意味が無いので「頼んだことの確認」→ 状態を伝える、で軽く返して閉じた。申し出は確認待ちにも聞き取りの確認にも
預けられておらず、返事との結び付きが切れていた。話す前の検査と同じ 1 回の Jev の呼び出しで「申し出で終わるか」を聞き、
申し出なら預ける。次の 1 回の返事を、同じ窓が開いているあいだだけ、確認待ちと同じ 3 択で読み、はいなら考えて返す。
Jev の確かめ（`jev-1.14.0`・合成した場面・3 回）：申し出 3 つは 0.99〜1.00、ふつうの答えは 0.00、聞き返しは 0.26〜0.31。
受けた言葉（調べて・はい・うん、お願い）は 3 択で「はい」。タイマーの確認の問いも申し出と読まれた（0.94）ので、
確認待ちのあいだは預けない。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop import arbiter
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, writer_says, ws_of
from tests.test_arbiter_utterance_by_meaning import _a, _jev
from tests.test_event_loop import _agent

OFFER = "読んだページは途中までで、残りは読めませんでした。続きを探してみましょうか？"


# ── 1. 話す前の検査と一緒に、申し出で終わるかを聞く ─────────────────────────


def test_the_check_asks_whether_it_is_an_offer():
    client = MagicMock()
    client.available = True
    client.ask = AsyncMock(return_value=JevAnswer(ok=True, answers={}))
    asyncio.run(
        jev_judges.judge_speech_and_verdicts(
            client,
            response=OFFER,
            recent="",
            facts="",
            rules={},
            rows={},
            min_conf=0.6,
            ask_offer=True,
        )
    )
    questions = client.ask.await_args.args[1]
    assert "offer" in questions and "申し出" in str(questions["offer"])


def test_an_offer_is_read_from_the_answer():
    yes = JevAnswer(ok=True, answers={"offer": _a("offer", 0.95)})
    no = JevAnswer(ok=True, answers={"offer": _a("none", 0.95)})
    unsure = JevAnswer(ok=True, answers={"offer": _a("offer", 0.4)})
    assert jev_judges.offered(yes, 0.6) is True
    assert jev_judges.offered(no, 0.6) is False
    assert jev_judges.offered(unsure, 0.6) is False  # **反証の側**：迷ったら預けない


# ── 2. ループが預ける ─────────────────────────────────────────────


def _checked(monkeypatch, answers: dict, *, confirming: bool = False):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    a = _agent(stream_returns=[])
    a.config.speech_check = True
    a.config.jev_confidence_min = 0.6
    a._jev = MagicMock()
    a._jev.available = True
    a._jev.ask = AsyncMock(return_value=JevAnswer(ok=True, answers=answers))
    a._pending_confirm = (
        MagicMock() if confirming else None
    )  # 偽のエージェントは MagicMock なので明示する
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"

    async def go():
        await ip._speech_check_violation(OFFER, "", [])
        await ip.close()

    asyncio.run(go())
    return ip


def test_a_spoken_offer_is_kept(monkeypatch):
    ip = _checked(monkeypatch, {"broken": _a("ok", 0.9), "offer": _a("offer", 0.95)})
    assert ip._offer is not None and ip._offer["question"] == OFFER


def test_not_an_offer_is_not_kept(monkeypatch):
    """**反証の側**：申し出でない返事は預けない。"""
    ip = _checked(monkeypatch, {"broken": _a("ok", 0.9), "offer": _a("none", 0.95)})
    assert ip._offer is None


def test_a_sent_back_offer_is_not_kept(monkeypatch):
    """**反証の側**：差し戻した返事は声にならないので預けない。"""
    answers = {"broken": _a("no-invented-knowledge", 0.9), "offer": _a("offer", 0.95)}
    ip = _checked(monkeypatch, answers)
    assert ip._offer is None


def test_a_later_reply_that_is_not_an_offer_replaces_it(monkeypatch):
    """**反証の側**：あとで申し出でない返事を話したら、前の申し出は預からない（次の「うん」はその返事への答え）。"""
    ip = _checked(monkeypatch, {"broken": _a("ok", 0.9), "offer": _a("none", 0.95)})
    ip._offer = {"question": OFFER, "session": 1}

    async def go():
        await ip._speech_check_violation("わかった。", "", [])
        await ip.close()

    asyncio.run(go())
    assert ip._offer is None


def test_nothing_is_kept_while_confirming(monkeypatch):
    """**反証の側**：確認待ちの問いも申し出と読まれる（0.94）。確認待ちが読むので預けない。"""
    answers = {"broken": _a("ok", 0.9), "offer": _a("offer", 0.95)}
    ip = _checked(monkeypatch, answers, confirming=True)
    assert ip._offer is None


# ── 3. 次の 1 回の返事に、同じ窓のあいだだけ渡す ──────────────────────────


def _loop():
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.trigger_kind = "発話"
    return ip


def _decide_with(ip, decision=None):
    seen: dict = {}

    async def fake_decide(self, inp, on_decided=None):
        seen["inp"] = inp
        return decision or arbiter.Decision(branch="light", text="うん")

    with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
        asyncio.run(
            ip._decide(utterance="調べて", ws=ws_of(), present_ctx="", capped=False, round_=1)
        )
    return seen["inp"]


def _kept(ip):
    """申し出を預け、次の返事は同じ窓の会話で受けた（門が決めた求めの窓の番号・環-ar 段 B）。"""
    wake = ip._wake_window()
    wake.open(time.monotonic())
    ip._offer = {"question": OFFER, "session": wake.session}
    ip._req.window_session = wake.session


def test_the_next_answer_carries_the_offer_once():
    ip = _loop()
    _kept(ip)
    assert _decide_with(ip).offer_question == OFFER
    assert _decide_with(ip).offer_question == ""  # **反証の側**：2 回目の返事では使わない


def test_another_window_drops_the_offer():
    """**反証の側**：窓の外で始まった求め（番号 0）や、開け直した窓（別の会話）の返事では使わない。"""
    ip = _loop()
    _kept(ip)
    ip._req.window_session = 0
    assert _decide_with(ip).offer_question == ""
    _kept(ip)
    ip._offer["session"] -= 1
    assert _decide_with(ip).offer_question == ""


def test_a_window_that_timed_out_after_the_gate_still_counts():
    """門で窓の会話として受けた返事は、判断までに窓が切れても使う（`_begin_request` と同じ扱い）。"""
    ip = _loop()
    _kept(ip)
    ip._wake_window().close()
    assert _decide_with(ip).offer_question == OFFER


# ── 4. 調停が 3 択で読み、はいなら考えて返す ───────────────────────────


def test_yes_to_the_offer_goes_to_the_main_llm(monkeypatch):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"answer": _a("yes", 0.9)})
    d = asyncio.run(decide(jev=jev, utterance="調べて", offer_question=OFFER))
    assert (d.branch, d.offer_accepted) == ("full", True)
    state, questions = jev.ask.await_args_list[0].args
    assert set(questions) == {"answer"} and OFFER in state


def test_no_or_other_reads_the_words_as_usual(monkeypatch):
    """**反証の側**：いいえ・それ以外は、いまの意味の道のまま。"""
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    for pick in ("no", "other"):
        jev = _jev(
            {"answer": _a(pick, 0.9)},
            {"meaning": _a("other", 0.9), "action_other": _a("reply_light", 0.9)},
        )
        d = asyncio.run(
            decide(
                jev=jev,
                writer=writer_says({"text": "そうなんだ"}),
                utterance="いいや",
                offer_question=OFFER,
            )
        )
        assert (d.branch, d.offer_accepted) == ("light", False)


def test_a_waiting_confirm_reads_first(monkeypatch):
    """**反証の側**：確認待ちがあれば、確認待ちが先に読む。"""
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"answer": _a("yes", 0.9)})
    d = asyncio.run(
        decide(jev=jev, utterance="はい", offer_question=OFFER, confirm_question="掛けますか？")
    )
    assert (d.branch, d.action) == ("action", "confirm")


# ── 5. 主LLM に 1 行添える（本人の決定 b） ─────────────────────────────


def test_the_main_llm_is_told_the_offer_was_accepted():
    ip = _loop()
    _kept(ip)
    _decide_with(ip, arbiter.Decision(branch="full", offer_accepted=True))
    system = "\n".join(ip._build_system(present_ctx="", workspace_ctx="", iter_ctx=""))
    assert (
        f"パジュの申し出（「{OFFER}」）に、相手は受けると答えた。申し出たことを、いまやる" in system
    )


def test_the_line_stays_until_the_main_llm_acts_and_a_new_request_starts_clean():
    """差し戻しの書き直しにも載る。調べものを投げたあとの反復や、次の求めには載せない（器は求めをまたいで使い回す）。"""
    from familiar_agent.loop.event_loop import Lookup

    ip = _loop()
    _kept(ip)
    _decide_with(ip, arbiter.Decision(branch="full", offer_accepted=True))
    ip._build_system(present_ctx="", workspace_ctx="", iter_ctx="")
    again = "\n".join(ip._build_system(present_ctx="", workspace_ctx="", iter_ctx=""))
    assert "パジュの申し出" in again  # 書き直し（調べものはまだ投げていない）
    ip._req.lookups.append(Lookup(index=1, action="search_deferred", query="続き", generation=0))
    after = "\n".join(ip._build_system(present_ctx="", workspace_ctx="", iter_ctx=""))
    assert "パジュの申し出" not in after  # **反証の側**：やったあとは「いまやる」を繰り返さない
    ip._req.accepted_offer = OFFER
    asyncio.run(ip._begin_request(kind="情動", text="ひま"))
    assert ip._req.accepted_offer == ""


def test_without_an_accepted_offer_nothing_is_added():
    """**反証の側**：受けていなければ添えない。"""
    ip = _loop()
    _kept(ip)
    _decide_with(ip)
    system = "\n".join(ip._build_system(present_ctx="", workspace_ctx="", iter_ctx=""))
    assert "パジュの申し出" not in system
