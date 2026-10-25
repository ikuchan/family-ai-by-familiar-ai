"""迷った聞き返しは聞き取りの確認にし、「はい」で見立ての動作を投げる（出-cm・2026-10-15・本人の決定①②）。

迷って聞き返すと「タイマーを開始しますか？」のような実行の確認を書き、次の「はい」はふつうの言葉として読まれて何も
動かなかった。聞き返しは「3分のタイマーと言ったの？」のように聞き取りを確かめる形にし、見立ての動作と元の言葉を預ける。
次の発話は確認待ちと同じ 3 択（はい・いいえ・それ以外）で読み、はいなら見立ての動作を元の言葉でふつうの道で書いて投げる。
預かりは 30 秒〔仮〕で切れ、次の発話で答えがどれでも消える。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

from familiar_agent.core import utterance_meaning as um
from familiar_agent.loop import arbiter
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.store import arbiter_records
from tests._arbiter_fakes import decide, prompt_of, writer_says, ws_of
from tests.test_arbiter_utterance_by_meaning import _a, _jev
from tests.test_event_loop import _agent


def _probs(choice: str, confidence: float, probs: dict) -> dict:
    return {"choice": choice, "confidence": confidence, "probabilities": probs}


# --- 1. 見立ての動作の名前 ---


def test_the_guessed_action_is_the_top_action_besides_asking():
    action = _probs("ask_back", 0.4, {"ask_back": 0.4, "set_timer": 0.35, "start_stopwatch": 0.25})
    assert um.guess_action(_a("time", 0.5), action) == "set_timer"


def test_no_guessed_action_for_names_denials_and_unread_words():
    """**反証の側**：名乗り・否定・文脈に合わない言葉・成立しないもの・聞き返すだけ、には動作が無い。"""
    assert um.guess_action(_a("claim", 0.5), _a("パパ", 0.5)) == ""
    assert um.guess_action(_a("deny", 0.5), _a("パパ", 0.5)) == ""
    assert um.guess_action(_a("off_context", 0.4), _a("ask_back", 0.9)) == ""
    assert um.guess_action(_a("unformed", 0.4), None) == ""
    only_asking = _probs("ask_back", 0.9, {"ask_back": 0.9, "silent": 0.1})
    assert um.guess_action(_a("time", 0.9), only_asking) == ""


# --- 2. 聞き返しの頼み文 ---


def test_the_ask_back_asks_what_was_heard_not_whether_to_do_it():
    words = arbiter._ASK_GUESS_WORDS + arbiter._FIELD_TEXT["text_ask_guess"]
    assert "と言ったの？" in words  # 聞き取りの確認の形
    assert "しますか？" in arbiter._FIELD_TEXT["text_ask_guess"]  # 実行の確認の形は名指しで禁じる
    assert "見立てが合っているか" not in words


# --- 3. 調停が見立てつきの聞き返しで預かりを載せる ---


def _doubtful_timer(monkeypatch, *, writer_text: str = "3分のタイマーと言ったの？"):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev(
        {
            "meaning": _a("time", 0.5),
            # 聞き返すが確率でまさる（出-db 段 A で 2 回目の問いを外した）。見立ては聞き返す・黙るを除いた最大のタイマー。
            "action_time": _probs(
                "ask_back", 0.6, {"ask_back": 0.6, "set_timer": 0.3, "silent": 0.1}
            ),
        },
    )
    return asyncio.run(
        decide(jev=jev, writer=writer_says({"text": writer_text}), utterance="3分測って")
    )


def test_a_guessed_ask_back_carries_the_heard_check(monkeypatch):
    d = _doubtful_timer(monkeypatch)
    assert (d.branch, d.text) == ("light", "3分のタイマーと言ったの？")
    assert d.heard_check == {"action": "set_timer", "utterance": "3分測って"}


def test_a_plain_ask_back_carries_nothing(monkeypatch):
    """**反証の側**：動作の見立ての無い聞き返し（確信度の足りない否定）は預けない。

    文脈に合わない言葉は、出-db 段 B から聞き返さずに黙るので、ここでは否定で確かめる。
    """
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"meaning": _a("deny", 0.5), "action_deny": _a("パパ", 0.5)})
    d = asyncio.run(decide(jev=jev, writer=writer_says({"text": "なんて？"}), utterance="ちがう"))
    assert (d.branch, d.text) == ("light", "なんて？") and d.heard_check is None


# --- 4. 預かりがあるときの答え ---

_HEARD = {
    "heard_check_action": "set_timer",
    "heard_check_utterance": "3分測って",
    "heard_check_question": "3分のタイマーと言ったの？",
}


def test_yes_runs_the_guessed_action_with_the_original_words(monkeypatch):
    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    jev = _jev({"answer": _a("yes", 0.9)})
    writer = writer_says({"tool_input": {"after_minutes": 3, "label": ""}})
    d = asyncio.run(
        decide(jev=jev, writer=writer, utterance="うん", extra_actions=("set_timer",), **_HEARD)
    )
    assert (d.branch, d.action) == ("action", "set_timer")
    assert d.tool_input and d.tool_input.get("after_minutes") == 3
    state, questions = jev.ask.await_args_list[0].args
    assert set(questions) == {"answer"}  # 3 択だけを聞いた
    assert "3分のタイマーと言ったの？" in state and "うん" in state
    assert "3分測って" in prompt_of(writer)  # 書き手には元の言葉を渡す


def test_no_or_other_reads_the_words_as_usual(monkeypatch):
    """**反証の側**：いいえ・それ以外なら、見立ての動作は投げず、意味の問いからふつうに読む。"""
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
                utterance="ちがうよ、天気の話",
                extra_actions=("set_timer",),
                **_HEARD,
            )
        )
        assert (d.branch, d.text) == ("light", "そうなんだ")
        assert "meaning" in jev.ask.await_args_list[1].args[1]


# --- 5. ループが預かる・渡す・消す ---


def _loop():
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "発話"
    return ip


def _decide_with(ip, decision, utterance: str):
    seen: dict = {}

    async def fake_decide(self, inp, on_decided=None):
        seen["inp"] = inp
        return decision

    with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
        asyncio.run(
            ip._decide(utterance=utterance, ws=ws_of(), present_ctx="", capped=False, round_=1)
        )
    return seen["inp"]


def _asked():
    return arbiter.Decision(
        branch="light",
        text="3分のタイマーと言ったの？",
        heard_check={"action": "set_timer", "utterance": "3分測って"},
    )


def _plain():
    return arbiter.Decision(branch="light", text="うん")


def test_the_loop_keeps_the_check_and_passes_it_once():
    ip = _loop()
    first = _decide_with(ip, _asked(), "3分測って")
    assert first.heard_check_action == ""
    second = _decide_with(ip, _plain(), "うん")
    assert (second.heard_check_action, second.heard_check_utterance) == ("set_timer", "3分測って")
    assert second.heard_check_question == "3分のタイマーと言ったの？"
    # **反証の側**：使った預かりは消える（2 回目の「はい」では動かない）。
    third = _decide_with(ip, _plain(), "うん")
    assert third.heard_check_action == ""


def test_an_old_check_is_not_used():
    """**反証の側**：30 秒を過ぎた預かりは渡さない。"""
    ip = _loop()
    _decide_with(ip, _asked(), "3分測って")
    ip._heard_check["at"] = time.monotonic() - 31
    later = _decide_with(ip, _plain(), "うん")
    assert later.heard_check_action == ""


def test_keeping_and_answering_are_logged_without_the_words(caplog, monkeypatch):
    import logging

    monkeypatch.setattr(arbiter_records, "record", lambda **kw: None)
    ip = _loop()
    with caplog.at_level(logging.INFO):
        _decide_with(ip, _asked(), "3分測って")
        asyncio.run(
            decide(
                jev=_jev({"answer": _a("yes", 0.9)}),
                writer=writer_says({"tool_input": {"after_minutes": 3}}),
                utterance="うん",
                extra_actions=("set_timer",),
                **_HEARD,
            )
        )
    lines = [r.getMessage() for r in caplog.records if "聞き取りの確認" in r.getMessage()]
    assert any("預けた" in x and "set_timer" in x for x in lines)
    assert any("はい" in x and "set_timer" in x for x in lines)
    assert not [x for x in lines if "3分測って" in x]


def test_a_tool_return_iteration_does_not_use_up_the_check():
    """道具の返りの反復（発話の求めの続き）では、預かりを渡さず消さない。"""
    ip = _loop()
    _decide_with(ip, _asked(), "3分測って")
    ws = MagicMock(
        for_arbiter="",
        memories=[],
        returned_actions=frozenset({"recall"}),
        returned_lookups=(("recall", False, "r"),),
    )
    seen: dict = {}

    async def fake_decide(self, inp, on_decided=None):
        seen["inp"] = inp
        return _plain()

    with patch("familiar_agent.loop.arbiter.Arbiter.decide", new=fake_decide):
        asyncio.run(ip._decide(utterance="", ws=ws, present_ctx="", capped=False, round_=2))
    assert seen["inp"].heard_check_action == ""
    assert ip._heard_check is not None
