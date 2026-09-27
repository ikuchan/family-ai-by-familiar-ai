"""考え直すかを Jev が決める（出-au 段 5-1・2026-09-27・`設計方針_判定の段` §2.2.3・§2.4）。

主LLM が考えているあいだに名前無しで言い足された言葉（「あ、東京のね」）があるとき、返りが来た時点で、
元の問い・主LLM の返事・言い足された言葉を組にして Jev に聞く。

- **考え直す**：その返事は声にしない。待っていた言葉を前の求めに添え、考えかけていた返事を W に載せて、決める反復を
  もう一度回す。
- **そのまま出す**：返事を声にし、言葉は答えの後の別の求めにする（いままでどおり）。
- Jev が使えない（鍵が無い・失敗・時間切れ）、確信度が 0.6 未満のときは、倒し先の「そのまま出す」。
- 待っている言葉が無ければ Jev を呼ばない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges
from familiar_agent.loop.event_loop import Trigger

from tests.test_act_on_decision import _ip, _run, _turn


def _answer(pick: str, conf: float) -> JevAnswer:
    return JevAnswer(ok=True, answers={"rethink": {"choice": pick, "confidence": conf}})


def _client(answer=None, exc=None):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(side_effect=exc, return_value=answer)
    return c


# ── 判定そのもの（純粋な部分） ───────────────────────────────────────────────


def _judge(client, min_conf=0.6):
    return asyncio.run(
        jev_judges.judge_rethink(
            client,
            question="明日の天気は？",
            draft="晴れだよ",
            added=["あ、東京のね"],
            min_conf=min_conf,
        )
    )


def test_a_confident_rethink_is_taken():
    assert _judge(_client(_answer("rethink", 0.8))) == jev_judges.RETHINK


def test_a_low_confidence_falls_to_as_is():
    assert _judge(_client(_answer("rethink", 0.5))) == jev_judges.AS_IS


def test_failures_fall_to_as_is():
    assert _judge(None) == jev_judges.AS_IS
    assert _judge(_client(JevAnswer(ok=False, error="時間切れ"))) == jev_judges.AS_IS
    assert _judge(_client(exc=RuntimeError("down"))) == jev_judges.AS_IS
    unavailable = _client(_answer("rethink", 0.9))
    unavailable.available = False
    assert _judge(unavailable) == jev_judges.AS_IS
    unavailable.ask.assert_not_awaited()


def test_the_state_carries_the_question_the_draft_and_the_added_words():
    c = _client(_answer("as_is", 0.9))
    _judge(c)
    state, questions = c.ask.await_args.args
    assert "明日の天気は？" in state and "晴れだよ" in state and "あ、東京のね" in state
    assert set(questions["rethink"]["criteria"]) == {"rethink", "as_is"}


# ── ループの中 ───────────────────────────────────────────────────────────


def _loop(answer, *, held=True):
    ip, a = _ip()
    ip._req.utterance = "明日の天気は？"
    a._jev = _client(answer)
    a.config.jev_confidence_min = 0.6
    ip._held = [Trigger(kind="会話入力", query="あ、東京のね", named=False)] if held else []
    ip._iterate = AsyncMock(return_value="")
    ip._attach_to_request = AsyncMock()
    return ip, a


def _say():
    return _turn(ToolCall(id="s", name="say", input={"text": "晴れだよ"}))


def test_rethinking_does_not_speak_and_runs_again_with_the_added_words():
    ip, a = _loop(_answer("rethink", 0.8))
    _run(ip, _say())
    ip._speak.assert_not_awaited()
    ip._attach_to_request.assert_awaited_once()
    assert ip._attach_to_request.await_args.args[0].query == "あ、東京のね"
    assert ip._held == []  # 待っていた言葉は前の求めへ移った
    assert ip._req.draft == "晴れだよ"  # 考えかけていた返事は W に載る
    ip._iterate.assert_awaited_once()


def test_as_is_speaks_and_keeps_the_words_waiting():
    ip, a = _loop(_answer("as_is", 0.9))
    _run(ip, _say())
    ip._speak.assert_awaited_once()
    assert [t.query for t in ip._held] == ["あ、東京のね"]
    ip._iterate.assert_not_awaited()


def test_nothing_waiting_means_no_question():
    ip, a = _loop(_answer("rethink", 0.9), held=False)
    _run(ip, _say())
    a._jev.ask.assert_not_awaited()
    ip._speak.assert_awaited_once()


def test_the_draft_is_shown_in_the_workspace():
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    req = Request()
    req.draft = "晴れだよ"
    oif = AsyncMock()
    oif.actors = lambda ids: {}
    oif.roles = lambda ids: {}
    text, _ = workspace.compose(oif, [], req)
    assert "晴れだよ" in text and "考えかけていた返事" in text
