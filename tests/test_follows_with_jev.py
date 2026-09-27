"""続き先を Jev が決める（出-au 段 5-4・2026-09-27・`設計方針_判定の段` §2.2.3）。

W 全部とその場の言葉を送り、「この言葉は W のどの行の続きか」を Choice で聞く。選択肢は W の中の `id:<12桁>` を持つ
行（id → その行の文）と「どれの続きでもない」。

- 確信度 0.6 未満なら「どれの続きでもない」（計測ログの結末は「途切れ」）。
- Jev が使えない・失敗したときは `JudgeFailed` を投げる。呼び手は「落ちた」と数えて辺を書かない（倒し先と同じ動き。
  「途切れ」と混ぜない）。
- 言葉か W が無ければ聞かない（「未判定」）。軽量LLM の口（`Evaluator.judge_follows`）は外した。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import jev_judges

_W = """[直近のやりとり（古い順）]
- 17:47 id:0123456789ab 相手：明日の運動会って何時から？
- 17:48 id:ba9876543210 わたし：自分が答えた：9時からだよ
[過去の記憶]
- 2026-09-01 id:aaaabbbbcccc (適合度:0.5) 相手が言った: キャンプの話"""


def _client(answer=None, exc=None):
    c = MagicMock()
    c.available = True
    c.ask = AsyncMock(side_effect=exc, return_value=answer)
    return c


def _ans(pick, conf=0.8):
    return JevAnswer(ok=True, answers={"follows": {"choice": pick, "confidence": conf}})


def _judge(client, utterance="開会式って何時だっけ", w=_W):
    return asyncio.run(
        jev_judges.judge_follows(client, workspace=w, utterance=utterance, min_conf=0.6)
    )


def test_the_rows_of_the_workspace_are_the_choices():
    c = _client(_ans("0123456789ab"))
    assert _judge(c) == "0123456789ab"
    state, questions = c.ask.await_args.args
    crit = questions["follows"]["criteria"]
    assert set(crit) == {"0123456789ab", "ba9876543210", "aaaabbbbcccc", "none"}
    assert "運動会" in crit["0123456789ab"]
    assert "開会式って何時だっけ" in state


def test_none_and_low_confidence_mean_no_continuation():
    assert _judge(_client(_ans("none"))) is None
    assert _judge(_client(_ans("0123456789ab", 0.5))) is None


def test_failure_is_raised_so_it_is_counted_apart():
    with pytest.raises(jev_judges.JudgeFailed):
        _judge(_client(JevAnswer(ok=False, error="時間切れ")))
    with pytest.raises(jev_judges.JudgeFailed):
        _judge(None)


def test_no_words_or_no_workspace_is_not_asked():
    c = _client(_ans("0123456789ab"))
    assert _judge(c, utterance="") is None
    assert _judge(c, w="") is None
    c.ask.assert_not_awaited()


def test_the_loop_asks_jev_and_the_light_llm_judge_is_gone():
    import inspect

    from familiar_agent.loop import evaluator
    from familiar_agent.loop.event_loop import InformationProcessing

    assert "self._judge_follows(" in inspect.getsource(InformationProcessing._iterate)
    assert "jev_judges.judge_follows" in inspect.getsource(InformationProcessing._judge_follows)
    assert not hasattr(evaluator.Evaluator, "judge_follows")
    assert not hasattr(evaluator, "_FOLLOWS_PROMPT")


def fake_follows(a, verdict=None, *, exc=None):
    """ループの試験で、続き先の判定を Jev の偽物にする（`verdict` は選ばせる id、None は「どれでもない」）。"""
    a._jev = _client(None if exc else _ans(verdict or "none", 0.9), exc=exc)
    a.config.jev_confidence_min = 0.6
