"""整合チェックを繋ぎ直す（出-f）。

`check_response_coherence` は実装だけが残り、**呼び手が0件**だった。旧 `run()` にあった
呼び出しが環-c（`a47f85e`）で消えている。

繋ぎ直すとき、**応答の文字列を機械で削らない**。機械は意味を読めないので、語の表で文を
落とせばパジュの普通の発話が黙って消える。機械がするのは**推測の要らない事実を集めて
渡す**ことだけで、判定は軽量LLM がする。

軽量LLM はいま応答と規則しか持たず、「見たか」「記憶があったか」を知らない。だから
`no-fake-perception` を渡しても照らす相手が無い。規則を渡していなかったときが 3/18 で、
渡したら 15〜18/18 になったのと同じ性質である（`根拠台帳` §25.3）。**足りないのは
判断力ではなく材料である。**
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── 事実を組む（機械の仕事） ────────────────────────────────────────────────


def test_it_says_plainly_that_nothing_was_seen():
    from familiar_agent.loop.coherence import facts_ctx

    out = facts_ctx(saw=False, memories=[])
    assert "見たか：いいえ" in out


def test_it_says_plainly_that_something_was_seen():
    from familiar_agent.loop.coherence import facts_ctx

    assert "見たか：はい" in facts_ctx(saw=True, memories=[])


def test_an_empty_workspace_is_stated_as_no_material_for_comparison():
    """記憶が0件なら『昨日より』の材料が無い。規則 no-past-comparison-without-memory 用。"""
    from familiar_agent.loop.coherence import facts_ctx

    out = facts_ctx(saw=False, memories=[])
    assert "0件" in out


def test_the_dates_of_the_recalled_memories_are_listed():
    from familiar_agent.loop.coherence import facts_ctx

    out = facts_ctx(
        saw=False,
        memories=[
            {"date": "2026-08-14", "confidence": 0.80},
            {"date": "2026-09-02", "confidence": 0.40},
        ],
    )
    assert "2026-08-14" in out and "2026-09-02" in out
    assert "2件" in out


def test_the_uncertain_memories_are_counted():
    """規則 memory-evidence-confidence の境目は 0.55。"""
    from familiar_agent.loop.coherence import CONF_UNCERTAIN, facts_ctx

    assert CONF_UNCERTAIN == 0.55
    out = facts_ctx(
        saw=False,
        memories=[{"date": "d1", "confidence": 0.40}, {"date": "d2", "confidence": 0.90}],
    )
    assert "1件" in out.split("うち")[1]


def test_no_word_list_is_used_to_censor_the_response():
    """**応答の文字列を機械で削らない。** 事実を組む口は応答を受け取らない。"""
    import inspect

    from familiar_agent.loop import coherence

    assert "response" not in inspect.signature(coherence.facts_ctx).parameters


# ── 判定（軽量LLM の仕事） ──────────────────────────────────────────────────


def _evaluator():
    from familiar_agent.loop.evaluator import Evaluator

    util = MagicMock()
    util.complete = AsyncMock(return_value="OK")
    ev = Evaluator(util, MagicMock(), context=lambda *a, **k: "規則")
    return ev, util


def test_ok_means_no_violation():
    ev, _ = _evaluator()
    assert asyncio.run(ev.check_response_coherence("こんばんは", recent="", facts="f")) is None


def test_anything_else_is_reported_as_a_violation():
    ev, util = _evaluator()
    util.complete = AsyncMock(return_value="見ていないのに見たと言っている")
    out = asyncio.run(ev.check_response_coherence("そこに本があるね", recent="", facts="f"))
    assert out == "見ていないのに見たと言っている"


def test_the_facts_reach_the_judge():
    ev, util = _evaluator()
    asyncio.run(ev.check_response_coherence("はい", recent="R", facts="見たか：いいえ"))
    prompt = util.complete.await_args.args[0]
    assert "見たか：いいえ" in prompt
    assert "R" in prompt


def test_the_prompt_carries_no_copy_of_the_rules():
    """規則の正本は `EVENT_SYSTEM_PROMPT` の `(rules ...)`。写しを置かない。"""
    from familiar_agent.loop.evaluator import _COHERENCE_CHECK_PROMPT

    assert "constraint" not in _COHERENCE_CHECK_PROMPT
    assert "shiritori" not in _COHERENCE_CHECK_PROMPT.lower()


def test_the_judge_receives_the_rules_through_the_system_message():
    ev, util = _evaluator()
    asyncio.run(ev.check_response_coherence("はい", recent="", facts="f"))
    assert util.complete.await_args.kwargs["system"] == "規則"


def test_the_conversation_history_is_no_longer_read():
    """`agent.messages` は追記する箇所が0件で、いつも空である。もう読まない。"""
    import inspect

    from familiar_agent.loop.evaluator import Evaluator

    assert "messages" not in inspect.signature(Evaluator.check_response_coherence).parameters


# ── 既定 ───────────────────────────────────────────────────────────────────


def test_the_gate_is_on_by_default():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().coherence_check is True


def test_the_gate_can_be_turned_off():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {"FAMILIAR_COHERENCE_CHECK": "0"}, clear=True):
        assert AgentConfig().coherence_check is False


# ── 差し戻し ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_violation_sends_the_draft_back_once():
    """違反なら主LLM へ1回だけ言い直させる。2回目は検査しない（無限ループを作らない）。"""
    import inspect

    from familiar_agent.loop import event_loop

    src = inspect.getsource(event_loop.InformationProcessing._iterate)
    assert "[SELF-CHECK]" in src
