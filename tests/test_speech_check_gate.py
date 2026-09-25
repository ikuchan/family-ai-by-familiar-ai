"""発話前の検査を繋ぎ直す（出-f）。

`check_speech` は実装だけが残り、**呼び手が0件**だった。旧 `run()` にあった
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


def _rec(obs_id="m1", content="昔の話", fit=0.5, conf=0.8, direction="発話", day=(2026, 9, 11)):
    """想起は口から `Recalled` で来る（環-e-い）。"""
    from datetime import datetime

    from familiar_agent.io.oif import MI, Recalled

    return Recalled(
        mi=MI(
            id=obs_id,
            obs_id=obs_id,
            content=content,
            timestamp=datetime(*day, 15, 0),
            direction=direction,
        ),
        fit=fit,
        groundedness=1.0,
        confidence=conf,
    )


# ── 事実を組む（機械の仕事） ────────────────────────────────────────────────


def test_it_says_plainly_that_nothing_was_seen():
    from familiar_agent.loop.speech_check import facts_ctx

    out = facts_ctx(saw=False, memories=[])
    assert "見たか：いいえ" in out


def test_it_says_plainly_that_something_was_seen():
    from familiar_agent.loop.speech_check import facts_ctx

    assert "見たか：はい" in facts_ctx(saw=True, memories=[])


def test_an_empty_workspace_is_stated_as_no_material_for_comparison():
    """記憶が0件なら『昨日より』の材料が無い。規則 no-past-comparison-without-memory 用。"""
    from familiar_agent.loop.speech_check import facts_ctx

    out = facts_ctx(saw=False, memories=[])
    assert "0件" in out


def test_the_dates_of_the_recalled_memories_are_listed():
    from familiar_agent.loop.speech_check import facts_ctx

    out = facts_ctx(
        saw=False,
        memories=[
            _rec("m1", conf=0.80, day=(2026, 8, 14)),
            _rec("m2", conf=0.40, day=(2026, 9, 2)),
        ],
    )
    assert "2026-08-14" in out and "2026-09-02" in out
    assert "2件" in out


def test_the_uncertain_memories_are_counted():
    """規則 memory-evidence-confidence の境目は 0.55。"""
    from familiar_agent.loop.speech_check import CONF_UNCERTAIN, facts_ctx

    assert CONF_UNCERTAIN == 0.55
    out = facts_ctx(
        saw=False,
        memories=[_rec("m1", conf=0.40), _rec("m2", conf=0.90)],
    )
    assert "1件" in out.split("うち")[1]


def test_no_word_list_is_used_to_censor_the_response():
    """**応答の文字列を機械で削らない。** 事実を組む口は応答を受け取らない。"""
    import inspect

    from familiar_agent.loop import speech_check

    assert "response" not in inspect.signature(speech_check.facts_ctx).parameters


# ── 判定（軽量LLM の仕事） ──────────────────────────────────────────────────


def _evaluator():
    from familiar_agent.loop.evaluator import Evaluator

    util = MagicMock()
    util.complete = AsyncMock(return_value="OK")
    ev = Evaluator(util, MagicMock(), context=lambda *a, **k: "規則")
    return ev, util


def test_ok_means_no_violation():
    ev, _ = _evaluator()
    assert asyncio.run(ev.check_speech("こんばんは", recent="", facts="f")) is None


def test_anything_else_is_reported_as_a_violation():
    ev, util = _evaluator()
    util.complete = AsyncMock(return_value="見ていないのに見たと言っている")
    out = asyncio.run(ev.check_speech("そこに本があるね", recent="", facts="f"))
    assert out == "見ていないのに見たと言っている"


def test_the_facts_reach_the_judge():
    ev, util = _evaluator()
    asyncio.run(ev.check_speech("はい", recent="R", facts="見たか：いいえ"))
    prompt = util.complete.await_args.args[0]
    assert "見たか：いいえ" in prompt
    assert "R" in prompt


def test_the_prompt_carries_no_copy_of_the_rules():
    """規則の正本は `EVENT_SYSTEM_PROMPT` の `(rules ...)`。写しを置かない。"""
    from familiar_agent.loop.evaluator import _SPEECH_CHECK_PROMPT

    assert "constraint" not in _SPEECH_CHECK_PROMPT
    assert "shiritori" not in _SPEECH_CHECK_PROMPT.lower()


def test_the_judge_receives_the_rules_through_the_system_message():
    ev, util = _evaluator()
    asyncio.run(ev.check_speech("はい", recent="", facts="f"))
    assert util.complete.await_args.kwargs["system"] == "規則"


def test_the_conversation_history_is_no_longer_read():
    """`agent.messages` は追記する箇所が0件で、いつも空である。もう読まない。"""
    import inspect

    from familiar_agent.loop.evaluator import Evaluator

    assert "messages" not in inspect.signature(Evaluator.check_speech).parameters


# ── 既定 ───────────────────────────────────────────────────────────────────


def test_the_gate_is_on_by_default():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().speech_check is True


def test_the_gate_can_be_turned_off():
    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {"FAMILIAR_SPEECH_CHECK": "0"}, clear=True):
        assert AgentConfig().speech_check is False


# ── 差し戻し ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_violation_sends_the_draft_back_once():
    """違反なら主LLM へ1回だけ言い直させる。2回目は検査しない（無限ループを作らない）。"""
    import inspect

    from familiar_agent.loop import event_loop

    # 差し戻しは `_act_on_decision` にある（環-h・段ろ で切り出した）。
    src = inspect.getsource(event_loop.InformationProcessing._act_on_decision)
    assert "[SELF-CHECK]" in src


# ── チェッカーに渡す規則は「文と事実で判定できるもの」だけ（出-n・2026-09-13） ──


def test_the_checker_only_gets_rules_it_can_judge_from_the_text_and_the_facts():
    """2 日間の違反 31 件のうち、`declare-memory-use` 絡みが 43 文（複数出るので超える）。
    申告（`say` の引数）はチェッカーに渡っておらず、文だけを見て「申告が欠けている」と
    言っていた。判定できないものを判定させない。"""
    from familiar_agent.loop.prompt import CHECKER_RULE_IDS, rules_for_checker

    text = rules_for_checker()
    for rid in (
        "no-fake-perception",
        "no-invented-knowledge",
        "no-past-comparison-without-memory",
        "memory-evidence-confidence",
        "workspace-is-notes-not-script",
        "no-raw-internal-metrics",
        "no-tts-tags",
    ):
        assert f":id {rid}\n" in text, rid
        assert rid in CHECKER_RULE_IDS
    for rid in (
        "declare-memory-use",  # 申告は `say` の引数で、文には無い（機械が数えている）
        "voice-only-from-say",  # `say` を呼んだかは機械が知っている
        "first-person-perspective-taking",
        "validation-before-advice",
        "bid-for-connection",
        "personality-from-me",
        "language-match",
    ):
        assert f":id {rid}\n" not in text, rid
    # 正本（主LLM へ渡る規則）は変わらない。
    from familiar_agent.loop.prompt import rules_section

    assert ":id declare-memory-use\n" in rules_section()


def test_the_checker_still_drops_tts_tags_when_the_voice_understands_them():
    from familiar_agent.loop.prompt import rules_for_checker

    assert ":id no-tts-tags\n" not in rules_for_checker(allow_tts_tags=True)


def test_the_agent_hands_the_checker_rules_to_the_instrument_stance():
    """`with_rules=True` の呼び手はチェッカーだけ。そこへ渡るのは絞った規則である。"""
    from unittest.mock import MagicMock

    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.core.context_parts import Stance

    a = MagicMock()
    a._me_md, a._family_md, a._tts = "", "", None
    system = EmbodiedAgent._stance_context(a, Stance.INSTRUMENT, with_rules=True)
    assert system is not None
    assert ":id no-invented-knowledge\n" in system
    assert ":id declare-memory-use\n" not in system


# ── 材料を切らない・申告に頼らない（出-n・2026-09-13） ─────────────────────────


def test_the_arrived_results_are_facts_whether_or_not_they_were_declared():
    """この求めで届いた結果（版・完了 O）は、主LLM が申告しなくても事実として渡る。

    15:55 実機：検索結果に「雨のち曇 · 最高 · 25 ℃」があるのに「事実に含まれていない」と
    差し戻された。申告した記憶の中身は 200 字で切られ、そこから先が見えなかった。
    """
    from familiar_agent.loop.speech_check import facts_ctx

    long = (
        "「明日の天気は？」と聞かれ、1番：search_deferred の結果が届いた："
        + "x" * 300
        + "雨のち曇 · 最高 · 25 ℃"
    )
    out = facts_ctx(saw=False, memories=[], arrived=[long])
    assert "この求めで届いた結果" in out
    assert "最高 · 25 ℃" in out, "届いた結果が切られている"


def test_the_declared_memories_are_not_cut_at_200_chars():
    from familiar_agent.loop.speech_check import facts_ctx

    body = "a" * 480 + "末尾の事実"
    out = facts_ctx(saw=False, memories=[], used=[body])
    assert "末尾の事実" in out


def test_a_very_long_arrived_result_is_capped_but_generously():
    from familiar_agent.loop.speech_check import ARRIVED_CHARS, facts_ctx

    out = facts_ctx(saw=False, memories=[], arrived=["y" * (ARRIVED_CHARS + 500)])
    assert out.count("y") == ARRIVED_CHARS and ARRIVED_CHARS >= 2000


def test_the_loop_hands_the_arrived_results_to_the_checker_without_a_declaration():
    """ループは open な記録（この求めの版・完了）の中身を `arrived` として渡す。"""
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.loop.request import Request

    ip = InformationProcessing.__new__(InformationProcessing)
    ip._req = Request()
    ip._req.request_id = "req-1"
    ip._req.live_version_id = "ver-2"
    ip._req.turn_records = []
    ip._agent = MagicMock()
    ip._agent.config.speech_check = True
    ip._seen_image = MagicMock(return_value=None)
    arrived = _rec("ver-2", content="1番：search_deferred の結果が届いた：雨のち曇 · 最高 · 25 ℃")
    old = _rec("m1", content="昔の話")
    facts = ip._checker_facts([arrived, old], verdicts=None, w_id_map={})
    assert "この求めで届いた結果" in facts and "最高 · 25 ℃" in facts
    assert "昔の話" not in facts  # 申告されていない過去の記憶は根拠にならない
