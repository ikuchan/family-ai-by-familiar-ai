"""情動で起きた求めは「自分がしたくなったこと」として渡す（情-e・2026-09-13）。

自発なので、誰かの許可は要らず、動作に理由も要らない。いまは内声が「理由まで結論づけて」
「その結果を踏まえて話す」と要求し、調停には `[人の言葉]` の下に、主LLM には人の発言と同じ
形で渡っていた（実験：調停のつなぎは 8/8 が相手への断り、主LLM は 4/8 が理由を作った）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.config import DriveConfig
from familiar_agent.loop import reply_budget
from familiar_agent.loop.arbiter import Decision as ArbiterDecision, arbitrate
from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.loop.prompt import build_event_system_prompt

from tests.test_event_loop import _agent


def test_the_inner_voices_ask_for_no_reason_and_no_listener() -> None:
    cfg = DriveConfig()
    for axis in ("seeking", "rest", "bond", "safety", "esteem"):
        voice = getattr(cfg, f"voice_{axis}")
        assert "理由" not in voice, axis
        assert "話す" not in voice and "働きかける" not in voice, axis
        assert "湧いている" in voice or "休みたい" in voice, axis


def _light():
    b = MagicMock(spec=["complete"])
    b.complete = AsyncMock(return_value='{"branch": "action", "action": "see"}')
    return b


def test_the_arbiter_gets_a_self_doing_frame_for_a_drive_request() -> None:
    b = _light()
    asyncio.run(
        arbitrate(b, utterance="探索したい気持ちが湧いている。", workspace_ctx="", origin="情動")
    )
    prompt = b.complete.call_args.args[0]
    assert "[いま湧いたこと]" in prompt and "[人の言葉]" not in prompt
    assert "許可は要らない" in prompt and "理由も要らない" in prompt
    assert "待ってもらうための短い一言" not in prompt, "つなぎは書かせない"


def test_a_human_request_keeps_the_reply_frame() -> None:
    b = _light()
    asyncio.run(arbitrate(b, utterance="おはよう", workspace_ctx="", origin="発話"))
    prompt = b.complete.call_args.args[0]
    assert "[人の言葉]" in prompt and "[いま湧いたこと]" not in prompt


def test_a_drive_decision_never_carries_a_filler() -> None:
    from familiar_agent.loop.arbiter import _parse

    d = _parse(
        '{"branch": "action", "action": "see", "text": "見てみますね"}', can_see=True, origin="情動"
    )
    assert d is not None and d.text == ""


def test_the_budget_line_becomes_a_soliloquy_for_a_drive_request() -> None:
    b = reply_budget.decide(effort="low", researched=False, w_count=3, origin="情動")
    assert b.line().startswith("[独り言]") and "誰にも向けない" in b.line()
    assert (b.target, b.limit) == (20, 40)
    assert (
        reply_budget.decide(effort="low", researched=False, w_count=3).line().startswith("[返事]")
    )


def test_the_main_llm_system_drops_the_answering_rule_for_a_drive_request() -> None:
    stable, _ = build_event_system_prompt(
        self_understanding="me",
        family_md="fam",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="",
        origin="情動",
    )
    assert "first-person-perspective-taking" not in stable
    stable2, _ = build_event_system_prompt(
        self_understanding="me",
        family_md="fam",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="",
    )
    assert "first-person-perspective-taking" in stable2


def test_the_loop_passes_the_origin_and_frames_the_user_message() -> None:
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = "情動"
    ip._req.cue = "[内的な促し:seeking] 探索したい気持ちが湧いている。"
    content = ip._user_content("", [])
    assert isinstance(content, str) and content.startswith("[いま湧いたこと]")
    assert "許可は要らない" in content

    async def scenario():
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full")),
        ) as arb:
            await ip._decide(
                utterance="", workspace_ctx="", present_ctx="", capped=False, round_=1, memories=[]
            )
        await ip.close()
        return arb.call_args.kwargs.get("origin")

    assert asyncio.run(scenario()) == "情動"
