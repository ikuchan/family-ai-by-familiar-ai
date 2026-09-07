"""角括弧タグを渡してよいかは、合成の担い手が決める。

`[cheerful]` は ElevenLabs の eleven_v3 だけが解する指示である。ローカルの
Style-Bert-VITS2（既定）には意味が無く、そのまま渡せばモデルまで届く。

合成の担い手は `TTS_ENGINE` で既に切り替わっていたが、**渡してよいものが担い手ごとに
分かれていなかった**。整え方・道具の説明・規則の3箇所が、同じ1つの値
（`TTSTool.understands_tags`）を見る形にする。担い手を切り替えたとき、どれかだけが
取り残されないようにするためである。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.tools.tts import TTSTool

_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


def _tool(engine: str) -> TTSTool:
    return TTSTool(api_key="dummy-key", voice_id="v1", output="local", engine=engine)


# ── 担い手の属性 ───────────────────────────────────────────────────────────


def test_sbv2_does_not_understand_bracket_tags():
    assert _tool("sbv2").understands_tags is False


def test_elevenlabs_understands_bracket_tags():
    assert _tool("elevenlabs").understands_tags is True


# ── 整え方 ─────────────────────────────────────────────────────────────────


def test_the_tag_is_dropped_before_the_local_synthesiser():
    """SBV2 へ `[cheerful]` を渡さない。モデルはそれを指示として解さない。"""
    tool = _tool("sbv2")
    with (
        patch.object(tool, "_synth_sbv2", new=AsyncMock(return_value=_WAV)) as synth,
        patch.object(tool, "_play_paths", new=AsyncMock(return_value=["local"])),
    ):
        asyncio.run(tool.say("[cheerful]こんばんは"))
    assert synth.await_args.args[0] == "こんばんは"


@pytest.mark.asyncio
async def test_the_tag_still_reaches_elevenlabs():
    """反証側。解する担い手には今までどおり届く。落としすぎていないことの確認。"""
    tool = _tool("elevenlabs")

    resp = MagicMock()
    resp.status = 200
    resp.read = AsyncMock(return_value=b"fake")
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    sent: list[dict] = []

    def capture_post(url, json=None, headers=None):
        if json:
            sent.append(json)
        return resp

    session = MagicMock()
    session.post = MagicMock(side_effect=capture_post)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=session),
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("os.unlink"),
        patch("tempfile.NamedTemporaryFile") as tmp,
    ):
        f = MagicMock()
        f.__enter__ = MagicMock(return_value=f)
        f.__exit__ = MagicMock(return_value=False)
        f.name = "/tmp/fake.mp3"
        tmp.return_value = f
        await tool.say("[cheerful]こんばんは")

    assert sent, "API を叩いていない"
    assert "[cheerful]" in sent[0]["text"]


def test_the_parenthetical_aside_is_dropped_by_both():
    """丸括弧のト書きはどちらでも落ちる（既存の振る舞いを変えていない）。"""
    for engine in ("sbv2", "elevenlabs"):
        assert _tool(engine)._clean_for_speech("（静かに待つ）こんばんは") == "こんばんは"


# ── 道具の説明 ─────────────────────────────────────────────────────────────


def test_the_say_tool_stops_advertising_tags_to_a_synthesiser_that_ignores_them():
    desc = _tool("sbv2").get_tool_definitions()[0]["input_schema"]["properties"]["text"]
    assert "[cheerful]" not in desc["description"]


def test_the_say_tool_still_advertises_tags_to_elevenlabs():
    desc = _tool("elevenlabs").get_tool_definitions()[0]["input_schema"]["properties"]["text"]
    assert "[cheerful]" in desc["description"]


# ── 規則 ───────────────────────────────────────────────────────────────────


def test_the_rule_forbidding_tags_is_on_by_default():
    """既定は SBV2 なので、禁止が既定でよい。"""
    from familiar_agent.loop.prompt import rules_section

    assert "no-tts-tags" in rules_section()


def test_the_rule_is_lifted_for_a_synthesiser_that_understands_tags():
    from familiar_agent.loop.prompt import rules_section

    sec = rules_section(allow_tts_tags=True)
    assert "no-tts-tags" not in sec
    # 削りすぎていない。前後の規則は残る。
    assert "voice-only-from-say" in sec
    assert "no-fake-perception" in sec
    assert sec.count("(") == sec.count(")")


def test_the_lifted_rule_is_gone_from_the_assembled_prompt():
    from familiar_agent.loop.prompt import build_event_system_prompt

    stable, _ = build_event_system_prompt(
        self_understanding="me",
        family_md="fam",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="",
        allow_tts_tags=True,
    )
    assert "no-tts-tags" not in stable
    assert "voice-only-from-say" in stable


# ── 繋ぎ込み ───────────────────────────────────────────────────────────────


def test_the_turn_asks_the_synthesiser_whether_tags_are_allowed():
    """ループが自分で決めない。担い手に聞く。"""
    import inspect

    from familiar_agent.loop import event_loop

    src = inspect.getsource(event_loop)
    assert "allow_tts_tags=" in src and "understands_tags" in src


def test_the_utility_stance_asks_the_same_synthesiser():
    """軽量LLM の立ち位置も同じ値を見る（規則の正本が2つに割れないため）。"""
    import inspect

    from familiar_agent import agent as agent_mod

    src = inspect.getsource(agent_mod.EmbodiedAgent._stance_context)
    assert "allow_tts_tags=" in src and "understands_tags" in src
