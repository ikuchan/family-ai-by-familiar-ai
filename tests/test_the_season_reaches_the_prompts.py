"""いまの季節とまわりを、主LLM と調停の両方へ毎ターン渡す（知-ac 段 4・2026-09-26）。

置き場は自己像と同じ形——システム文の**安定部**、`[いまの自分]` の直後。書き換わるのは季節の層が
回った晩だけなので、キャッシュの効き方は自己像と同じ（`設計方針_季節の層` v0.1 §2）。
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import date
from unittest.mock import MagicMock

from familiar_agent.core import season_env as se
from familiar_agent.core.context_parts import Stance, build_context

SELF = "[いまの自分]\n- 望み：家族と話す"
SEASON = "[いまの季節とまわり]\n- 暦：秋分（9/23）を過ぎたころ。次は寒露（10/8）。"


def test_the_season_comes_right_after_the_self_image_in_the_stable_part():
    ctx = build_context(
        stance=Stance.PAJU,
        self_understanding="パジュ",
        family="パパ",
        self_image=SELF,
        season_env=SEASON,
        now='(now :datetime "2026-09-26")',
    )
    assert SEASON in ctx.stable and SEASON not in ctx.variable
    assert ctx.stable.index(SELF) < ctx.stable.index(SEASON)


def test_the_main_llm_prompt_carries_it():
    from familiar_agent.loop.prompt import build_event_system_prompt

    stable, _ = build_event_system_prompt(
        self_understanding="パジュ",
        family_md="パパ",
        present_ctx="",
        pi_ctx="",
        workspace_ctx="",
        season_env=SEASON,
    )
    assert SEASON in stable


def test_the_arbiter_is_told_it_too():
    from familiar_agent.loop import arbiter

    seen: dict = {}

    async def complete(prompt, max_tokens, **kw):
        seen["system"] = kw.get("system") or ""
        return '{"branch":"light","text":"寒くなってきたね"}'

    b = MagicMock()
    b.complete = complete
    asyncio.run(
        arbiter.arbitrate(
            b,
            utterance="寒いね",
            workspace_ctx="",
            self_understanding="パジュ",
            family_md="パパ",
            season_env=SEASON,
        )
    )
    assert SEASON in seen["system"]


def test_the_loop_renders_what_is_stored(monkeypatch):
    from familiar_agent.loop import event_loop

    env = se.SeasonEnv(
        written_on=date.today(), rows={"まわり": (se.Row("金木犀が咲き始めた。", "q"),)}
    )
    monkeypatch.setattr(se, "stored", lambda: env)
    text = event_loop._season_env_text()
    assert text.startswith("[いまの季節とまわり]") and "金木犀が咲き始めた。" in text


def test_both_calls_from_the_loop_pass_it():
    """主LLM（`build_event_system_prompt`）と調停（`arbitrate`）の両方に渡す。"""
    from familiar_agent.loop import event_loop

    src = inspect.getsource(event_loop)
    assert src.count("season_env=_season_env_text()") == 2
