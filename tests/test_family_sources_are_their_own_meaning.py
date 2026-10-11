"""家族の予定や家のきまりの問いは、意味「家族に関わる情報源を見に行く問い」で読む（知-ax・2026-10-11・本人の決定ア）。

10/11 08:24「パジュ、今日の予定は?」を Jev は「その他の会話」0.42 と読み、聞き返しに倒れた。「調べる必要のある問い」の説明は
世の中のことと記憶の話で、家族の情報源を見に行く受け皿が無かった。意味を足し、予定表と Notion の説明を書き分けた形で
測ってから入れた（予定の 4 場面中 3 つが 0.76〜0.93 でこの意味に読まれた）。
"""

from __future__ import annotations

import asyncio

from familiar_agent.core import utterance_meaning as um
from tests._arbiter_fakes import decide, writer_says
from tests.test_arbiter_utterance_by_meaning import _a, _jev


def test_the_family_sources_follow_research():
    keys = list(um.MEANINGS)
    assert keys[keys.index("research") + 1] == "家族の情報源"
    assert um.MEANINGS["家族の情報源"][0] == "家族に関わる情報源を見に行く問い"
    assert "今日の予定は？" in um.MEANINGS["家族の情報源"][1]


def test_the_schedule_and_notion_are_told_apart():
    assert "何時に帰る" in um.ACTIONS["family_schedule"]
    assert "予定は予定表にある" in um.ACTIONS["notion_search"]


def test_the_family_sources_reach_the_schedule():
    jev = _jev(
        {"meaning": _a("家族の情報源", 0.9), "action_家族の情報源": _a("family_schedule", 0.9)}
    )
    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer_says({"tool_input": {}}),
            utterance="パジュ、今日の予定は?",
            origin="発話",
            extra_actions=("family_schedule",),
        )
    )
    assert (d.branch, d.action) == ("action", "family_schedule")
