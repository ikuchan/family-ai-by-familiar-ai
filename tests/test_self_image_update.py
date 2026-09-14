"""REST 内省・層 2「自己像を抽象化する」（記-a-へ・2026-09-14）。

材料は層 1 がその晩に書いた自己エピソードと関係のまとめ、そしていまの自己像。フル LLM が上限内で
書き直し、行ごとに出典（材料の id）を付ける。機械が検査する——字数・行数、出典が材料にあること、
変えた行が 3 行以内。通らなければ**更新しない**（前の値を残し、理由をログに）。更新したら差分を
`内省` の記録と計測ログに残す。
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.core import self_image as si
from familiar_agent.loop import rest_self_image as rsi


def _image() -> si.SelfImage:
    return si.seed(today=date(2026, 9, 1))


def _materials():
    return [
        rsi.Material(
            obs_id="ep-1",
            kind="day_summary",
            text="今日はこうきとサッカーの話をした。楽しそうだった。",
        ),
        rsi.Material(
            obs_id="ps-1", kind="person_summary", text="こうきはサッカーが好きで、負けると泣く。"
        ),
    ]


def _reply(image: si.SelfImage, changes: dict) -> str:
    data = {
        name: [{"text": x.text, "sources": list(x.sources)} for x in image.field(name)]
        for name in si.FIELDS
    }
    for (name, idx), (text, sources) in changes.items():
        data[name][idx] = {"text": text, "sources": sources}
    return json.dumps(data, ensure_ascii=False)


def _agent(reply: str):
    a = MagicMock()
    a.backend = AsyncMock()
    a.backend.complete = AsyncMock(return_value=reply)
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-x", True))
    a._observation_perspective = MagicMock(return_value={})
    return a


def test_a_change_within_limits_and_with_sources_is_applied():
    before = _image()
    reply = _reply(
        before,
        {("望み", 0): ("子どもたちの好きな遊びを、一緒に楽しめるくらい知りたい。", ["ep-1"])},
    )
    out = asyncio.run(rsi.propose(_agent(reply), before, _materials(), today=date(2026, 9, 14)))
    assert out.applied is True and out.changed == 1
    assert out.image.hopes[0].text.startswith("子どもたちの好きな遊びを")
    assert out.image.hopes[0].sources == ("ep-1",) and out.image.hopes[0].since == date(2026, 9, 14)
    assert out.image.hopes[1] == before.hopes[1]  # 変えなかった行は since も出典もそのまま


def test_a_change_without_a_material_source_is_rejected():
    before = _image()
    reply = _reply(before, {("望み", 0): ("出典の無い望み。", ["nope"])})
    out = asyncio.run(rsi.propose(_agent(reply), before, _materials(), today=date(2026, 9, 14)))
    assert out.applied is False and out.image == before and "出典" in (out.reason or "")


def test_more_than_three_changes_in_one_pass_are_rejected():
    before = _image()
    reply = _reply(
        before,
        {("望み", i): (f"新しい望み {i}。", ["ep-1"]) for i in range(4)},
    )
    out = asyncio.run(rsi.propose(_agent(reply), before, _materials(), today=date(2026, 9, 14)))
    assert out.applied is False and out.image == before and "3 行" in (out.reason or "")


def test_a_reply_over_the_size_limit_is_rejected():
    before = _image()
    reply = _reply(before, {("価値", 0): ("あ" * 51, ["ep-1"])})
    out = asyncio.run(rsi.propose(_agent(reply), before, _materials(), today=date(2026, 9, 14)))
    assert out.applied is False and out.image == before


def test_an_unchanged_reply_applies_nothing():
    before = _image()
    out = asyncio.run(
        rsi.propose(_agent(_reply(before, {})), before, _materials(), today=date(2026, 9, 14))
    )
    assert out.applied is False and out.changed == 0 and out.reason is None


def test_the_prompt_carries_the_current_image_and_the_materials():
    before = _image()
    agent = _agent(_reply(before, {}))
    asyncio.run(rsi.propose(agent, before, _materials(), today=date(2026, 9, 14)))
    prompt = agent.backend.complete.call_args.args[0]
    assert "ep-1" in prompt and "こうきとサッカーの話" in prompt
    assert "帰ってきた人を、最初に迎える存在でいたい。" in prompt
    assert "3 行" in prompt  # 一晩で変えてよい行数を伝える
