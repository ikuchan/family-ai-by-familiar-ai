"""申告は想起と同じ面へ当てる（環-ab・2026-09-28）。

想起は `current_speaker_id or AGENT_SELF_ID` の面から引く。申告は `_active_memory()` と同じ選び方
（話者が「分かっている」ときだけその人の面）で、話者の指定が切れてから T の刻みが話者を戻すまでの
あいだ、想起は話者の面・申告はパジュの面とずれていた。`situated_memories` は人ごとなので、ずれると
申告は 0 行に当たる（出-h-ろ ③）。記憶の口（OIF）を通したとき（R-6）に見つけた。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.loop.event_loop import InformationProcessing
from tests._arbiter_fakes import jev_says, writer_says
from tests.test_event_loop import _agent


def test_the_declaration_uses_the_face_the_recall_used():
    a = _agent(stream_returns=[])
    a._pmm.current_speaker_id = "p1"  # 話者は設定されている
    a.speaker_known = MagicMock(return_value=False)  # が、もう「分かっている」とはみなさない
    a._jev = jev_says("light")
    a._utility_backend = writer_says({"text": "うん"})
    ip = InformationProcessing(a)
    seen: dict = {}
    ip._declare_light_memory_use = lambda **kw: seen.update(kw)  # type: ignore[method-assign]

    async def scenario():
        await ip.push_utterance("ねえ")
        for _ in range(200):
            if seen:
                break
            await asyncio.sleep(0.005)
        await ip.close()

    asyncio.run(scenario())
    assert seen["verdict_view"] == "p1"  # 想起と同じ面
