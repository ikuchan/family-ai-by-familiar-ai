"""調停は直近のやりとりを見て判断する（2026-09-13 実機で露見）。

「明日の天気は？」→（検索が空振り）→「調べて」で、調停は直近のやりとりを渡されておらず、
「調べて」を新しい検索（今日のニュース）にした。直近は W の先頭の枠として、主LLM と
**同じ作り方**で調停にも渡る。違うのは窓の幅（`recent_exchanges_arbiter` ＜
`recent_exchanges_main`）だけである（記-h）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from familiar_agent.backends import ToolCall
from familiar_agent.loop.arbiter import Decision as ArbiterDecision
from familiar_agent.loop.event_loop import InformationProcessing
from tests.test_event_loop import _agent, _turn


def _exchange(i: int, ask: str, ans: str) -> list[dict]:
    # 直近の窓には 5 分の上限がある（出-ae(2)）ので、いまから数十秒前に置く。収集時に固めると
    # 全体テストの後半で実行されたとき 5 分を越えるので、呼ぶたびに取る。
    when = datetime.now(timezone.utc) - timedelta(seconds=60 - 10 * i)
    return [
        {"obs_id": f"q{i}", "content": ask, "role": "起点", "timestamp": when, "depth": 0},
        {"obs_id": f"a{i}", "content": ans, "role": "答え", "timestamp": when, "depth": 0},
    ]


def _agent_with_four_exchanges():
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "うん"})])])
    chains = {
        "q1": _exchange(1, "おはよう", "おはよう！"),
        "q2": _exchange(2, "今日は暑いね", "ほんとだね"),
        "q3": _exchange(3, "明日の天気は？", "気象庁のサイトは見つかったんですが"),
        "q4": _exchange(4, "調べて", "はい"),
    }
    a._memory.latest_exchange_origins = MagicMock(
        side_effect=lambda n: ["q4", "q3", "q2", "q1"][:n]
    )
    a._memory.recent_exchanges = MagicMock(side_effect=lambda o: chains[o])
    return a


def test_the_arbiter_and_the_main_llm_get_the_same_frame_with_different_widths(monkeypatch):
    from familiar_agent import config_overrides as co

    # 窓は層 3 の設定値（DB > 既定・env は読まない・記-a-に）。
    co._delete_all()
    assert co.save_override("MemoryConfig.recent_exchanges_arbiter", 2)
    assert co.save_override("MemoryConfig.recent_exchanges_main", 3)
    co.clear_cache()
    a = _agent_with_four_exchanges()
    ip = InformationProcessing(a)
    seen: dict[str, str] = {}

    async def scenario():
        with patch(
            "familiar_agent.loop.event_loop.arbitrate",
            new=AsyncMock(return_value=ArbiterDecision(branch="full")),
        ) as arb:
            await ip.push_utterance("もう一回")
            for _ in range(200):
                if a.backend.stream_turn.await_count >= 1:
                    break
                await asyncio.sleep(0.005)
        seen["arbiter"] = arb.call_args.kwargs["workspace_ctx"]
        seen["main"] = "\n".join(a.backend.stream_turn.call_args.kwargs["system"])
        await ip.close()

    asyncio.run(scenario())
    arb, main = seen["arbiter"], seen["main"]
    # 調停：最新 2 往復。主LLM：最新 3 往復。どちらも同じ枠の見出しと同じ行の形。
    assert "直近のやりとり" in arb and "直近のやりとり" in main
    assert "調べて" in arb and "明日の天気は？" in arb and "今日は暑いね" not in arb
    assert "調べて" in main and "今日は暑いね" in main and "おはよう" not in main
    arb_line = next(line for line in arb.splitlines() if "明日の天気は？" in line)
    assert arb_line in main, "同じ記録が、調停と主LLM で違う形で印字されている"
    co._delete_all()
    co.clear_cache()
