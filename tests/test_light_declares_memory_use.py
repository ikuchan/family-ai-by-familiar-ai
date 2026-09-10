"""**軽量LLM** も、想起した記憶をどう扱ったかを申告する（出-h-ろ）。

記憶を見て答える口は2つある——**主LLM**（`say`）と**軽量LLM**（調停の `light`）である。
申告の口を持っていたのは主LLM だけで、**軽量LLM が答えて閉じた反復では記憶が何も動かな
かった**。記憶が育つ経路は申告1本しかないのに、その1本を通らない道があった。

**申告は調停とは別に聞く**（案ホ）。調停の JSON へ足すと、実測で `light` を選ぶ側へ判断が
寄った（light 6/24 → 11/24）。切り離せば調停のプロンプトは一字も変わらないので分岐は動かない。

**答えたあとの後片付けなので、背景で走らせる**（実測 1.03 秒）。ただし打ち切りでは消さない
——申告は「実際にその記憶を使った」という事実で、あとから古くならない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import workspace


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.loop.request import Request

    ip = object.__new__(InformationProcessing)
    ip._agent = MagicMock()
    ip._req = Request()
    ip._verdict_tasks = set()
    return ip


def test_the_light_llm_is_asked_how_it_used_the_memories():
    """申告の口は軽量LLM へ1回だけ聞き、W に並んだ id で返させる。"""
    backend = MagicMock()
    backend.complete = AsyncMock(
        return_value='{"memory_verdicts": [{"id": "abcdef123456", "verdict": "referred"}]}'
    )
    raw = asyncio.run(
        workspace.ask_verdicts(
            backend,
            utterance="海って楽しいよね",
            reply="楽しいですよね。",
            workspace_ctx="- 2025-08-03 12:00 id:abcdef123456 (発話): 海で泳げるようになった",
        )
    )
    assert raw == [{"id": "abcdef123456", "verdict": "referred"}]
    prompt = backend.complete.call_args.args[0]
    # 4つの判定に**別々の引き金**を与える（主LLM で効いた形と同じ）。
    for verdict in ("important", "referred", "useless", "unused"):
        assert f"`{verdict}`" in prompt
    assert "海って楽しいよね" in prompt and "楽しいですよね。" in prompt


def test_a_broken_reply_declares_nothing():
    """JSON にならない返事でも落ちない。申告が無いだけで、発話には関わらない。"""
    backend = MagicMock()
    backend.complete = AsyncMock(return_value="ごめん、わからない")
    assert (
        asyncio.run(workspace.ask_verdicts(backend, utterance="", reply="", workspace_ctx="")) == []
    )


def test_closing_with_the_light_llm_asks_for_the_verdicts_in_the_background():
    """`light` で閉じた反復は、申告を**背景で**投げる。発話を待たせない。"""
    ip = _ip()
    mem = MagicMock()
    ip._agent._utility_backend.complete = AsyncMock(
        return_value='{"memory_verdicts": [{"id": "abcdef123456", "verdict": "important"}]}'
    )

    async def scenario():
        ip._declare_light_memory_use(
            utterance="運動会の話、覚えてる？",
            reply="覚えてますよ。",
            workspace_ctx="- id:abcdef123456",
            w_id_map={"abcdef123456": "m1"},
            mem=mem,
        )
        assert len(ip._verdict_tasks) == 1  # 立った時点では待っていない
        await asyncio.gather(*list(ip._verdict_tasks))

    asyncio.run(scenario())
    mem.apply_verdicts.assert_called_once_with({"m1": "important"})


def test_an_aborted_request_does_not_cancel_the_verdicts():
    """**打ち切りで申告を消さない。** 主LLM の返りは古くなるが、申告は事実である。"""
    ip = _ip()
    ip._background_tasks = set()
    ip._completion_queue = asyncio.Queue()
    ip._drained_completions = []
    ip._request_generation = 0

    async def scenario():
        started = asyncio.Event()
        done = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(0.05)
            done.set()

        ip._verdict_tasks.add(t := asyncio.create_task(slow()))
        t.add_done_callback(ip._verdict_tasks.discard)
        await started.wait()
        await ip._abort_lookups()
        await asyncio.sleep(0.1)
        assert done.is_set(), "打ち切りで申告のタスクが消えた"

    asyncio.run(scenario())
