"""主LLM を投げっぱなしにする（環-h・段は-1）。

主LLM は 出-c で資源になったのに、**調べものと違って同期で居残っていた**。投げっぱなしに
すれば、返りは完了キュー（QC）を通って**別の反復**で実行される。

反復は2種類になる。

| 反復 | 中身 | 軽量LLM |
|---|---|---|
| **決める反復** | 取込 → 想起 → 調停 → 出力（発話／調べもの投げ／**主LLM 投げ**） | 回す |
| **出す反復** | 取込 → **主LLM の決定をそのまま実行** | 回さない |

**投げたときの W を持ち越す**（`Decision`）。共起は「その反復で一緒に活性した記録」なので
主LLM が見た W でなければ意味がなく、申告は W に印字された12桁で返るので、その対応表でないと
引けない。飛行中に別の完了が届けば求めの寿命の状態は上書きされるので、**返りと一緒に運ぶ**。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.backends.types import TurnResult
from familiar_agent.loop.event_loop import Trigger, Decision, InformationProcessing
from familiar_agent.loop.request import Request


def _ip():
    ip = InformationProcessing.__new__(InformationProcessing)
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._triggers = asyncio.Queue()
    ip._drained_completions = []
    ip._req.lookups = []
    ip._background_tasks = set()
    ip._request_generation = 0
    ip._asyncio_loop = None
    ip._req.cue = "手がかり"
    ip._req.request_id = "req-1"
    ip._write_version = AsyncMock(return_value="ver-1")
    a = MagicMock()
    a.config.max_tokens = 1000
    ip._agent = a
    ip._tools = MagicMock(return_value=[])
    return ip, a


def _decision(**kw) -> Decision:
    base = dict(
        result=TurnResult(stop_reason="tool_use", text="", tool_calls=[]),
        memories=[{"memory_id": "m1"}],
        w_id_map={"abcdef123456": "m1"},
        mem=MagicMock(),
        recent_ctx="",
        system=("安定", "可変"),
        effort="high",
        capped=False,
    )
    base.update(kw)
    return Decision(**base)


# ── 投げる ─────────────────────────────────────────────────────────────────


def test_dispatching_counts_the_main_llm_as_in_flight():
    """飛行中に数える。数えないと、駆動体が「調査中ではない」とみなして待ち方を変える。"""
    ip, a = _ip()
    a.backend.stream_turn = AsyncMock(return_value=(TurnResult("end_turn", ""), None))

    async def scenario():
        ip._dispatch_main_llm(
            messages=[],
            system=("安定", "可変"),
            effort="high",
            capped=False,
            memories=[],
            w_id_map={},
            mem=MagicMock(),
            recent_ctx="",
        )
        got = ip._in_flight_count
        for t in list(ip._background_tasks):
            t.cancel()
        return got

    assert asyncio.run(scenario()) == 1


def test_the_dispatched_main_llm_appears_in_the_lookups():
    ip, a = _ip()
    a.backend.stream_turn = AsyncMock(return_value=(TurnResult("end_turn", ""), None))

    async def scenario():
        ip._dispatch_main_llm(
            messages=[],
            system=("安定", "可変"),
            effort="high",
            capped=False,
            memories=[],
            w_id_map={},
            mem=MagicMock(),
            recent_ctx="",
        )
        for t in list(ip._background_tasks):
            t.cancel()
        return [(lk.action, lk.in_flight) for lk in ip._req.lookups]

    assert asyncio.run(scenario()) == [("主LLM", True)]


def test_the_return_lands_in_the_queue_with_what_it_saw():
    """返りは、投げたときの W と一緒に QC へ積まれる。"""
    ip, a = _ip()
    tr = TurnResult("tool_use", "", [ToolCall("t", "say", {"text": "はい"})])
    a.backend.stream_turn = AsyncMock(return_value=(tr, None))

    async def scenario():
        ip._dispatch_main_llm(
            messages=[],
            system=("安定", "可変"),
            effort="low",
            capped=True,
            memories=[{"memory_id": "m1"}],
            w_id_map={"abcdef123456": "m1"},
            mem=MagicMock(),
            recent_ctx="R",
        )
        for t in list(ip._background_tasks):
            await t
        return ip._triggers.get_nowait()

    c = asyncio.run(scenario())
    assert c.kind == "決定"
    d = c.decision
    assert d.result is tr
    assert d.memories == [{"memory_id": "m1"}]
    assert d.w_id_map == {"abcdef123456": "m1"}
    assert (d.recent_ctx, d.effort, d.capped) == ("R", "low", True)


# ── 取り込む ───────────────────────────────────────────────────────────────


def test_intake_hands_the_decision_back():
    ip, _a = _ip()
    ip._req.lookups = []
    ip._drained_completions = [Trigger(kind="決定", decision=_decision())]
    drained, decided = asyncio.run(ip._intake())
    assert decided is not None
    assert decided.result.stop_reason == "tool_use"


def test_the_version_does_not_carry_the_返り():
    """版には結果を載せない（`see` と同じ。中身は `_finish` の「自分が答えた」が持つ）。"""
    from familiar_agent.loop.event_loop import Lookup

    ip, _a = _ip()
    ip._req.lookups = [Lookup(index=1, action="主LLM", query="主LLM1", generation=0)]
    ip._drained_completions = [
        Trigger(
            kind="決定",
            query="主LLM1",
            decision=_decision(
                result=TurnResult("tool_use", "", [ToolCall("t", "say", {"text": "秘密の答え"})])
            ),
        )
    ]
    asyncio.run(ip._intake())
    assert "秘密の答え" not in (ip._req.lookups[0].result or "")
    assert ip._req.lookups[0].in_flight is False  # 飛行中ではなくなる


# ── 出す反復 ───────────────────────────────────────────────────────────────


def test_the_iteration_acts_on_a_decision_without_arbitrating():
    """出す反復では調停を回さない。回すと主LLM の決定を覆せてしまう。"""
    import inspect

    src = inspect.getsource(InformationProcessing._iterate)
    i_act = src.index("_act_on_decision(")
    i_arb = src.index("arbitrate(")
    assert i_act < i_arb, "調停より前に決定を実行していない"


# ── 持ち越しの守り ─────────────────────────────────────────────────────────


def test_the_verdicts_use_the_map_that_the_main_llm_saw():
    """**飛行中に別の完了が届いても、申告が正しい記憶へ当たる。**

    主LLM が返るまでに、W も対応表も次の反復のもので作り直される。12桁が当たってしまえば、
    申告が黙って別の記憶へ適用される。だから**返りと一緒に運ぶ**。

    に-5-に-2 で対応表は属性でなくなったので、`Decision` が唯一の持ち場である。
    """
    ip, a = _ip()
    ip._speak = AsyncMock(return_value=("はい", "発話"))
    ip._finish = AsyncMock()
    ip._coherence_violation = AsyncMock(return_value=None)
    seen_mem = MagicMock()  # 主LLM が見た W を作った面（出-h-ろ ③）

    said = ToolCall(
        "t",
        "say",
        {"text": "はい", "memory_verdicts": [{"id": "abcdef123456", "verdict": "important"}]},
    )
    asyncio.run(
        ip._act_on_decision(
            _decision(
                result=TurnResult("tool_use", "", [said]),
                memories=[],
                w_id_map={"abcdef123456": "主LLM が見た記憶"},  # 持ち越した表
                mem=seen_mem,
            ),
            utterance="こんばんは",
            gen=0,
        )
    )
    seen_mem.apply_verdicts.assert_called_once_with({"主LLM が見た記憶": "important"})
    a._memory.apply_verdicts.assert_not_called()  # 基底の面へは行かない
