"""名前の無い入力は、途中の求めに添える（出-au 段 3・2026-09-27・`設計方針_判定の段` §2.4）。

段 1 では、名前の無い入力（「あ、東京のね」）は飛行中のものがある間は待たせ、前の答えが出てから別の求めに
していた。前の求めの材料にはならないので、「東京の」が答えに入らない。

- **調べもの中**（主LLM は飛んでいない）：入力を O へ書き（役割「添え」）、求めの「途中で言われたこと」に足し、
  画面へはすぐ返す。結果が届いた後の決める反復で W に載る。
- **主LLM が考えている最中**：いまどおり待たせる。返りの時点で「考え直すか、そのまま出すか」を判定する口を
  通す。判定（Jev）は段 5 で入れるので、いまは既定の「そのまま出す」。
- 直近のやりとりは「添え」も引く（人の言葉として次のターンにも見える）。起点にはしない（やりとりの境目が崩れる）。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock

from familiar_agent.loop import workspace
from familiar_agent.loop.event_loop import InformationProcessing, Trigger
from familiar_agent.loop.request import Lookup, Request

from tests.test_event_loop import _agent


def _ip(action: str):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.request_id = "req-1"
    ip._req.trigger_kind = "発話"
    ip._req.lookups = [Lookup(index=1, action=action, query="q", generation=0)]
    return ip, a


def _unnamed(text: str):
    async def make():
        fut = asyncio.get_running_loop().create_future()
        return Trigger(kind="会話入力", query=text, source="voice", named=False, future=fut)

    return make


def test_words_during_a_lookup_join_the_request():
    ip, a = _ip("search_deferred")

    async def go():
        trig = await _unnamed("あ、東京のね")()
        ip._triggers.put_nowait(trig)
        got = await asyncio.wait_for(ip._take_trigger(), timeout=1.0)
        assert got is trig
        await ip._attach_to_request(got)
        return trig

    trig = asyncio.run(go())
    assert trig.future.result() == ""  # 画面はすぐ返る
    assert ip._req.added == [("obs1", "あ、東京のね")]
    assert ip._held == []
    assert ("obs1", "添え") in ip._req.turn_records
    assert ip._req.request_id == "req-1"  # 前の求めのまま


def test_the_driver_attaches_instead_of_starting_a_request():
    src = inspect.getsource(InformationProcessing._drive)
    assert "_attach_to_request" in src


def test_words_during_the_main_llm_still_wait():
    ip, _ = _ip("主LLM")

    async def go():
        trig = await _unnamed("あ、東京のね")()
        ip._triggers.put_nowait(trig)
        ip._triggers.put_nowait(Trigger(kind="完了", query="主LLM1", result="{}"))
        return await asyncio.wait_for(ip._take_trigger(), timeout=1.0)

    assert asyncio.run(go()) is None  # 先に主LLM の返りを取り込む
    assert [t.query for t in ip._held] == ["あ、東京のね"]


def test_the_added_words_are_shown_in_the_workspace():
    req = Request()
    req.added = [("obs9", "あ、東京のね")]
    oif = AsyncMock()
    oif.actors = lambda ids: {}
    oif.roles = lambda ids: {}
    text, _ = workspace.compose(oif, [], req)
    assert "あ、東京のね" in text and "言い足したこと" in text


def test_the_main_llm_return_asks_whether_to_rethink():
    """判定の口は段 5-1 で Jev に替えた（`test_rethink_with_jev.py`）。Jev が使えなければ「そのまま出す」。"""
    ip, a = _ip("主LLM")
    a._jev = None
    ip._held.append(Trigger(kind="会話入力", query="あ、東京のね", named=False))
    assert asyncio.run(ip._rethink_or_speak("晴れだよ")) == "そのまま出す"
    assert "_rethink_or_speak" in inspect.getsource(InformationProcessing._act_on_decision)


def test_recent_exchanges_include_added_words():
    from familiar_agent.store.relations import RelationStore

    sig = inspect.signature(RelationStore.recent_exchanges)
    assert "添え" in sig.parameters["roles"].default


def test_a_new_request_starts_with_nothing_added():
    ip, _ = _ip("search_deferred")
    ip._req.added = [("obs1", "x")]

    async def go():
        await ip._begin_request(kind="発話", text="パジュ、別の話", utterance="パジュ、別の話")

    asyncio.run(go())
    assert ip._req.added == []


def test_added_words_are_the_other_persons_words():
    """話者が分からないと面は自分に寄る。「添え」も起点と同じく相手の言葉として読む。"""
    from datetime import datetime

    from familiar_agent.io.oif import MI, Recalled

    req = Request()
    req.turn_records = [("obs5", "添え")]
    oif = AsyncMock()
    oif.actors = lambda ids: {}
    oif.roles = lambda ids: {}
    rec = Recalled(
        mi=MI(
            id="obs5",
            obs_id="obs5",
            content="あ、東京のね",
            timestamp=datetime(2026, 9, 27, 9, 0),
            direction="発話",
        ),
        fit=0.5,
        groundedness=1.0,
        confidence=0.8,
    )
    text, _ = workspace.compose(oif, [rec], req)
    assert "相手" in text
