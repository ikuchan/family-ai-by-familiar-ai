"""つなぎを、やりとりの項としては持ち、想起には出さない（段 4）。

**つなぎは声に出している。** 記録に無いと、相手が聞いた会話とパジュが読み返す会話が
食い違う。「調べてみるね」と言ったことを、あとで辿れない。

054 で O から外したのは、想起の候補を汚すからだった。役割が「想起に出さない」を担う形に
なったので（段 2）、**項として持ちながら想起から外す**ことができる。

想起に出さない役割は二つある。`旧`（もう現行ではない）と `つなぎ`（口に出したが、
覚えておく中身がない）である。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from familiar_agent.backends import ToolCall
from familiar_agent.legacy.semantic_layer import LegacySemanticLayer
from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext, viewpoint_of
from familiar_agent.store.observations import ObservationStore
from familiar_agent.store.relations import RelationStore
from familiar_agent.store.situated import SituatedVectors
from tests.test_event_loop import _agent, _turn

_VEC = "[" + ",".join(["0.03125"] * 1024) + "]"


@pytest.fixture
def ctx() -> StoreContext:
    db = get_db()
    return StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)


def test_the_filler_is_written_as_an_observation():
    """054 の書式（`つなぎに言った：…`）で O に書く。"""
    a = _agent(stream_returns=[_turn([ToolCall(id="s", name="say", input={"text": "はい"})])])

    async def scenario():
        from familiar_agent.loop.event_loop import InformationProcessing

        ip = InformationProcessing(a)
        await ip._say_filler("調べてみるね")

    asyncio.run(scenario())

    written = [c.args[0] for c in a._memory.save_async_with_id.call_args_list]
    assert any("つなぎに言った：調べてみるね" in str(t) for t in written), written


def test_the_filler_joins_the_exchange_in_order():
    """つなぎはやりとりの項に入る。一つのターンで何度出ても、その順に並ぶ。"""
    # 発話と動作が一緒に来たら、発話はつなぎとして出る（`event_loop.py:1393`）。
    a = _agent(
        stream_returns=[
            _turn(
                [
                    ToolCall(id="s0", name="say", input={"text": "調べてみるね"}),
                    ToolCall(id="r", name="recall", input={"query": "昨日の天気"}),
                ]
            ),
            _turn([ToolCall(id="s", name="say", input={"text": "晴れてたよ"})]),
        ]
    )
    from tests.test_event_loop import _run_chain

    _run_chain(a)
    _, kwargs = a._run_post_response_pipeline.call_args
    roles = [r for _, r in kwargs["exchange"]]
    assert "つなぎ" in roles, roles
    assert roles.index("つなぎ") < roles.index("答え"), roles


def test_a_filler_does_not_come_back_from_recall(ctx):
    """つなぎは想起に出ない。出ると、中身の無い一言が候補を食う（054 の理由）。"""
    store = ObservationStore(ctx, situated=SituatedVectors(ctx), legacy=LegacySemanticLayer(ctx))
    mark = f"つなぎの検査 {uuid.uuid4()}"
    oid = str(uuid.uuid4())
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind) "
                "VALUES (%s, %s, %s, '発話', 'observation')",
                (oid, f"つなぎに言った：{mark}", datetime.now(timezone.utc)),
            )
            cur.execute(
                "INSERT INTO situated_memories "
                "(id, obs_id, person_id, relation_key, vector, content) "
                "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                (str(uuid.uuid4()), oid, viewpoint_of(DEFAULT_PERSON_ID), _VEC, mark),
            )
        conn.commit()

    def _found():
        return {
            str(r.get("memory_id") or r.get("id"))
            for r in store.recency_fallback(400, None)
            if mark in str(r.get("summary") or r.get("content") or "")
        }

    # 役割を付ける前は出る（絞りが常に偽になっていないことの確認）。
    assert _found() == {oid}
    RelationStore(ctx).add("やりとり", [(oid, "つなぎ", 0)])
    assert _found() == set()
