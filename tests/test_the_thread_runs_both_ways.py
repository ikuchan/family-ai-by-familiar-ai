"""継起の鎖を前にも後ろにもたどる（記-q・2026-10-15・本人の決定ハ・ホ・メ）。

たどる口は前へさかのぼる `recent_exchanges` だけで、「あのあとどうなった？」に答えられなかった。
- ハ：主LLM の道具 `recall_thread`。W に載った記録の id（12 桁）を渡すと、その前後の鎖を返す。前後の件数は主LLM が
  引数で決め、上限は 10 件〔仮〕（超えたら 10 に切り、そう書く。上限は道具の説明でも教える）。
- ホ：関連想起の候補集めに、継起の前後 1 つずつを 4 つ目の出どころとして混ぜる（上限 `diffuse_max_add` の中で取り合う）。
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from familiar_agent.store.relations import KIND_REVISION, KIND_SUCCESSION

# ── 店（DB）──────────────────────────────────────────────────────────────────


@pytest.fixture
def store():
    from familiar_agent.db import get_db
    from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
    from familiar_agent.store.context import StoreContext
    from familiar_agent.store.relations import RelationStore

    db = get_db()
    return RelationStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )


def _put(store, content: str, at: datetime) -> str:
    oid = str(uuid.uuid4())
    with store._ctx.lock:
        conn = store._ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction) "
                "VALUES (%s, %s, %s, '最終版')",
                (oid, content, at),
            )
        conn.commit()
    return oid


def _link(store, before: str, after: str) -> None:
    store.add(KIND_SUCCESSION, [(before, "前", 0), (after, "後", 1)])


def _linked(store, n: int) -> "list[str]":
    """古い順に n 件を継起で結ぶ。"""
    base = datetime(1991, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=random.randint(0, 10**7))
    ids = [_put(store, f"〔最終版 {i}〕", base + timedelta(minutes=i)) for i in range(n)]
    for a, b in zip(ids, ids[1:]):
        _link(store, a, b)
    return ids


def test_the_thread_reads_before_and_after_in_order(store):
    a, b, c = _linked(store, 3)
    got = store.thread(b, before=1, after=1)
    assert [r["id"] for r in got] == [a, b, c]
    assert [r["offset"] for r in got] == [-1, 0, 1]
    only_after = store.thread(b, before=0, after=1)
    assert [r["id"] for r in only_after] == [b, c]


def test_the_thread_stops_at_the_asked_length(store):
    ids = _linked(store, 5)
    got = store.thread(ids[2], before=1, after=1)
    assert [r["id"] for r in got] == ids[1:4]  # **反証の側**：2 つ先はたどらない


def test_a_revised_version_is_skipped_and_a_lone_record_stands_alone(store):
    a, b, c = _linked(store, 3)
    newer = _put(store, "〔最終版 畳んだ側〕", datetime(1991, 1, 1, tzinfo=timezone.utc))
    store.add(KIND_REVISION, [(c, "旧", 0), (newer, "新", 1)])  # c は畳まれた
    assert [r["id"] for r in store.thread(b, before=1, after=1)] == [a, b]
    lone = _put(store, "〔最終版 ひとつ〕", datetime(1991, 1, 1, tzinfo=timezone.utc))
    assert [r["id"] for r in store.thread(lone, before=3, after=3)] == [lone]


def test_a_short_id_finds_the_record(store):
    (a,) = _linked(store, 1)
    short = a.replace("-", "")[:12]
    assert store.id_from_short(short) == a
    assert store.id_from_short("000000000000") is None
    assert store.id_from_short("xyz") is None  # 形の違う id は引かない


def test_the_neighbors_are_one_step_each_way(store):
    ids = _linked(store, 5)
    got = set(store.succession_neighbors([ids[2]]))
    assert got == {ids[1], ids[3]}
    assert ids[0] not in got and ids[4] not in got  # **反証の側**：2 つ先は返さない


# ── 道具（ループ）──────────────────────────────────────────────────────────


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    a = _agent(stream_returns=[])
    return InformationProcessing(a), a


def test_the_tool_is_offered_and_tells_its_limit():
    from familiar_agent.loop import event_loop

    assert "recall_thread" in event_loop._FULL_ACTIONS
    desc = event_loop._WIDEN_DEFS["recall_thread"]["description"]
    assert "10 件まで" in desc
    req = event_loop._WIDEN_DEFS["recall_thread"]["input_schema"]["required"]
    assert set(req) == {"id", "before", "after"}  # 件数は主LLM が決める（既定は持たない）


def test_a_long_ask_is_cut_to_ten_and_says_so():
    ip, a = _ip()
    a._oif.id_from_short = MagicMock(return_value="full-id")
    a._oif.thread = MagicMock(
        return_value=[
            {"id": "p", "content": "前の話", "timestamp": None, "offset": -1},
            {"id": "full-id", "content": "この話", "timestamp": None, "offset": 0},
            {"id": "n", "content": "あとの話", "timestamp": None, "offset": 1},
        ]
    )
    out = asyncio.run(
        ip._run_widen("recall_thread", {"id": "abcdef012345", "before": 11, "after": 3})
    )
    _args, kw = a._oif.thread.call_args
    assert (kw["before"], kw["after"]) == (10, 3)
    first = out.splitlines()[0]
    assert "10 件まで" in first
    assert "【前 1】" in out and "【これ】" in out and "【後 1】" in out
    assert out.index("前の話") < out.index("この話") < out.index("あとの話")


def test_an_unknown_id_is_said_so():
    ip, a = _ip()
    a._oif.id_from_short = MagicMock(return_value=None)
    a._oif.thread = MagicMock()
    out = asyncio.run(
        ip._run_widen("recall_thread", {"id": "abcdef012345", "before": 1, "after": 1})
    )
    assert "見つからない" in out
    a._oif.thread.assert_not_called()


# ── 関連想起の 4 つ目の出どころ ─────────────────────────────────────────────


def test_the_diffuse_candidates_include_the_thread_neighbors(monkeypatch):
    """ほかの出どころを空の偽物にすると、継起の前後だけが候補に残る。"""
    from familiar_agent import diffuse_store
    from familiar_agent.store import relations
    from familiar_agent.tools.memory import ObservationMemory

    monkeypatch.setattr(relations.RelationStore, "cooccurring", lambda self, *a, **k: [])
    monkeypatch.setattr(
        relations.RelationStore,
        "succession_neighbors",
        lambda self, ids, conn=None: ["prev-x", "next-x"],
    )
    monkeypatch.setattr(diffuse_store, "fetch_relation_persons", lambda conn, ids: [])
    monkeypatch.setattr(diffuse_store, "shared_memory_ids", lambda conn, ids, limit: [])
    monkeypatch.setattr(diffuse_store, "order_ids_by_stalest", lambda conn, ids, pid: list(ids))
    monkeypatch.setattr(
        diffuse_store, "fetch_diffuse_rows", lambda conn, ids: [{"memory_id": i} for i in ids]
    )

    mem = ObservationMemory.__new__(ObservationMemory)
    mem._person_id = "someone"
    mem._db = MagicMock()
    mem._db_lock = MagicMock()
    mem._ctx = MagicMock()
    cfg = MagicMock(diffuse_max_add=4, diffuse_max_depth=1, diffuse_far_share=0.5)
    got = mem._diffuse_extend([{"memory_id": "seed"}], cfg)
    assert {r["memory_id"] for r in got} == {"prev-x", "next-x"}
