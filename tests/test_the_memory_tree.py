"""記憶の木（記-m・2026-10-01）。「人 → 年 → 月 → 日」の節を指して、その節の要約を引く。

REST の層 1 が日ごとの要約（向き `記憶`）と人ごとのまとめ（向き `人物`）を書き、③ 暦のまとめが
月と年の要約を書く。節は日（YYYY-MM-DD）・月（YYYY-MM）・年（YYYY）で、要約の時刻（その日・月・年の
終わり・UTC）の範囲で引く。家族全体と人ごとは向きで分け（記-u 段 2）、人ごとはその人の面から引く。

**畳まれた記録も返す**——日ごとの要約は核の固め（`rest_core`）で隠れることがあるが、木は暦で決まる
索引で、意味で束ねる核の固めとは別の仕組みである。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from familiar_agent.core.memory_tree import directions_for, parse_node
from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext, viewpoint_of
from familiar_agent.store.observations import ObservationStore
from familiar_agent.store.situated import SituatedVectors

_VEC = "[" + ",".join(["0.03125"] * 1024) + "]"
UTC = timezone.utc


# ── 節の読み方（純関数）──────────────────────────────────────────────────


def test_a_year_a_month_and_a_day_are_nodes():
    assert parse_node("2025") == (
        "年",
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert parse_node("2025-08") == (
        "月",
        datetime(2025, 8, 1, tzinfo=UTC),
        datetime(2025, 9, 1, tzinfo=UTC),
    )
    assert parse_node("2025-12") == (
        "月",
        datetime(2025, 12, 1, tzinfo=UTC),
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert parse_node(" 2025-08-15 ") == (
        "日",
        datetime(2025, 8, 15, tzinfo=UTC),
        datetime(2025, 8, 16, tzinfo=UTC),
    )


@pytest.mark.parametrize("bad", ["", "去年の夏", "2025-13", "2025-02-30", "25-08", "2025/08"])
def test_other_writings_are_not_nodes(bad):
    assert parse_node(bad) is None


def test_the_directions_follow_the_level_and_whose_tree():
    assert directions_for("日", person=False) == ("記憶",)
    assert directions_for("月", person=False) == ("月のまとめ",)
    assert directions_for("年", person=False) == ("年のまとめ",)
    assert directions_for("日", person=True) == ("人物",)
    assert directions_for("月", person=True) == ("人の月のまとめ",)
    assert directions_for("年", person=True) == ("人の年のまとめ",)


# ── 読み出し（テスト DB）─────────────────────────────────────────────────


@pytest.fixture
def store() -> ObservationStore:
    db = get_db()
    ctx = StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    return ObservationStore(ctx, situated=SituatedVectors(ctx))


def _put(store: ObservationStore, content: str, when: datetime, direction: str, person: str) -> str:
    oid = str(uuid.uuid4())
    with store._ctx.lock:
        conn = store._ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction) "
                "VALUES (%s, %s, %s, %s)",
                (oid, content, when, direction),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, relation_key, vector, content) "
                "VALUES (%s, %s, %s, 'present', %s::vector, %s)",
                (str(uuid.uuid4()), oid, person, _VEC, content),
            )
        conn.commit()
    return oid


def _person(store: ObservationStore, pid: str) -> None:
    """面の人は人物表に居る必要がある（外部キー）。名前は重ならないよう id から作る。"""
    now = datetime.now(UTC).isoformat()
    with store._ctx.lock:
        conn = store._ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, created_at, updated_at) VALUES (%s, %s, %s, %s)",
                (pid, f"木の試験 {pid}", now, now),
            )
        conn.commit()


def test_a_day_node_returns_that_days_summaries_even_folded(store):
    mark = uuid.uuid4().hex[:8]
    me = viewpoint_of(DEFAULT_PERSON_ID)
    d15 = datetime(1999, 8, 15, 23, 59, tzinfo=UTC)
    kept = _put(store, f"{mark} 15 日のこと", d15, "記憶", me)
    folded = _put(store, f"{mark} 15 日のもう一つ", d15, "記憶", me)
    _put(store, f"{mark} 16 日のこと", datetime(1999, 8, 16, 23, 59, tzinfo=UTC), "記憶", me)
    other = _put(store, f"{mark} 核のまとめ", d15, "まとめ", me)
    store.mark_superseded(folded, other)

    _, start, end = parse_node("1999-08-15")
    got = [r["content"] for r in store.tree_summaries(("記憶",), start, end)]
    assert f"{mark} 15 日のこと" in got and f"{mark} 15 日のもう一つ" in got  # 畳まれていても返す
    assert f"{mark} 16 日のこと" not in got and f"{mark} 核のまとめ" not in got
    assert kept


def test_a_persons_node_comes_from_that_persons_face(store):
    mark = uuid.uuid4().hex[:8]
    papa, mama = str(uuid.uuid4()), str(uuid.uuid4())
    _person(store, papa)
    _person(store, mama)
    when = datetime(1999, 8, 31, 23, 59, tzinfo=UTC)
    _put(store, f"{mark} パパの 8 月", when, "人の月のまとめ", papa)
    _put(store, f"{mark} ママの 8 月", when, "人の月のまとめ", mama)

    _, start, end = parse_node("1999-08")
    got = [
        r["content"] for r in store.tree_summaries(("人の月のまとめ",), start, end, person_id=papa)
    ]
    assert got == [f"{mark} パパの 8 月"]


def test_the_months_with_a_direction_are_listed(store):
    """暦のまとめが何を書くかを決める読み口（段 3）：その向きの要約がある月の一覧（UTC の暦・記-u 段 2）。"""
    kind = (
        f"tree_test_{uuid.uuid4().hex[:8]}"  # 試験だけの向きにして、ほかの行と混ざらないようにする
    )
    me = viewpoint_of(DEFAULT_PERSON_ID)
    papa = str(uuid.uuid4())
    _person(store, papa)
    _put(store, "8 月末", datetime(1999, 8, 31, 23, 59, tzinfo=UTC), kind, me)
    _put(store, "9 月頭", datetime(1999, 9, 1, 0, 0, tzinfo=UTC), kind, me)
    _put(store, "パパの 7 月", datetime(1999, 7, 10, 12, 0, tzinfo=UTC), kind, papa)
    assert store.tree_months(kind) == ["1999-07", "1999-08", "1999-09"]
    assert store.tree_months(kind, person_id=papa) == ["1999-07"]
