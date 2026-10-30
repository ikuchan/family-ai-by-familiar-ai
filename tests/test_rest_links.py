"""経験繋がり（出-bq・2026-10-29・本人の決定）の、DB を使う側。

関係の層：材料を読む口（直近の共起・継起・やりとりと、最終版の親と、既存の経験繋がり）、項を差し替える口、
関係を外す口。観測の層：埋め込みと時刻を読む口。どれも OIF の面から呼ぶ。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.io.oif import OIF
from familiar_agent.person_memory_manager import AGENT_SELF_ID

KIND = "経験繋がり"


def _conn():
    c = psycopg2.connect(os.environ["DATABASE_URL"], cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


@pytest.fixture
def oif() -> OIF:
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return OIF(ObservationMemory().for_person(AGENT_SELF_ID))


def _members(rid: int) -> "list[tuple[str, str, int]]":
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT obs_id, role, position FROM relation_members "
                "WHERE relation_id = %s ORDER BY position",
                (rid,),
            )
            return [(r["obs_id"], r["role"], r["position"]) for r in cur.fetchall()]
    finally:
        conn.close()


def _terms(ids):
    return [(i, "項", p) for p, i in enumerate(ids)]


# ── 項を差し替える・外す ──────────────────────────────────────────────


def test_relink_replaces_the_members_and_keeps_the_id(oif):
    rid = oif.link(KIND, _terms(["x", "y", "z"]))
    assert _members(rid) == [
        ("x", "項", 0),
        ("y", "項", 1),
        ("z", "項", 2),
    ]  # 反証の側：入れ替える前
    assert oif.relink(KIND, rid, _terms(["u", "v", "w"])) == 3
    assert _members(rid) == [("u", "項", 0), ("v", "項", 1), ("w", "項", 2)]


def test_a_failing_relink_leaves_the_old_members(oif):
    rid = oif.link(KIND, _terms(["x", "y", "z"]))
    with pytest.raises(Exception):
        oif.relink(KIND, rid, [("u", "項", 0), ("v", "項", "位置でない")])
    assert _members(rid) == [("x", "項", 0), ("y", "項", 1), ("z", "項", 2)]


def test_relink_without_members_leaves_the_old_members(oif):
    """項の無い関係を残さない（`add` と同じ）。"""
    rid = oif.link(KIND, _terms(["x", "y", "z"]))
    assert oif.relink(KIND, rid, []) == 0
    assert _members(rid) == [("x", "項", 0), ("y", "項", 1), ("z", "項", 2)]


def test_relink_and_unlink_touch_only_the_named_kind(oif):
    """id を取り違えても、ほかの種類の関係（改訂の旧など）を書き換えたり消したりしない。"""
    old, new = f"{uuid.uuid4().hex[:8]}-x", f"{uuid.uuid4().hex[:8]}-y"  # 旧は 1 記録に 1 本
    rid = oif.link("改訂", [(old, "旧", 0), (new, "新", 1)])
    assert oif.relink(KIND, rid, _terms(["u", "v", "w"])) == 0
    assert oif.unlink(KIND, rid) is False
    assert _members(rid) == [(old, "旧", 0), (new, "新", 1)]
    assert oif.unlink("改訂", rid) is True  # 反証の側：種類が合えば外れる
    assert _members(rid) == []


def test_unlink_removes_the_relation_and_its_members(oif):
    rid = oif.link(KIND, _terms(["x", "y", "z"]))
    other = oif.link(KIND, _terms(["p", "q", "r"]))
    assert oif.unlink(KIND, rid) is True
    assert _members(rid) == []
    assert len(_members(other)) == 3  # 反証の側：ほかの関係は残る


# ── 材料を読む ──────────────────────────────────────────────────────


def test_the_material_holds_recent_edges_parents_and_links(oif):
    now = datetime.now(timezone.utc)
    final, origin = str(uuid.uuid4()), str(uuid.uuid4())
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, parent_id) "
            "VALUES (%s, 'c', now(), '最終版', %s), (%s, 'c', now(), '会話', NULL)",
            (final, origin, origin),
        )
    conn.close()
    co = oif.link("共起", _terms(["s", "a", "b"]))
    su = oif.link("継起", [("f0", "前", 0), (final, "後", 1)])
    ex = oif.link("やりとり", [("s", "起点", 0)])
    t = uuid.uuid4().hex[:8]  # 役割「旧」は 1 つの記録に 1 本（`idx_relation_members_old`）
    rev = oif.link("改訂", [(f"{t}-a", "旧", 0), (f"{t}-a2", "新", 1)])
    old = oif.link("共起", _terms(["s", "c", "d"]))
    link = oif.link(KIND, _terms(["a", "b", "f0"]))
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE relations SET created_at = %s WHERE id = %s", (now - timedelta(days=8), old)
        )
    conn.close()

    # 関係の表は試験ごとに空にしない（`conftest._TRUNCATE_TABLES`）ので、ここで書いたものだけを見る。
    m = oif.experience_material(now - timedelta(days=7))
    got = {e["id"]: e for e in m["edges"]}
    assert {co, su, ex} <= set(got)
    assert rev not in got and old not in got  # 改訂と 8 日前の共起は入らない
    assert got[co]["kind"] == "共起" and got[co]["members"] == [
        ("s", "項"),
        ("a", "項"),
        ("b", "項"),
    ]
    assert got[su]["created_at"].tzinfo is not None
    assert m["parents"].get(final) == origin and origin not in m["parents"]
    assert m["links"][link]["ids"] == ["a", "b", "f0"]
    assert m["links"][link]["at"].tzinfo is not None
    assert rev not in m["links"] and co not in m["links"]


def test_embeddings_return_the_time_and_the_vector(oif):
    have, none = str(uuid.uuid4()), str(uuid.uuid4())
    v = np.arange(1024, dtype=np.float32)
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction) "
            "VALUES (%s, 'c', '2026-10-20 10:00+09', '会話'), (%s, 'c', now(), '会話')",
            (have, none),
        )
        cur.execute(
            "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s)", (have, v.tobytes())
        )
    conn.close()
    got = oif.embeddings([have, none])
    assert set(got) == {have}  # 埋め込みの無い記録は返らない
    ts, vec = got[have]
    assert ts == datetime(2026, 10, 20, 1, 0, tzinfo=timezone.utc)
    assert np.array_equal(vec, v)


# ── REST の段（実 DB） ─────────────────────────────────────────────────


def _axis(i: int, lean: "int | None" = None) -> bytes:
    """i 番目の軸の単位ベクトル。`lean` を渡すと、その軸へ少し寄せる（ほぼ重なる記録）。"""
    v = np.zeros(1024, dtype=np.float32)
    v[i] = 1.0
    if lean is not None:
        v[lean] = 8.0
    return (v / np.linalg.norm(v)).astype(np.float32).tobytes()


def _plant_cluster(t: str, now: datetime) -> "dict[str, str]":
    """今日の起点 s と、使われた記憶 a・b・c・d の共起（5 件）。a・b・c は互いに直交し、s は a に、d は b に寄る。"""
    ids = {k: f"{t}-{k}" for k in "sabcd"}
    axes = {
        "s": _axis(0, lean=1),
        "a": _axis(1),
        "b": _axis(2),
        "c": _axis(3),
        "d": _axis(4, lean=2),
    }
    conn = _conn()
    with conn.cursor() as cur:
        for n, k in enumerate("abcds"):
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction) VALUES (%s, 'c', %s, '会話')",
                (ids[k], now - timedelta(hours=5 - n)),
            )
            cur.execute(
                "INSERT INTO obs_embeddings (obs_id, vector) VALUES (%s, %s)", (ids[k], axes[k])
            )
    conn.close()
    return ids


def _agent(oif: OIF):
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent._oif = oif
    return agent


def _links_with(ids) -> "dict[int, list[tuple[str, str, int]]]":
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT r.id FROM relations r JOIN relation_members m ON m.relation_id = r.id "
                "WHERE r.kind = %s AND m.obs_id = ANY(%s)",
                (KIND, list(ids)),
            )
            rids = [int(r["id"]) for r in cur.fetchall()]
    finally:
        conn.close()
    return {rid: _members(rid) for rid in rids}


JST = timezone(timedelta(hours=9))


def _created_at(rid: int) -> datetime:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT created_at FROM relations WHERE id = %s", (rid,))
            return cur.fetchone()["created_at"]
    finally:
        conn.close()


def test_rest_links_the_jump_once_and_keeps_it_on_the_next_pass(oif):
    import asyncio

    from familiar_agent.loop.rest_links import link_experiences

    now = datetime.now(timezone.utc)
    t = uuid.uuid4().hex[:8]
    ids = _plant_cluster(t, now)
    oif.link("やりとり", [(ids["s"], "起点", 0)])
    oif.link("共起", _terms([ids[k] for k in "sabcd"]))
    assert _links_with(ids.values()) == {}  # 反証の側：回す前は無い

    first = asyncio.run(link_experiences(_agent(oif), now=now))
    links = _links_with(ids.values())
    assert first.written >= 1 and len(links) == 1
    (members,) = links.values()
    (rid,) = links
    # 関係の時刻は、書いた時刻でなく塊の日（日本時間の正午）。7 日の窓から出た日の関係に触らないため
    assert _created_at(rid) == datetime.combine(
        now.astimezone(JST).date(), datetime.min.time().replace(hour=12), JST
    )
    # a・b・c が互いに直交（類似度 0）。位置は記録の時刻の古い順（a → b → c）。
    assert members == [(ids["a"], "項", 0), (ids["b"], "項", 1), (ids["c"], "項", 2)]

    asyncio.run(link_experiences(_agent(oif), now=now))
    assert _links_with(ids.values()) == links  # 2 回目は、id も項も位置も変わらない


def test_a_cluster_of_four_is_left_alone(oif):
    import asyncio

    from familiar_agent.loop.rest_links import link_experiences

    now = datetime.now(timezone.utc)
    ids = _plant_cluster(uuid.uuid4().hex[:8], now)
    oif.link("やりとり", [(ids["s"], "起点", 0)])
    oif.link("共起", _terms([ids[k] for k in "sabc"]))
    asyncio.run(link_experiences(_agent(oif), now=now))
    assert _links_with(ids.values()) == {}
