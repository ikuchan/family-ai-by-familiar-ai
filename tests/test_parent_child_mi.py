"""親子の MI（求めと、その求めのために投げた調査）。

調査は複数並行しうるので、状態は1本の鎖では表せない。親（人の求め・情動）と子（調査）の
**2階層だけ**を持ち、孫は作らない。**親を閉じるとき、生きている子を全部閉じる**（一段だけ・
再帰なし）。子どうしの supersede は従来どおり。

抜けの検出は作らない。W から落ちたものは薄れた＝忘れたのであって、改めて調べるのが自然な
振る舞いである（W は「速く薄れる」・用語一覧）。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = False
    return c


_AGENT_SELF = "00000000-0000-0000-0000-000000000000"


def _plant(conn, content: str, parent_id: str | None = None) -> str:
    oid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, created_at, updated_at) VALUES (%s, %s, now(), now()) "
            "ON CONFLICT (id) DO NOTHING",
            (_AGENT_SELF, "__self__"),
        )
        cur.execute(
            "INSERT INTO observations (id,content,timestamp,direction,kind,emotion,person_id,"
            " writer_id,subject_id,participants_json,scope,parent_id) "
            "VALUES (%s,%s,%s,'unknown','observation','neutral',%s,%s,%s,'[]','speaker',%s)",
            (oid, content, datetime.now(timezone.utc), _AGENT_SELF, _AGENT_SELF,
             _AGENT_SELF, parent_id),
        )
    conn.commit()
    return oid


def test_parent_id_column_exists() -> None:
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='observations' AND column_name='parent_id'")
        row = cur.fetchone()
    conn.close()
    assert row is not None


def test_closing_a_parent_closes_its_live_children() -> None:
    from familiar_agent.tools.memory import ObservationMemory
    from unittest.mock import patch
    from familiar_agent.tools.memory import _EmbeddingModel

    conn = _conn()
    tag = uuid.uuid4().hex[:8]
    parent = _plant(conn, f"{tag} 求め")
    child_a = _plant(conn, f"{tag} 調査A", parent_id=parent)
    child_b = _plant(conn, f"{tag} 調査B", parent_id=parent)
    closer = _plant(conn, f"{tag} 決着")

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    mem.close_with_children(parent, closer)

    with conn.cursor() as cur:
        cur.execute("SELECT id, superseded_by FROM observations WHERE id = ANY(%s)",
                    ([parent, child_a, child_b],))
        got = {r["id"]: r["superseded_by"] for r in cur.fetchall()}
    conn.close()
    assert got[parent] == closer
    assert got[child_a] == closer      # 生きている子は全部閉じる
    assert got[child_b] == closer
