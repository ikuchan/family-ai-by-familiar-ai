"""042（`observations.person_id` の撤去）が効いていることを確かめる。

**所有者絞りは、そもそも人を分けていなかった。** 2026-08-03 のダンプでは 5080 行のうち
4904 行（96.5%）が既定値 `default` のままで、家族4人のうち2人（いくながこうき・
いくながたえこ）は所有行を1件も持たない。010 が `NOT NULL DEFAULT 'default'` で入れた
列を、書き込み側が文脈の person のまま埋め続けた結果である。

**人と記憶の結びつきは situated が担う**（[D-在席相関/V2]）。047 で `actor`（誰がやったか）
と `present`（誰が居たか）の面が立ったので、設計が定めた順序——**関係生成が立ってから
列を落とす**——の条件が満たされた。設計は「所有者フィルタは廃し、p 軸は在席関係の行を
使う」と定めている（`gap分析` §4）。

**重複判定の30秒窓だけは絞りを保ち、`writer_id` へ移す。** 重複とは「同じ書き手が同じ
内容を同じ kind で窓の内に」であって、家族の二人が同じ挨拶をしたものは重複ではない。

落とすのは `observations.person_id`（所有者）だけである。`situated_memories.person_id`
（誰と関係する面か）と、想起の視点（`store.context.viewpoint_of`）は残る。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID
from familiar_agent.store import clock

_DB_URL = os.environ["DATABASE_URL"]
_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=clock.local_tz())


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _mem():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _insert(cur, obs_id: str, content: str, kind: str, writer_id: str, ts: datetime) -> None:
    """所有者列を書かずに観測を1件植える（042 後の列構成で書けることも兼ねて確かめる）。"""
    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion, writer_id, subject_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (obs_id, content, ts, "unknown", kind, "neutral", writer_id, writer_id),
    )


def test_observations_has_no_person_id_column() -> None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'observations'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "person_id" not in columns, sorted(columns)


def test_situated_keeps_its_person_id() -> None:
    """落とすのは所有者だけ。**誰と関係する面か**は残る。"""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'situated_memories'"
            )
            columns = {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()
    assert "person_id" in columns


def test_read_observations_by_kind_does_not_filter_by_owner() -> None:
    """kind が合う行はすべて返る（所有者では絞らない）。

    042 の前はここが「1件だけ返る」ことを仕様として固定していた。だが所有者は実データで
    `default` に潰れており、分かれて見えたのは fixture が `person_id` を明示して書いて
    いたからだった。人の視点で絞るなら `_read_observations_by_situated` を使う。
    """
    tag = uuid.uuid4().hex[:8]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _insert(cur, f"k-self-{tag}", f"agent curiosity {tag}", "curiosity",
                    AGENT_SELF_ID, _NOW)
            _insert(cur, f"k-user-{tag}", f"user curiosity {tag}", "curiosity",
                    DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1))
    finally:
        conn.close()

    rows = _mem()._observations._read_observations_by_kind(
        "curiosity", 50, ("id", "content")
    )
    contents = {r["content"] for r in rows}
    assert f"agent curiosity {tag}" in contents
    assert f"user curiosity {tag}" in contents


def test_the_fallbacks_do_not_filter_by_owner() -> None:
    """最後の網（語で拾う・新しい順）は所有者で絞らない。

    C-1 はフォールバックに「situated 行を持たない観測を拾う」役目を残したが、その母集合は
    0 行だった（生存する観測の全件が situated 行を持つ）。最後の網を狭めると役目が消える。

    植えた行の所有者は既定（`default`）なので、**AGENT_SELF の文脈**で引いて返れば
    所有者絞りは残っていない。
    """
    tag = uuid.uuid4().hex[:8]
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _insert(cur, f"f-a-{tag}", f"fallback {tag} alpha", "observation",
                    AGENT_SELF_ID, _NOW)
            _insert(cur, f"f-b-{tag}", f"fallback {tag} beta", "observation",
                    DEFAULT_PERSON_ID, _NOW + timedelta(seconds=1))
    finally:
        conn.close()

    store = _mem().for_person(AGENT_SELF_ID)._observations
    by_keyword = {r["summary"] for r in store.keyword_fallback(tag, 50, "observation")}
    assert f"fallback {tag} alpha" in by_keyword
    assert f"fallback {tag} beta" in by_keyword

    by_recency = {r["summary"] for r in store.recency_fallback(50, "observation")}
    assert f"fallback {tag} alpha" in by_recency
    assert f"fallback {tag} beta" in by_recency


def test_dedup_keeps_rows_from_different_writers() -> None:
    """別の書き手が同じ文面を窓の内に書いたら、畳まない。

    重複判定の絞りを外すのでなく**書き手**へ移す。家族の二人が同じ挨拶をしたものは
    重複ではない。
    """
    mem = _mem()
    content = f"別書き手テスト_{uuid.uuid4()}"

    original = os.environ.get("MEMORY_DEDUP_WINDOW_SECS")
    os.environ["MEMORY_DEDUP_WINDOW_SECS"] = "30"
    try:
        mem.save_with_id(content, kind="utterance",
                         writer_id=AGENT_SELF_ID, subject_id=AGENT_SELF_ID)
        mem.save_with_id(content, kind="utterance",
                         writer_id=DEFAULT_PERSON_ID, subject_id=DEFAULT_PERSON_ID)
    finally:
        if original is None:
            os.environ.pop("MEMORY_DEDUP_WINDOW_SECS", None)
        else:
            os.environ["MEMORY_DEDUP_WINDOW_SECS"] = original

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM observations "
                "WHERE content=%s AND kind=%s AND superseded_by IS NULL",
                (content, "utterance"),
            )
            n = cur.fetchone()["n"]
    finally:
        conn.close()
    assert n == 2, f"別の書き手の行まで畳まれた（{n} 行）"
