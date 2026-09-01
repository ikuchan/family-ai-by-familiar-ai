"""043（`observations.recall_count` の撤去）が効いていることを確かめる。

この列は**強化A（durability）**のためのもので、実効半減期を `2^recall_count` で
伸ばしていた。`課題5` F 節が「半減期延長は 根づき の n と役割重複」として廃止を
確定させており、採点側は既に使っていない（`_score_breakdown` は引数で受け取るだけで
`DecayState` へ渡していない）。残っていたのは SELECT と UPDATE と引数の受け渡しだけである。

2026-08-03 のダンプでは 5080 行のうち 637 行が `recall_count≠0` だったが、
その値は採点に一切効いていなかった。

若返り（時間の起点の更新）は `apply_verdicts` が担う。`_mark_recalled` にも同じ
UPDATE があったが、**本番からの呼び出しは0件**だったので 044 で撤去した。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _columns() -> set[str]:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'observations'"
            )
            return {r["column_name"] for r in cur.fetchall()}
    finally:
        conn.close()


def test_observations_has_no_recall_count_column() -> None:
    assert "recall_count" not in _columns()


def test_the_intake_surprise_survives() -> None:
    """落とすのは `recall_count` だけ。取込の驚きは出来事に残る。

    `last_recalled_at` と `groundedness_n` は 043 の時点では観測にあったが、044 で
    `situated_memories` へ移した。ここで確かめるのは、**出来事に残るべきもの**が
    残っていることである。
    """
    cols = _columns()
    for name in ("timestamp", "groundedness_g0"):
        assert name in cols, name


def test_score_breakdown_no_longer_takes_recall_count() -> None:
    """採点の正本が `recall_count` を受け取らない。

    受け取るだけで使っていなかった引数を外す。残しておくと「効いている」と読まれる。
    """
    from familiar_agent.tools.memory import _score_breakdown

    old_ts = datetime.now(timezone.utc) - timedelta(days=30)
    parts = _score_breakdown(
        0.5, old_ts, None, 1.0, 0,
        half_life_days=3.0, floor=0.001,
    )
    assert 0.0 <= parts.t <= 1.0


def test_apply_verdicts_still_refreshes_the_time_origin() -> None:
    """若返りは `apply_verdicts` が担い続ける（`recall_count` 無しで動く）。

    列を落とすと同じ UPDATE 文の中の `recall_count = recall_count + 1` が
    `UndefinedColumn` で落ちるので、ここが落ちれば外し忘れである。

    044 で起点は**面**へ移ったので、確かめるのも面の側である。
    """
    from unittest.mock import patch

    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID

    vec = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s, %s, now(), %s, %s, %s)",
                (obs_id, f"若返りテスト_{obs_id}", "unknown", "observation", "neutral"),
            )
            cur.execute(
                "INSERT INTO situated_memories (id, obs_id, person_id, vector) "
                "VALUES (%s, %s, %s, %s)",
                (str(uuid.uuid4()), obs_id, DEFAULT_PERSON_ID, vec),
            )
    finally:
        conn.close()

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    touched = mem._observations.apply_verdicts({obs_id: "referred"})
    assert touched == 1

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT last_recalled_at FROM situated_memories "
                "WHERE obs_id = %s AND person_id = %s",
                (obs_id, DEFAULT_PERSON_ID),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None and row["last_recalled_at"] is not None, "面の時間の起点が更新されない"
