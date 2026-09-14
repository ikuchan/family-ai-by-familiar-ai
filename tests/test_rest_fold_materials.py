"""日次の畳み込みの材料を引く口（記-a-ろ-は・2026-09-14）。

前回の内省（`direction='内省'` の最新）より後の記録のうち、指定の `direction` で、まだ畳まれて
おらず（役割 `旧` が無い）、核でない（根づき $n < 1$）ものを古い順に返す。`内省` が無ければ
全期間。`求め`（版）・`保留`・`内省`・`記憶` は呼び手が `direction` で外す。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.person_memory_manager import AGENT_SELF_ID

_DB_URL = os.environ["DATABASE_URL"]
_VEC = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
FOLD_DIRECTIONS = ("発話", "会話", "観察", "独白", "情動", "機器")


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _plant(cur, content: str, ts: datetime, *, direction="発話", n=0, hidden=False) -> str:
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion, groundedness_g0)"
        " VALUES (%s, %s, %s, %s, 'observation', 'neutral', 0.5)",
        (oid, content, ts, direction),
    )
    cur.execute(
        "INSERT INTO situated_memories (id, obs_id, person_id, vector, relation_key, groundedness_n)"
        " VALUES (%s, %s, %s, %s, 'present', %s)",
        (str(uuid.uuid4()), oid, AGENT_SELF_ID, _VEC, n),
    )
    if hidden:
        cur.execute("INSERT INTO relations (kind) VALUES ('畳み込み') RETURNING id")
        rid = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO relation_members (relation_id, obs_id, role, position) VALUES (%s, %s, '旧', 0)",
            (rid, oid),
        )
    return oid


def _oif():
    from familiar_agent.io.oif import OIF
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return OIF(ObservationMemory().for_person(AGENT_SELF_ID))


def test_materials_are_the_uncore_unfolded_records_since_the_last_rest():
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            old = _plant(cur, "前回の内省より前", now - timedelta(hours=30))
            _plant(cur, "内省を回した", now - timedelta(hours=24), direction="内省")
            a = _plant(cur, "朝の会話", now - timedelta(hours=20))
            b = _plant(cur, "昼に見たもの", now - timedelta(hours=12), direction="観察")
            core = _plant(cur, "大事な話", now - timedelta(hours=10), n=1)
            folded = _plant(cur, "もう畳んだ", now - timedelta(hours=8), hidden=True)
            req = _plant(cur, "「天気は？」と聞かれ…", now - timedelta(hours=6), direction="求め")
    finally:
        conn.close()

    rows = _oif().since_last_rest(FOLD_DIRECTIONS)
    ids = [r.obs_id for r in rows]
    assert ids == [a, b], ids  # 古い順
    for excluded in (old, core, folded, req):
        assert excluded not in ids


def test_without_a_previous_rest_everything_is_material():
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            a = _plant(cur, "初日の会話", now - timedelta(days=3))
            b = _plant(cur, "二日目", now - timedelta(days=2), direction="独白")
    finally:
        conn.close()
    assert [r.obs_id for r in _oif().since_last_rest(FOLD_DIRECTIONS)] == [a, b]


def test_the_rows_carry_what_the_fold_needs():
    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, "こうきとサッカーの話", now - timedelta(hours=1))
    finally:
        conn.close()
    r = _oif().since_last_rest(FOLD_DIRECTIONS)[0]
    assert r.content == "こうきとサッカーの話" and r.direction == "発話"
    assert r.timestamp is not None


# ── 書く・畳む（実 DB） ───────────────────────────────────────────────────────


def _agent_for_fold(backend_reply: str):
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.io.oif import OIF
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    a = MagicMock()
    with (
        patch.object(_EmbeddingModel, "pre_warm"),
        patch.object(_EmbeddingModel, "encode_document", return_value=[[1.0] + [0.0] * 1023]),
        patch.object(_EmbeddingModel, "encode_query", return_value=[[1.0] + [0.0] * 1023]),
    ):
        a._memory = ObservationMemory()
    a._oif = OIF(a._memory, for_person=lambda pid: a._memory.for_person(pid))
    a.backend = AsyncMock()
    a.backend.complete = AsyncMock(return_value=backend_reply)
    a._family_md = "## パパ\n名前: ゆうすけ\n\n## こうき\n名前: こうき\n"
    a._pmm.find_person_id_by_name = MagicMock(side_effect=lambda n: {"こうき": "pid-kouki"}.get(n))
    a._pmm.get_present_ids = MagicMock(return_value=[])
    a._observation_perspective = MagicMock(
        return_value={"writer_id": AGENT_SELF_ID, "participants": []}
    )
    a._evaluator.emotion_for_turn = AsyncMock(return_value=(None, 0.3, "neutral"))
    return a


def test_fold_day_writes_the_episode_and_person_summaries_and_folds_the_materials():
    import asyncio

    from familiar_agent.loop import rest_fold

    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES ('pid-kouki', 'こうき', 'こうき', now()::text, now()::text) ON CONFLICT DO NOTHING"
            )
            a1 = _plant(cur, "こうき：サッカーしたよ", now - timedelta(hours=5))
            a2 = _plant(
                cur, "自分が答えた：いいね、楽しかった？", now - timedelta(hours=5, minutes=-1)
            )
    finally:
        conn.close()
    agent = _agent_for_fold(
        '{"episode": "今日はこうきとサッカーの話をした。楽しそうで、ぼくもうれしかった。", '
        '"persons": {"こうき": "サッカーが好き。今日は楽しそうだった。"}}'
    )
    with patch("familiar_agent.loop.rest_fold.family_names_of", return_value=("パパ", "こうき")):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert result.folded == 2 and result.written == 2 and result.skipped == 0

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, kind, direction FROM observations WHERE direction IN ('記憶','人物') ORDER BY timestamp"
            )
            written = cur.fetchall()
            # 関係の表はテスト間で消えないので、自分の材料が「畳み込み」の旧で、新が自己エピソードであることだけを見る。
            cur.execute(
                "SELECT old.obs_id AS old_id, new.obs_id AS new_id FROM relation_members old "
                "JOIN relations r ON r.id = old.relation_id AND r.kind = '畳み込み' "
                "JOIN relation_members new ON new.relation_id = r.id AND new.role = '新' "
                "WHERE old.role = '旧' AND old.obs_id = ANY(%s)",
                ([a1, a2],),
            )
            pairs = {(r["old_id"], r["new_id"]) for r in cur.fetchall()}
            cur.execute(
                "SELECT person_id FROM situated_memories WHERE obs_id = (SELECT id FROM observations WHERE kind='person_summary' LIMIT 1)"
            )
            facets = {r["person_id"] for r in cur.fetchall()}
    finally:
        conn.close()
    kinds = {w["kind"] for w in written}
    assert kinds == {"day_summary", "person_summary"}, kinds
    episode_id = next(w["id"] for w in written if w["kind"] == "day_summary")
    assert pairs == {(a1, episode_id), (a2, episode_id)}
    assert "pid-kouki" in facets  # 関係のまとめはその人の面に立つ


def test_a_rejected_summary_writes_nothing_and_folds_nothing():
    import asyncio

    from familiar_agent.loop import rest_fold

    now = datetime.now(timezone.utc)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            _plant(cur, "何か", now - timedelta(hours=1))
    finally:
        conn.close()
    agent = _agent_for_fold('{"episode": "", "persons": {}}')
    with patch("familiar_agent.loop.rest_fold.family_names_of", return_value=("パパ",)):
        result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert result.written == 0 and result.folded == 0 and result.skipped == 1
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM observations WHERE direction='記憶'")
            assert cur.fetchone()["n"] == 0
    finally:
        conn.close()


def test_nothing_to_fold_is_reported():
    import asyncio

    from familiar_agent.loop import rest_fold

    agent = _agent_for_fold("{}")
    result = asyncio.run(rest_fold.fold_since_last_rest(agent))
    assert (result.materials, result.written, result.folded) == (0, 0, 0)
