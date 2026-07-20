"""層ごとの単体テスト（C3）.

合成へ組み替えたことで、各層を `StoreContext` だけで組み立てられるようになった。
これまで `ObservationMemory` 越しにしか触れなかった振る舞いに、直接テストを当てる。

撤去予定の `LegacySemanticLayer` には足さない。隔離が目的であって延命ではない。
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from familiar_agent.db import get_db
from familiar_agent.legacy.semantic_layer import LegacySemanticLayer
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.jobs import JobQueue
from familiar_agent.store.observations import ObservationStore
from familiar_agent.store.situated import SituatedVectors


class _StubEmbedder:
    """埋め込みモデルを読み込まずに済ませる（層のテストに GPU は要らない）。"""

    def encode_document(self, texts):
        return [[0.1] * 1024 for _ in texts]


@pytest.fixture
def ctx() -> StoreContext:
    db = get_db()
    return StoreContext(
        db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=_StubEmbedder()
    )


@pytest.fixture
def layers(ctx):
    situated = SituatedVectors(ctx)
    legacy = LegacySemanticLayer(ctx)
    observations = ObservationStore(ctx, situated=situated, legacy=legacy)
    jobs = JobQueue(ctx, observations=observations)
    return situated, observations, jobs


# ── SituatedVectors ─────────────────────────────────────────────────────────

def test_perspective_vector_round_trips(ctx) -> None:
    """視点ベクトルを更新すると、次に読んだとき反映されている。"""
    situated = SituatedVectors(ctx)
    before = situated._get_perspective_vec(DEFAULT_PERSON_ID)
    situated.update_perspective_vec(DEFAULT_PERSON_ID, np.ones(1024, dtype=np.float32))
    after = situated._get_perspective_vec(DEFAULT_PERSON_ID)
    assert not np.allclose(before, after)


def test_situated_rows_are_created_for_each_person(ctx) -> None:
    """観測1件につき、person ごとの situated 行ができる（想起の母集合）。"""
    situated = SituatedVectors(ctx)
    obs_id = str(uuid.uuid4())
    conn = ctx.conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO observations (id, content, timestamp, direction, kind, person_id) "
            "VALUES (%s,%s,NOW(),%s,%s,%s)",
            (obs_id, f"unit situated {obs_id}", "会話", "conversation", DEFAULT_PERSON_ID),
        )
    conn.commit()

    situated.refresh_situated_embeddings(conn, obs_id, np.ones(1024, dtype=np.float32))
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM situated_embeddings WHERE obs_id = %s", (obs_id,)
        )
        assert int(cur.fetchone()["n"]) >= 1


# ── ObservationStore ────────────────────────────────────────────────────────

def test_saved_observation_is_readable_by_kind(layers) -> None:
    """書いたものが by_kind で読める（追記と読み出しが噛み合う）。"""
    _, observations, _ = layers
    content = f"unit store {uuid.uuid4()}"
    observations.materialize_save_event(
        str(uuid.uuid4()),
        {"content": content, "direction": "会話", "kind": "curiosity", "emotion": "neutral"},
    )
    rows = observations._read_observations_by_kind(
        "curiosity", DEFAULT_PERSON_ID, 50, ("content",)
    )
    assert any(r["content"] == content for r in rows)


def test_supersede_marks_the_old_row_without_deleting_it(layers) -> None:
    """supersede は追記であって削除でない。"""
    _, observations, _ = layers
    old_id, new_id = str(uuid.uuid4()), str(uuid.uuid4())
    for oid in (old_id, new_id):
        observations.materialize_save_event(
            oid, {"content": f"unit supersede {oid}", "direction": "会話", "kind": "conversation"}
        )
    observations.mark_superseded(old_id, new_id)

    chain = observations._read_supersede_chain(new_id, ("id",))
    assert len(chain) >= 2, "版チェーンに旧版が残っていない"


def test_reader_filters_by_person(layers, ctx) -> None:
    """person が違えば読めない（by_kind は person で絞る）。"""
    _, observations, _ = layers
    content = f"unit person {uuid.uuid4()}"
    observations.materialize_save_event(
        str(uuid.uuid4()),
        {"content": content, "direction": "会話", "kind": "curiosity"},
    )
    other = observations._read_observations_by_kind(
        "curiosity", str(uuid.uuid4()), 50, ("content",)
    )
    assert all(r["content"] != content for r in other)


# ── JobQueue ────────────────────────────────────────────────────────────────

def test_job_round_trip_from_enqueue_to_materialize(layers) -> None:
    """積む → 拾う → 実体化 → 完了、の一巡が通る。"""
    _, observations, jobs = layers
    content = f"unit job {uuid.uuid4()}"
    event_id, created = jobs.append_memory_event(
        "memory.save", {"content": content, "direction": "会話", "kind": "conversation"}
    )
    assert created and event_id

    claimed = jobs.claim_pending_jobs(limit=50)
    assert any(j["event_id"] == event_id for j in claimed), "積んだジョブを拾えない"

    assert jobs.materialize_event(event_id) is True
    rows = observations._read_observations_by_kind(
        "conversation", DEFAULT_PERSON_ID, 100, ("content",)
    )
    assert any(r["content"] == content for r in rows)


def test_failed_job_retries_then_becomes_dead_letter(layers) -> None:
    """失敗は再試行し、上限で dead_letter になる（無限に居座らない）。"""
    _, _, jobs = layers
    event_id, _ = jobs.append_memory_event(
        "memory.save", {"content": f"unit fail {uuid.uuid4()}", "kind": "conversation"}
    )
    claimed = [j for j in jobs.claim_pending_jobs(limit=50) if j["event_id"] == event_id]
    assert claimed
    job_id = claimed[0]["job_id"]

    assert jobs.mark_job_failed(job_id, "unit test", retry_delay=0.0, max_attempts=1) == "dead_letter"


def test_duplicate_event_is_not_queued_twice(layers, ctx) -> None:
    """同じ dedupe_key のイベントは二重に積まれず、既存のものが返る。

    「新規でない」だけを見ると、重複を弾いた場合と、書き込みに失敗した場合を
    区別できない（失敗しても新規でないと返る）。**同じ event_id が返ること**と
    **行が1件しかないこと**まで見る。
    """
    _, _, jobs = layers
    key = f"unit dedupe {uuid.uuid4()}"
    payload = {"content": key, "direction": "会話", "kind": "conversation"}
    first_id, first_new = jobs.append_memory_event("memory.save", payload, dedupe_key=key)
    second_id, second_new = jobs.append_memory_event("memory.save", payload, dedupe_key=key)

    assert first_new is True and first_id
    assert second_new is False, "同じ dedupe_key で二重に積まれた"
    assert second_id == first_id, "既存のイベントが返っていない（弾けていない）"

    conn = ctx.conn()
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM memory_events WHERE dedupe_key = %s", (key,))
        assert int(cur.fetchone()["n"]) == 1


# ── ObservationStore.by_vector（S6c） ───────────────────────────────────────

def test_by_vector_returns_rows_without_scoring(layers, ctx) -> None:
    """類似検索は行を返すだけで、5軸の採点はしない。

    採点（`_score_breakdown`）は W の構築＝想起の仕事で、ストアの仕事ではない
    （[D-データモデル]：層は採点も想起判断も持たない）。ここで返る `score` は
    合成スコアではなく**生のコサイン**である。
    """
    from familiar_agent.db import vec_to_sql

    situated, observations, _ = layers
    obs_id = str(uuid.uuid4())
    content = f"unit by_vector {obs_id}"
    observations.materialize_save_event(
        obs_id, {"content": content, "direction": "会話", "kind": "conversation"}
    )

    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    rows = observations.by_vector(vec_to_sql(vec.tolist()), 20)

    assert rows, "類似検索が何も返さない"
    for r in rows:
        assert -1.0001 <= float(r["score"]) <= 1.0001, "生のコサインでない値が返っている"
    assert {"activation_a0", "activation_n", "emotion_p", "last_recalled_at"} <= set(rows[0])


def test_by_vector_filters_by_kind(layers) -> None:
    """kind を渡すとその種別だけが返る。"""
    from familiar_agent.db import vec_to_sql

    _, observations, _ = layers
    content = f"unit by_vector kind {uuid.uuid4()}"
    observations.materialize_save_event(
        str(uuid.uuid4()), {"content": content, "direction": "会話", "kind": "curiosity"}
    )
    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    rows = observations.by_vector(vec_to_sql(vec.tolist()), 50, kind="day_summary")
    assert all(r["kind"] == "day_summary" for r in rows)


def test_by_vector_applies_the_cosine_floor_inside_the_query(layers) -> None:
    """min_cosine は SQL の中で効く（LIMIT と組み合わさるため意味が変わらない）。

    後段で絞ると「閾値を満たす n 件」でなく「n 件のうち閾値を満たすもの」になり、
    挙動が変わる。境界の切り出しで意味論を変えないため、ここは引数で渡す。
    """
    from familiar_agent.db import vec_to_sql

    _, observations, _ = layers
    observations.materialize_save_event(
        str(uuid.uuid4()),
        {"content": f"unit floor {uuid.uuid4()}", "direction": "会話", "kind": "conversation"},
    )
    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    q = vec_to_sql(vec.tolist())

    assert observations.by_vector(q, 50, min_cosine=0.0), "床0で何も返らない"
    assert observations.by_vector(q, 50, min_cosine=1.5) == [], "床を超える行が返っている"


def test_store_layers_do_not_read_configuration(layers) -> None:
    """層は Config を持たない（環境変数を直接読まない）。

    設定は呼び出し側が持ち、層は引数で受け取る。層が設定を抱えると、同じ層を
    別の設定で使えなくなり、テストでも差し替えにくくなる。
    """
    import pathlib

    for path in (
        "src/familiar_agent/store/observations.py",
        "src/familiar_agent/store/jobs.py",
        "src/familiar_agent/store/situated.py",
        "src/familiar_agent/legacy/semantic_layer.py",
    ):
        src = pathlib.Path(path).read_text()
        assert "os.environ" not in src, f"{path} が環境変数を直接読んでいる"
