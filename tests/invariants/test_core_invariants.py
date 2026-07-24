"""システムが守ると言っていることの生存確認（不変条件）。

個々の機能の正しさではなく、**壊れていたら気づけること**を目的にする少数のテスト。
中身（どの記憶がどの順で返るか）は固定しない。想起の改良のたびに落ちて邪魔になり、
守りたいものと違うためである。

各不変条件の根拠は設計にある（[D-O書込]／[D-データモデル]／[D-記憶単一化]／
[D-B分離]／[D-値踏み]）。
"""

from __future__ import annotations

import os

import math
import uuid
from unittest.mock import AsyncMock, patch

import psycopg2
import psycopg2.extras
import pytest

from familiar_agent.mood_register import (
    MOOD_STATE_KEY,
    MoodPAD,
    decay_to_rest,
    nudge_current_mood,
)

pytestmark = pytest.mark.invariant


_DB_URL = os.environ["DATABASE_URL"]


def _pg():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _count(cur, sql: str, params=()) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(next(iter(row.values())))


# ── 1. 追記される（[D-O書込]） ───────────────────────────────────────────────

def test_saving_adds_an_observation(memory) -> None:
    """保存すると観測が1件増える。経路の中身は問わない。"""
    conn = _pg()
    with conn.cursor() as cur:
        before = _count(cur, "SELECT count(*) FROM observations")
    memory.save(f"invariant append {uuid.uuid4()}", direction="会話", kind="conversation")
    with conn.cursor() as cur:
        after = _count(cur, "SELECT count(*) FROM observations")
    conn.close()
    assert after == before + 1


@pytest.mark.asyncio
async def test_conversation_turn_persists_the_memory() -> None:
    """会話ターンの後、記憶の保存が呼ばれる。

    2026-06-29 から 2026-07-20 まで、`say()` だけで話したターンで永続化が丸ごと
    飛ばされ、記憶が3週間書かれなかった。経路を問わず「ターンが記憶を残す」ことを
    見る。
    """
    from familiar_agent.agent import EmbodiedAgent
    from familiar_agent.backend import ToolCall

    from tests.test_agent_react_loop import _make_agent, _patch_heavy, _turn

    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (
                _turn("tool_use", text="", tool_calls=[
                    ToolCall(id="t1", name="say", input={"text": "おはよう"})
                ]),
                "",
            ),
            (_turn("end_turn", text=""), ""),
        ]
    )

    # 永続化パイプラインだけは本物を通す（保存が呼ばれるかを見たいため）。
    ps = _patch_heavy({
        "familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline":
            EmbodiedAgent._run_post_response_pipeline,
    })
    for p in ps:
        p.start()
    try:
        await agent.run("おはよう")
        # 永続化は背景タスクなので、run() を await しただけでは終わっていない。
        await agent._drain_background_tasks()
    finally:
        for p in ps:
            p.stop()

    # 永続化は save_async／save_async_with_id のどちらかを通る（会話 save は id 捕捉で
    # save_async_with_id・拡散想起 WR の新記憶↔W 接続のため）。経路を問わず残ることを見る。
    _persisted = (
        agent._memory.save_async.await_count
        + agent._memory.save_async_with_id.await_count
    )
    assert _persisted >= 1, "ターンが記憶を残していない"


# ── 2. 想起の母集合に入る ───────────────────────────────────────────────────

def test_saved_observation_gets_vectors(memory) -> None:
    """保存した観測に埋め込みと situated 行が付く。無ければ想起で引けない。"""
    content = f"invariant vectors {uuid.uuid4()}"
    memory.save(content, direction="会話", kind="conversation")

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM observations WHERE content = %s", (content,))
        obs_id = cur.fetchone()["id"]
        obs_vec = _count(cur, "SELECT count(*) FROM obs_embeddings WHERE obs_id = %s", (obs_id,))
        sit = _count(
            cur, "SELECT count(*) FROM situated_embeddings WHERE obs_id = %s", (obs_id,)
        )
    conn.close()
    assert obs_vec == 1, "埋め込みが無い"
    assert sit >= 1, "situated 行が無い"


# ── 3. 遅延書き込みが最後まで届く（[D-O書込] のイベントログ） ────────────────

def test_deferred_write_reaches_observations(memory) -> None:
    """materialize_now=False で積んだ書き込みが、観測として現れる。"""
    content = f"invariant deferred {uuid.uuid4()}"
    memory.save(content, direction="会話", kind="conversation", materialize_now=False)

    # save は bool を返すので、積まれたイベントを memory_events から引いて流す。
    conn0 = _pg()
    with conn0.cursor() as cur:
        cur.execute(
            "SELECT event_id FROM memory_events WHERE payload_json LIKE %s",
            (f"%{content}%",),
        )
        row = cur.fetchone()
    conn0.close()
    assert row is not None, "イベントが積まれていない"
    memory.materialize_event(row["event_id"])

    conn = _pg()
    with conn.cursor() as cur:
        found = _count(cur, "SELECT count(*) FROM observations WHERE content = %s", (content,))
    conn.close()
    assert found == 1, "遅延書き込みが観測に現れていない"


# ── 4. 追記であって削除でない（[D-データモデル]） ───────────────────────────

def test_supersede_keeps_the_original_row(memory) -> None:
    """supersede しても元の行は消えない。"""
    old = f"invariant supersede old {uuid.uuid4()}"
    new = f"invariant supersede new {uuid.uuid4()}"
    memory.save(old, direction="会話", kind="conversation")
    memory.save(new, direction="会話", kind="conversation")

    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM observations WHERE content = %s", (old,))
        old_id = cur.fetchone()["id"]
        cur.execute("SELECT id FROM observations WHERE content = %s", (new,))
        new_id = cur.fetchone()["id"]

    memory.mark_superseded(old_id, new_id)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT superseded_by FROM observations WHERE id = %s", (old_id,)
        )
        row = cur.fetchone()
    conn.close()
    assert row is not None, "元の行が消えている（追記モデルが壊れている）"
    assert str(row["superseded_by"]) == str(new_id)


# ── 5. W が O から作れる（[D-記憶単一化]） ──────────────────────────────────

def test_recall_returns_usable_scores(memory) -> None:
    """想起が候補を返し、スコアが有限かつ 0 以上。中身は問わない。"""
    memory.save(f"invariant recall {uuid.uuid4()}", direction="会話", kind="conversation")

    from familiar_agent.tools.memory import _EmbeddingModel

    with patch.object(_EmbeddingModel, "encode_query", return_value=[[0.1] * 1024]):
        results = memory.recall("invariant recall", n=5)

    assert results, "想起が何も返さない"
    for r in results:
        score = r["score"]
        assert math.isfinite(score), f"スコアが有限でない: {score}"
        assert score >= 0.0, f"スコアが負: {score}"


# ── 6・7. T レジスタが動き、放っておけば戻る（[D-B分離]／[D-値踏み]） ────────

def test_mood_is_persisted_after_a_nudge() -> None:
    """nudge の後、mood が保存されている。"""
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM agent_state WHERE state_key = %s", (MOOD_STATE_KEY,))

    nudge_current_mood([(MoodPAD(0.9, 0.1, 0.8, 0.7), 1.0)])

    with conn.cursor() as cur:
        found = _count(
            cur, "SELECT count(*) FROM agent_state WHERE state_key = %s", (MOOD_STATE_KEY,)
        )
    conn.close()
    assert found == 1, "mood が保存されていない"


def test_mood_returns_toward_rest_when_left_alone() -> None:
    """放置した mood は中立（rest=0.5）へ寄る。"""
    far = MoodPAD(1.0, 0.0, 1.0, 1.0)
    decayed = decay_to_rest(far, elapsed_seconds=1800.0)
    for before, after in (
        (far.p, decayed.p), (far.pn, decayed.pn), (far.a, decayed.a), (far.dom, decayed.dom)
    ):
        assert abs(after - 0.5) < abs(before - 0.5), "中立へ寄っていない"
