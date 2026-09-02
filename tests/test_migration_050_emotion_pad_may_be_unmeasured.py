"""050：PAD は未測定でありうる（P/Pn/Dom を気分で埋めない）。

**感情軸の母集合は、半分が同じ一点に潰れていた。** 2026-08-21 のダンプでは 6433 行のうち
**2941 行（45.7%）が PAD 全部 0.5** で、`emotion_vec` がゼロベクトルになっていた。内訳は
ほぼ知覚の観察（`observation` 2867）である。L2 ではこの 2941 行がすべて原点に重なるので、
感情軸が候補を並べ替えられない。

**原因は「埋める」ことだった。** 評価器（軽量LLM）は値踏みゲート（`A_GATE`＝0.25）未満だと
呼ばれず、そのとき P/Pn/Dom を**気分の値で埋めていた**。気分が平静なら 0.5 が3つ入る。
埋めた値は測ったものではないので、感情軸の母集合に混ぜてはいけない。0.5 が入っていると
「測ったのか埋めたのか」を後から見分けられず、REST 内省が埋め直す余地も消える。

**A（高ぶり）は機械値なので常に入る**（`_turn_arousal` ＝ 内容の新規性 novelty）。だから
050 は `emotion_a` を触らない。未測定になるのは P/Pn/Dom の3つだけである。

**049（索引を cosine へ張り替える）は復元しない。** cosine ならゼロベクトルが `NaN` に
なって候補から落ちる、という応急処置だったと読めるが、033 の設計は「感情距離はロジット空間の
重み付きユークリッド距離で、pgvector の **L2 距離がそのまま D** になる」と定めている。
050 が入れば未測定は `emotion_vec IS NOT NULL` で母集合から外れるので、NaN に頼る必要が
なくなる（`復旧記録` v0.20）。
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, patch

import psycopg2
import psycopg2.extras

from familiar_agent.mood_register import MoodPAD

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


def _mem():
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _row(obs_id: str) -> dict:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT emotion, emotion_p, emotion_pn, emotion_a, emotion_dom, emotion_vec "
                "FROM observations WHERE id = %s",
                (obs_id,),
            )
            return dict(cur.fetchone())
    finally:
        conn.close()


# ── ① 列の制約 ──────────────────────────────────────────────────────────────

def test_the_three_axes_may_be_null_but_arousal_may_not() -> None:
    """P/Pn/Dom は未測定でありうる。A は機械値なので常に入る。"""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_name='observations' AND column_name LIKE 'emotion_%'"
            )
            nullable = {r["column_name"]: r["is_nullable"] for r in cur.fetchall()}
    finally:
        conn.close()
    for col in ("emotion_p", "emotion_pn", "emotion_dom"):
        assert nullable[col] == "YES", f"{col} が未測定を許していない"
    assert nullable["emotion_a"] == "NO", "A は機械値なので常に入る"


# ── ②③ 評価器は測れないとき「未測定」を返す ────────────────────────────────

def test_the_evaluator_reports_unmeasured_below_the_gate() -> None:
    """ゲート未満では評価器を呼ばず、**気分で埋めずに**未測定を返す。"""
    from familiar_agent.loop.evaluator import A_GATE, _evaluate_emotion_pad

    backend = AsyncMock()
    mood = MoodPAD(p=0.8, pn=0.1, a=0.9, dom=0.7)
    pad, arousal = asyncio.run(
        _evaluate_emotion_pad(backend, "静かな一日", mood, A_GATE - 0.01)
    )
    assert pad is None, "気分で埋めている"
    assert arousal == A_GATE - 0.01, "A は機械値なので返る"
    backend.complete.assert_not_awaited()


def test_the_evaluator_reports_unmeasured_when_it_fails() -> None:
    """評価器が失敗しても、気分で埋めずに未測定にする。測れなかったのは同じである。"""
    from familiar_agent.loop.evaluator import _evaluate_emotion_pad

    backend = AsyncMock()
    backend.complete.side_effect = RuntimeError("軽量LLM が落ちた")
    pad, arousal = asyncio.run(
        _evaluate_emotion_pad(backend, "何か", MoodPAD(p=0.8, pn=0.1, a=0.9, dom=0.7), 0.9)
    )
    assert pad is None
    assert arousal == 0.9


# ── ④⑤⑥ 書き込みと想起 ────────────────────────────────────────────────────

def test_an_unmeasured_observation_is_written_as_null() -> None:
    """未測定なら3列と `emotion_vec` は NULL。A には機械値が入る。"""
    content = f"未測定テスト_{uuid.uuid4()}"
    obs_id = _mem().save_with_id(content, kind="observation", arousal=0.4)[0]

    got = _row(obs_id)
    assert got["emotion_p"] is None
    assert got["emotion_pn"] is None
    assert got["emotion_dom"] is None
    assert got["emotion_vec"] is None, "PAD が揃わないのに感情ベクトルを作っている"
    assert got["emotion_a"] == 0.4, "A は機械値なので入る"


def test_a_measured_observation_keeps_its_values() -> None:
    """測れた観測は従来どおり値が入る（反証側）。"""
    content = f"測定済みテスト_{uuid.uuid4()}"
    pad = MoodPAD(p=0.8, pn=0.2, a=0.6, dom=0.7)
    obs_id = _mem().save_with_id(content, kind="observation", emotion_pad=pad)[0]

    got = _row(obs_id)
    assert got["emotion_p"] == 0.8
    assert got["emotion_dom"] == 0.7
    assert got["emotion_vec"] is not None


def test_unmeasured_observations_stay_out_of_the_emotion_axis() -> None:
    """未測定の記録は感情軸の候補に出ない（`emotion_vec IS NOT NULL` で外れる）。"""
    from familiar_agent.person_memory_manager import AGENT_SELF_ID

    content = f"感情軸から外れる_{uuid.uuid4()}"
    obs_id = _mem().save_with_id(content, kind="observation", arousal=0.1)[0]

    store = _mem().for_person(AGENT_SELF_ID)._observations
    rows = store.by_emotion("[0,0,0,0]", 200)
    assert all(r["id"] != obs_id for r in rows), "未測定の記録が感情軸に出ている"
