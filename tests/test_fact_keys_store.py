"""記-ad：抜き待ちの記録を読み、鍵の文を書き、鍵の軸で読む（080 の表・本物の DB）。

Ruri は conftest の偽物（文から決まる数値）。
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import psycopg2
import psycopg2.extras

from familiar_agent.core.fact_keys import Fact
from familiar_agent.db import vec_to_sql
from familiar_agent.store.embedding import _RuriModel
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_DB_URL = os.environ["DATABASE_URL"]
_FIXED = [1.0] + [0.0] * 1023
_BGE_Q = "[" + ",".join(["1"] + ["0"] * 1023) + "]"
_ALL = 1_000_000


def _mem() -> ObservationMemory:
    with patch.object(_EmbeddingModel, "pre_warm"):
        return ObservationMemory()


def _save(mem, body, **kw):
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        obs_id, _ = mem.save_with_id(body, materialize_now=True, **kw)
    return obs_id


def _q(text: str) -> str:
    """偽物の Ruri で、鍵の文（文書として点にしたもの）と同じ点の問い。"""
    from tests.conftest import _HashRuri

    return vec_to_sql(_HashRuri().encode(["検索文書: " + text])[0].tolist())


def _rows(sql, *args):
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            return list(cur.fetchall())
    finally:
        conn.close()


def _pending_of(mem, ids):
    return {
        frozenset(src.obs_ids) & set(ids): src.content
        for src in mem.fact_pending(_ALL)
        if set(src.obs_ids) & set(ids)
    }


def test_pending_groups_records_of_the_same_body_and_skips_hidden_and_empty():
    mem = _mem()
    body = "夏祭り " + uuid.uuid4().hex
    a = _save(mem, body, direction="発話")
    b = _save(mem, body, direction="会話")  # 同じ本文の別の記録
    c = _save(mem, "旧 " + uuid.uuid4().hex)
    d = _save(mem, "新 " + uuid.uuid4().hex)
    assert mem._observations.mark_superseded(c, d) is True
    got = _pending_of(mem, [a, b, c, d])
    assert got[frozenset({a, b})] == body
    assert frozenset({d}) in got
    assert all(c not in k for k in got)  # 隠された記録は抜かない


def test_writing_keys_marks_the_records_and_they_leave_pending():
    mem = _mem()
    body = "夏祭り " + uuid.uuid4().hex
    a = _save(mem, body, direction="発話")
    b = _save(mem, body, direction="会話")
    e = _save(mem, "挨拶だけ " + uuid.uuid4().hex)
    srcs = [s for s in mem.fact_pending(_ALL) if set(s.obs_ids) & {a, b, e}]
    facts = [[Fact("町内会", "夏祭りの日", "8月3日")] if a in s.obs_ids else [] for s in srcs]
    # (書けた本文の種類, 書いた行の数)。行は 2 記録 × 鍵の文 2 つ。
    assert mem.write_fact_keys(srcs, facts, model="haiku") == (len(srcs), 4)
    keys = _rows(
        "SELECT obs_id, side, sentence FROM obs_fact_keys WHERE obs_id = ANY(%s)", [a, b, e]
    )
    assert {(k["obs_id"], k["side"], k["sentence"]) for k in keys} == {
        (a, "答え", "町内会の夏祭りの日"),
        (a, "主語", "夏祭りの日が「8月3日」であるもの"),
        (b, "答え", "町内会の夏祭りの日"),
        (b, "主語", "夏祭りの日が「8月3日」であるもの"),
    }
    runs = {
        r["obs_id"]: r
        for r in _rows("SELECT * FROM obs_fact_runs WHERE obs_id = ANY(%s)", [a, b, e])
    }
    assert runs[a]["n_facts"] == runs[b]["n_facts"] == 1 and runs[e]["n_facts"] == 0
    assert runs[a]["model"] == "haiku"
    assert not _pending_of(mem, [a, b, e])


def test_when_ruri_fails_nothing_is_written_and_the_records_stay_pending():
    mem = _mem()
    a = _save(mem, "夏祭り " + uuid.uuid4().hex)
    srcs = [s for s in mem.fact_pending(_ALL) if a in s.obs_ids]
    with patch.object(_RuriModel, "encode_document", return_value=None):
        assert mem.write_fact_keys(srcs, [[Fact("s", "r", "a")]], model="m") is None
    assert not _rows("SELECT 1 FROM obs_fact_runs WHERE obs_id = %s", a)
    assert _pending_of(mem, [a])


def test_by_fact_keys_ranks_records_by_their_nearest_key_and_skips_hidden_and_excluded():
    mem = _mem()
    tag = uuid.uuid4().hex
    a = _save(mem, "本文は関係ない " + tag)
    b = _save(mem, "べつの本文 " + tag)
    srcs = [s for s in mem.fact_pending(_ALL) if set(s.obs_ids) & {a, b}]
    facts = [
        [Fact("町内会" + tag, "夏祭りの日", "8月3日")]
        if a in s.obs_ids
        else [Fact("x" + tag, "y", "z")]
        for s in srcs
    ]
    mem.write_fact_keys(srcs, facts, model="m")
    obs = mem._observations
    rows = obs.by_fact_keys(_q(f"町内会{tag}の夏祭りの日"), _BGE_Q, 10)
    assert rows[0]["id"] == a and abs(rows[0]["score"] - 1.0) < 1e-4
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))  # 鍵の文が 2 つでも 1 記録 1 行
    for k in ("facet_id", "timestamp", "groundedness_g0", "groundedness_n", "content", "direction"):
        assert k in rows[0]
    assert a not in [
        r["id"]
        for r in obs.by_fact_keys(_q(f"町内会{tag}の夏祭りの日"), _BGE_Q, 10, exclude_ids=[a])
    ]
    cos = obs.fact_key_cosines(_q(f"町内会{tag}の夏祭りの日"), [a, b, "無い"])
    assert abs(cos[a] - 1.0) < 1e-4 and b in cos and "無い" not in cos
    c = _save(mem, "新しい版 " + tag)
    assert obs.mark_superseded(a, c) is True
    assert a not in [r["id"] for r in obs.by_fact_keys(_q(f"町内会{tag}の夏祭りの日"), _BGE_Q, 10)]


def test_appending_to_a_body_drops_its_keys_so_it_is_extracted_again():
    """本文に書き足したら、古い本文から抜いた鍵の文と印を消す（足した事実も抜き直す）。"""
    mem = _mem()
    a = _save(mem, "夏祭り " + uuid.uuid4().hex)
    srcs = [s for s in mem.fact_pending(_ALL) if a in s.obs_ids]
    mem.write_fact_keys(srcs, [[Fact("s", "r", "a")]], model="m")
    assert not _pending_of(mem, [a])
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        assert mem._observations.append_and_reembed(a, "あとから足した一行") is True
    assert not _rows("SELECT 1 FROM obs_fact_keys WHERE obs_id = %s", a)
    assert _pending_of(mem, [a])


def test_records_seen_only_from_a_family_members_view_are_extracted_too():
    """抜き待ちは視点を問わない（想起はその人の視点で鍵の軸を引くので、どの面の記録にも鍵の文が要る）。"""
    mem = _mem()
    a = _save(mem, "夏祭り " + uuid.uuid4().hex)
    conn = psycopg2.connect(_DB_URL)
    try:
        other = "fact-keys-" + uuid.uuid4().hex
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                "VALUES (%s, %s, '', NOW(), NOW())",
                (other, other),
            )
            cur.execute("UPDATE situated_memories SET person_id = %s WHERE obs_id = %s", (other, a))
        conn.commit()
    finally:
        conn.close()
    assert _pending_of(mem, [a])


def test_a_body_changed_after_it_was_read_is_not_written():
    """抜いているあいだに本文が書き足されたら、古い本文の鍵の文を書かない（印も付けず、次に抜き直す）。"""
    mem = _mem()
    a = _save(mem, "夏祭り " + uuid.uuid4().hex)
    srcs = [s for s in mem.fact_pending(_ALL) if a in s.obs_ids]
    with patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]):
        assert mem._observations.append_and_reembed(a, "あとから足した一行") is True
    assert mem.write_fact_keys(srcs, [[Fact("s", "r", "a")]], model="m") == (0, 0)
    assert not _rows("SELECT 1 FROM obs_fact_runs WHERE obs_id = %s", a)
    assert _pending_of(mem, [a])


def test_fact_ready_says_whether_ruri_can_embed():
    mem = _mem()
    assert mem.fact_ready() is True
    with patch.object(_RuriModel, "ensure", return_value=None):
        assert mem.fact_ready() is False


def test_a_record_the_reply_skipped_is_not_marked():
    mem = _mem()
    tag = uuid.uuid4().hex
    a = _save(mem, "一つ目 " + tag)
    b = _save(mem, "二つ目 " + tag)
    srcs = [s for s in mem.fact_pending(_ALL) if set(s.obs_ids) & {a, b}]
    facts = [None if a in s.obs_ids else [Fact("s", "r", "a")] for s in srcs]
    assert mem.write_fact_keys(srcs, facts, model="m") == (1, 2)
    assert _pending_of(mem, [a]) and not _pending_of(mem, [b])
