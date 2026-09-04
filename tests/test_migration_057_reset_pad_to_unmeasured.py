"""057（旧目盛りの PAD を未測定へ戻す）が効いていることを確かめる。

案A で快と不快の目盛りを「0＝無い」に揃えた（`emotion_pad.LABEL_PAD["neutral"]` と
mood の戻り先を移した）。既存行の P/Pn/Dom は旧目盛りで書かれており、機械的に変換
できない。旧値には「0＝無い」と「0.5＝中立」の二つの読みが混ざっているためで、
その混在こそが目盛りを直した理由である。

そこで全行を**未測定**へ戻す。050 が定めた形（`emotion_p`／`emotion_pn`／`emotion_dom`
と `emotion_vec` を NULL・ラベルは `neutral`）にそろえるので、DB は一貫した状態のまま
移る。測り直しは別の道具が担う（マイグレーションから軽量LLM を呼ばない。テスト DB でも
走るため、API 鍵と課金と数分の待ちがテストに入り込む）。

**A（高ぶり）は触らない。** 機械値で、内容の新規性から作る。評価器へ渡していないので
目盛りの変更と関係が無く、6433 行すべてに残っている。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    return conn


def _insert(cur, *, p, pn, dom, a, emotion, vec):
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations "
        "(id, content, timestamp, direction, kind, emotion,"
        " emotion_p, emotion_pn, emotion_a, emotion_dom, emotion_vec) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (oid, "057 の検査用", datetime.now(timezone.utc), "自分", "observation",
         emotion, p, pn, a, dom, vec),
    )
    return oid


def test_the_old_scale_pad_is_cleared_and_arousal_survives():
    from importlib import util

    path = "migration/2026-09-05-057_reset_pad_to_unmeasured.py"
    spec = util.spec_from_file_location("m057", path)
    mod = util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    conn = _conn()
    with conn.cursor() as cur:
        filled = _insert(cur, p=0.5, pn=0.5, dom=0.5, a=0.5,
                         emotion="neutral", vec="[0,0,0,0]")
        measured = _insert(cur, p=0.8, pn=0.15, dom=0.6, a=0.72,
                           emotion="happy", vec="[1,2,3,4]")
        below = _insert(cur, p=0.6, pn=0.4, dom=0.5, a=0.1,
                        emotion="curious", vec="[1,1,1,1]")

    mod.upgrade(conn)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, emotion, emotion_p, emotion_pn, emotion_dom, emotion_a, emotion_vec "
            "FROM observations WHERE id = ANY(%s)", ([filled, measured, below],))
        rows = {r["id"]: r for r in cur.fetchall()}

    for oid, was_a in ((filled, 0.5), (measured, 0.72), (below, 0.1)):
        r = rows[oid]
        assert r["emotion_p"] is None, oid
        assert r["emotion_pn"] is None, oid
        assert r["emotion_dom"] is None, oid
        assert r["emotion_vec"] is None, oid
        assert r["emotion"] == "neutral", oid
        # A は機械値なので残る（050）。ここが落ちたら高ぶりを巻き込んで消している。
        assert abs(r["emotion_a"] - was_a) < 1e-9, oid

    with conn.cursor() as cur:
        cur.execute("DELETE FROM observations WHERE id = ANY(%s)",
                    ([filled, measured, below],))
    conn.close()
