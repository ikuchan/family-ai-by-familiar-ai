"""Tests for mood-c の nudge 接続（decay_and_nudge 純 と nudge_current_mood DB）。

課題5：評価器の後に、想起記憶＋現ターン感情＋自己認識 MI フラット項から N_PAD を作り、
mood を decay（updated_at からの経過）→ nudge → save する。空 items でもフラット項で
中立へ寄る。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from familiar_agent.db import get_db
from familiar_agent.mood_register import (
    MoodPAD,
    MOOD_STATE_KEY,
    compute_n_pad,
    decay_and_nudge,
    decay_to_rest,
    nudge_toward,
    nudge_current_mood,
)


# ── decay_and_nudge（純関数） ───────────────────────────────────────────────
def test_decay_and_nudge_composes_decay_then_nudge() -> None:
    mood = MoodPAD(0.9, 0.1, 0.8, 0.9)
    items = [(MoodPAD(0.2, 0.8, 0.6, 0.3), 2.0)]
    elapsed = 600.0  # 半減期1つぶん
    got = decay_and_nudge(mood, elapsed, items)
    expected = nudge_toward(decay_to_rest(mood, elapsed), compute_n_pad(items))
    assert got == expected


def test_decay_and_nudge_zero_elapsed_no_decay() -> None:
    mood = MoodPAD(0.7, 0.2, 0.6, 0.55)
    items = [(MoodPAD(0.5, 0.5, 0.5, 0.5), 1.0)]
    got = decay_and_nudge(mood, 0.0, items)
    expected = nudge_toward(mood, compute_n_pad(items))  # decay なし
    assert got == expected


# ── nudge_current_mood（DB・自己接続） ──────────────────────────────────────
def _set_mood_with_updated_at(mood: MoodPAD, updated_at: datetime) -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s,%s,%s) "
                "ON CONFLICT (state_key) DO UPDATE SET value_json=EXCLUDED.value_json, "
                "updated_at=EXCLUDED.updated_at",
                (MOOD_STATE_KEY, json.dumps(mood.to_json_dict()), updated_at.isoformat()),
            )
        conn.commit()


def _load_raw() -> MoodPAD:
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute("SELECT value_json FROM agent_state WHERE state_key=%s", (MOOD_STATE_KEY,))
            row = cur.fetchone()
    return MoodPAD.from_json_dict(json.loads(row[0]))


def test_nudge_current_mood_decays_nudges_and_saves() -> None:
    # 非中立 mood を1時間前に保存（十分減衰）
    start = MoodPAD(0.9, 0.1, 0.8, 0.9)
    _set_mood_with_updated_at(start, datetime.now(timezone.utc) - timedelta(hours=1))
    items = [(MoodPAD(0.2, 0.8, 0.6, 0.3), 2.0)]

    got = nudge_current_mood(items)

    # 保存され、返り値と一致
    assert _load_raw() == got
    # 1時間の減衰でほぼ中立まで戻ってから nudge されるので、開始値そのままではない
    assert got != start


def test_nudge_current_mood_empty_items_pulls_toward_the_rest_point() -> None:
    _set_mood_with_updated_at(MoodPAD(0.9, 0.1, 0.9, 0.1), datetime.now(timezone.utc))
    got = nudge_current_mood([])
    # W が空なら N_PAD は自己認識 MI だけ＝軸ごとの戻り先 (0.10, 0.10, 0.50, 0.50)。
    # 戻り先より上の軸は下がり、下の軸は上がる。
    assert got.p < 0.9      # 0.10 へ向かって下がる
    assert got.a < 0.9      # 0.50 へ向かって下がる
    assert got.dom > 0.1    # 0.50 へ向かって上がる
    assert got.pn <= 0.1    # すでに戻り先にあるので動かない（上がらない）
