"""Tests for mood_register.load_current_mood（自己接続の mood 読み出し・W2b-2）。

他状態モジュールと同じく get_db() で自己接続し、agent_state の mood_pad を読む。
行が無ければ中立。W2b-2 は mood を読むだけ（更新は mood スライス）。
"""

from __future__ import annotations

from familiar_agent.db import get_db
from familiar_agent.mood_register import MoodPAD, load_current_mood, save_mood, MOOD_STATE_KEY


def _delete_mood_row() -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM agent_state WHERE state_key = %s", (MOOD_STATE_KEY,))
        conn.commit()


def test_load_current_mood_returns_saved() -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        save_mood(conn, MoodPAD(0.7, 0.2, 0.35, 0.5))
        conn.commit()
    got = load_current_mood()
    assert got == MoodPAD(0.7, 0.2, 0.35, 0.5)


def test_load_current_mood_neutral_when_absent() -> None:
    _delete_mood_row()
    assert load_current_mood() == MoodPAD()
