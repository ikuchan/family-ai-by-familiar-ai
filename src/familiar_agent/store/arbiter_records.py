"""調停ごとの記録の器（出-ay 段 3・2026-10-08・本人の決定）。表 `arbiter_records`（074）。

調停のたびに、Jev に渡したそのままの文（`state`・W を含む）・問い・答え・結末を残す。正解（`JEV_正解.md`）の時刻から
`near` で引き、前提を足して投げ直すのに使う。記-o（W に何が載っていたか）もこれで見る。

- **書くのは別スレッド**（`record`）。調停を待たせない。失敗しても調停は止めず、警告を 1 行出すだけ（中身は出さない）。
- 会話の中身はログに出さない。DB の中だけに置く。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

_pending: "set[asyncio.Task[None]]" = set()


def _insert(
    *, origin: str, utterance: str, state: str, questions: Any, answer: Any, outcome: str
) -> None:
    from ..db import get_db

    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO arbiter_records (origin, utterance, state, questions, answer, outcome) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s)",
                (
                    origin,
                    utterance,
                    state,
                    json.dumps(questions, ensure_ascii=False, default=str),
                    json.dumps(answer, ensure_ascii=False, default=str),
                    outcome,
                ),
            )
        conn.commit()


def record(**fields: Any) -> None:
    """1 回の調停を残す（別スレッドで書く）。走っているループが無ければ、その場で書く。"""

    def _write() -> None:
        try:
            _insert(**fields)
        except Exception as e:  # noqa: BLE001
            logger.warning("調停の記録を残せなかった：%s", type(e).__name__)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _write()
        return
    task = loop.create_task(asyncio.to_thread(_write))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def flush() -> None:
    """書きかけを待つ（試験と、閉じる前）。"""
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)


def near(at: datetime, *, seconds: float) -> "list[dict]":
    """`at` から `seconds` 秒のあいだの記録を、古い順に引く（正解の時刻から、そのときの調停を探す）。"""
    import psycopg2.extras

    from ..db import get_db

    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, at, origin, utterance, state, questions, answer, outcome "
                "FROM arbiter_records WHERE at >= %s AND at < %s ORDER BY at",
                (at, at + timedelta(seconds=seconds)),
            )
            rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
    return rows
