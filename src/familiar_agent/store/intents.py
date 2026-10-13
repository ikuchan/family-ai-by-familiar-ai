"""意図の台帳の記録の器（出-cc・2026-10-13・本人の決定）。表 `intents`（076）。

意図（調べもの・道具・主LLM の答え）ごとに uuid を振り、完了は uuid で引く（引き当てはメモリ上の `loop/intent_ledger`）。
ここはあとから追うための記録で、投げたときに 1 行足し、終わったときに状態・失敗の印・結果の字数・終わった時刻を書く。
結果の中身は持たない（O にあり、家族の話も含む）。

- **書くのは別スレッドで、1 本の列に並べる**。終わりの更新が始まりの追加より先に走ると 0 行の更新になるので、前の書き込みを
  待ってから書く。引き当てを待たせない。失敗しても止めず、警告を 1 行出すだけ。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

#: 意図の状態。
OPEN = "開いている"
RETURNED = "結果が届いた"
ABORTED = "打ち切った"
CLOSED = "求めと一緒に閉じた"
LATE = "閉じた後に届いた"

_last: "asyncio.Task[None] | None" = None
_pending: "set[asyncio.Task[None]]" = set()


def _insert(*, intent_id: str, request_id: str, origin: str, action: str, query: str) -> None:
    from ..db import get_db

    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO intents (id, request_id, origin, action, query, state) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (intent_id, request_id, origin, action, query, OPEN),
            )
        conn.commit()


def _update(*, intent_id: str, state: str, failed: bool, result_chars: "int | None") -> None:
    from ..db import get_db

    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE intents SET state = %s, failed = %s, result_chars = %s, ended_at = now() "
                "WHERE id = %s",
                (state, failed, result_chars, intent_id),
            )
        conn.commit()


def _write(fn, kwargs: dict) -> None:
    """1 件書く。走っているループがあれば前の書き込みのあとに別スレッドで、無ければその場で。"""
    global _last

    def _do() -> None:
        try:
            fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.warning("意図の記録を残せなかった：%s", type(e).__name__)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _do()
        return
    before = _last if _last is not None and _last.get_loop() is loop else None

    async def _after() -> None:
        if before is not None:
            await asyncio.gather(before, return_exceptions=True)
        await asyncio.to_thread(_do)

    task = loop.create_task(_after())
    _last = task
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def opened(intent_id: str, *, request_id: str, origin: str, action: str, query: str) -> None:
    """意図を投げた。"""
    _write(
        _insert,
        {
            "intent_id": intent_id,
            "request_id": request_id,
            "origin": origin,
            "action": action,
            "query": query,
        },
    )


def ended(
    intent_id: str, *, state: str, failed: bool = False, result_chars: "int | None" = None
) -> None:
    """意図が終わった（結果が届いた・打ち切った・求めと一緒に閉じた・閉じた後に届いた）。"""
    _write(
        _update,
        {"intent_id": intent_id, "state": state, "failed": failed, "result_chars": result_chars},
    )


async def flush() -> None:
    """書きかけを待つ（試験と、閉じる前）。"""
    if _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
