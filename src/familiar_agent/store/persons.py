"""人物レジストリ（`persons`）。

家族それぞれを識別する表。視点ベクトル（`perspective_vec`）は 045 で落とした
（ベクトルの差は人でなく関係が作る）。
観測とは別の表なので、持ち主を分けてある。

使うものは文脈（`StoreContext`）から受け取る。
"""

from __future__ import annotations

import logging
import uuid

from . import clock
from .context import StoreContext

logger = logging.getLogger(__name__)


class PersonRegistry:
    """人物レジストリの持ち主。"""

    def __init__(self, ctx: StoreContext) -> None:
        self._ctx = ctx

    def register_person(
        self, name: str, display_name: str = "", person_id: str | None = None
    ) -> str:
        pid = person_id or str(uuid.uuid4())
        now = clock.now_utc_iso()
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM persons WHERE name = %s", (name,))
                row = cur.fetchone()
                if row:
                    return str(row["id"])
                cur.execute(
                    "INSERT INTO persons (id, name, display_name, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (pid, name, display_name or name, now, now),
                )
            conn.commit()
        return pid

    def update_display_name(self, person_id: str, display_name: str) -> bool:
        """呼び方を書き直す（知-af）。登録は同じ名前の行があれば呼び方を書き直さないので、`FAMILY.md` に合わせる口。"""
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE persons SET display_name = %s, updated_at = %s WHERE id = %s",
                    (display_name, clock.now_utc_iso(), person_id),
                )
                changed = cur.rowcount > 0
            conn.commit()
        return changed

    def list_persons(self) -> list[dict]:
        with self._ctx.lock:
            conn = self._ctx.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, display_name, created_at FROM persons ORDER BY created_at"
                )
                return [dict(r) for r in cur.fetchall()]
