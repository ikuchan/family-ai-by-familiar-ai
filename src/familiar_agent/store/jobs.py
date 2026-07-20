"""非同期の書き込みキュー（`memory_events` と `memory_jobs`）。

記憶の書き込みは、いきなり `observations` へ入らず、いったんイベントとして積んで
から実体化される（[D-O書込]：O は追記＝イベントログ）。重い処理（埋め込みの生成
など）を応答の経路から外す狙いもある。

    save(materialize_now=False)
      → memory_events に追記    （何を書くか）
      → memory_jobs に積む      （いつ実体化するか）
      → claim_pending_jobs で拾って materialize → observations に現れる

この2テーブルを触るのはこのモジュールだけにする。実体化の本体
（`_materialize_save_event`）は observations 側の仕事なので、宿主から借りる。
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable

from . import clock

logger = logging.getLogger(__name__)


class MemoryJobsMixin:
    """キューの持ち主。`ObservationMemory` が継承する。

    宿主から借りる道具を下に宣言してある。これがこの層の依存の全てである。
    """

    # 宿主（ObservationMemory）が備えるもの。mixin 自身は持たない。
    _db_lock: threading.Lock
    _person_id: str

    def _ensure_connected(self) -> Any: ...  # 宿主が実装する

    @staticmethod
    def _now() -> str:
        """宿主が実装する（TEXT 列向けの現在時刻）。"""
        raise NotImplementedError

    # 実体化の本体は観測層（ObservationWriteMixin）が持つ。ここでは型の宣言だけに
    # とどめる。メソッドとして定義すると MRO で本物より先に見つかり、実行時に
    # こちらが呼ばれてしまう。
    _materialize_save_event: Callable[..., bool]

    def _enqueue_job(self, conn, event_id: str, job_type: str, now: str) -> bool:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_jobs "
                    "(job_id,event_id,job_type,status,attempts,available_at,last_error,created_at,updated_at) "
                    "VALUES (%s,%s,%s,'pending',0,%s,NULL,%s,%s)",
                    (str(uuid.uuid4()), event_id, job_type, now, now, now),
                )
            return True
        except Exception:
            return False

    def append_memory_event(
        self,
        event_type: str,
        payload: dict,
        dedupe_key: str | None = None,
        queue_job: bool = True,
        job_type: str = "materialize_observation",
    ) -> tuple[str | None, bool]:
        now = self._now()
        payload_json = json.dumps(payload, ensure_ascii=False, separators=(",",":"), sort_keys=True)
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                if dedupe_key:
                    with conn.cursor() as cur:
                        cur.execute("SELECT event_id FROM memory_events WHERE dedupe_key = %s", (dedupe_key,))
                        row = cur.fetchone()
                    if row:
                        eid = str(row["event_id"])
                        if queue_job:
                            self._enqueue_job(conn, eid, job_type, now)
                        conn.commit()
                        return eid, False
                eid = str(uuid.uuid4())
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_events (event_id,created_at,event_type,dedupe_key,payload_json,person_id) "
                        "VALUES (%s,%s,%s,%s,%s,%s)",
                        (eid, now, event_type, dedupe_key, payload_json, self._person_id),
                    )
                if queue_job:
                    self._enqueue_job(conn, eid, job_type, now)
                conn.commit()
                return eid, True
        except Exception as e:
            logger.warning("append_memory_event failed: %s", e)
            return None, False

    async def append_memory_event_async(self, *a, **kw):
        return await asyncio.to_thread(self.append_memory_event, *a, **kw)

    def claim_pending_jobs(self, limit: int = 10) -> list[dict]:
        now = self._now()
        claimed = []
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT j.job_id,j.event_id,j.job_type,j.attempts, "
                    "e.event_type,e.payload_json "
                    "FROM memory_jobs j JOIN memory_events e ON e.event_id = j.event_id "
                    "WHERE j.status='pending' AND j.available_at <= %s "
                    "ORDER BY j.created_at LIMIT %s",
                    (now, limit),
                )
                rows = cur.fetchall()
            for row in rows:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memory_jobs SET status='running',attempts=attempts+1,updated_at=%s "
                        "WHERE job_id=%s AND status='pending' RETURNING job_id",
                        (now, row["job_id"]),
                    )
                    if cur.rowcount != 1:
                        continue
                try:
                    payload = json.loads(row["payload_json"])
                except Exception:
                    payload = {"raw_payload": row["payload_json"]}
                claimed.append({
                    "job_id":     row["job_id"],
                    "event_id":   row["event_id"],
                    "job_type":   row["job_type"],
                    "attempts":   int(row["attempts"]) + 1,
                    "event_type": row["event_type"],
                    "payload":    payload,
                })
            conn.commit()
        return claimed

    def mark_job_done(self, job_id: str) -> bool:
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memory_jobs SET status='done',updated_at=%s,last_error=NULL WHERE job_id=%s",
                    (self._now(), job_id),
                )
            conn.commit()
            return True

    def mark_job_failed(self, job_id: str, error: str, retry_delay: float = 10.0, max_attempts: int = 3) -> str:
        now = datetime.fromisoformat(clock.now_local_iso())
        with self._db_lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("SELECT attempts FROM memory_jobs WHERE job_id=%s", (job_id,))
                row = cur.fetchone()
            if row is None:
                return "missing"
            attempts = int(row["attempts"])
            status = "dead_letter" if attempts >= max_attempts else "pending"
            avail = now.isoformat() if status == "dead_letter" else \
                    (now + timedelta(seconds=max(retry_delay, 0.0))).isoformat()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memory_jobs SET status=%s,available_at=%s,last_error=%s,updated_at=%s WHERE job_id=%s",
                    (status, avail, error[:500], now.isoformat(), job_id),
                )
            conn.commit()
        return status

    # ── Core save ──────────────────────────────────────────────────────────

    def materialize_event(self, event_id: str) -> bool:
        try:
            with self._db_lock:
                conn = self._ensure_connected()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT event_type, payload_json FROM memory_events WHERE event_id=%s",
                        (event_id,),
                    )
                    row = cur.fetchone()
            if not row:
                return False
            payload = json.loads(row["payload_json"])
            if row["event_type"] == "memory.save":
                return self._materialize_save_event(event_id, payload)
            return False
        except Exception as e:
            logger.warning("materialize_event failed: %s", e)
            return False

    # ── Recall ─────────────────────────────────────────────────────────────
