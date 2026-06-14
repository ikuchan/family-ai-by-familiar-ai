"""pending_speech ストア。

話したいことを覚えた記憶に紐づけて溜める。
observations とは独立し、想起系テーブルを汚染しない（フックI）。

鮮度切れ or 参照先 supersede で失効。意図が古びたら話さない。
reinforce_count は強化A用カラム。増やす契機の検出は別 Issue。本モジュールでは 0 固定。
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extras

from ..config import PendingSpeechConfig
from ..time_decay import DecayState

logger = logging.getLogger(__name__)


class PendingSpeechStore:
    """CRUD store for pending_speech table."""

    def __init__(self, database_url: str | None = None) -> None:
        self._url = database_url or os.environ.get(
            "DATABASE_URL",
            "postgresql://familiar_ai:familiar_ai@localhost:5432/familiar_ai",
        )
        self._conn: Any = None
        self._lock = threading.Lock()

    def _ensure_connected(self) -> Any:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                self._url, cursor_factory=psycopg2.extras.RealDictCursor
            )
        return self._conn

    def add(self, observation_id: str, target_person_id: str | None) -> str | None:
        """observation_id 実在チェック → INSERT。実在しなければ None（拒否）。"""
        with self._lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM observations WHERE id = %s", (observation_id,)
                )
                if cur.fetchone() is None:
                    conn.rollback()
                    return None
                pid = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO pending_speech (id, observation_id, target_person_id) "
                    "VALUES (%s, %s, %s)",
                    (pid, observation_id, target_person_id),
                )
            conn.commit()
            return pid

    def list_active(self) -> list[dict]:
        """全 pending を返す（observations の content/timestamp/superseded_by を JOIN）。"""
        with self._lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ps.id, ps.observation_id, ps.target_person_id,
                           ps.created_at, ps.reinforce_count,
                           o.content, o.timestamp, o.superseded_by
                    FROM pending_speech ps
                    JOIN observations o ON o.id = ps.observation_id
                    ORDER BY ps.created_at
                """)
                return [dict(r) for r in cur.fetchall()]

    def delete(self, pending_id: str) -> None:
        with self._lock:
            conn = self._ensure_connected()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pending_speech WHERE id = %s", (pending_id,))
            conn.commit()

    def freshness_score(self, row: dict, now_epoch: float, cfg: PendingSpeechConfig) -> float:
        """DecayState で鮮度スコアを計算する。

        origin = created_at, reinforce_count 反映, half_life = cfg.half_life_days * 86400。
        """
        created_at = row["created_at"]
        if isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            origin = created_at.timestamp()
        else:
            origin = now_epoch  # unknown → no decay
        state = DecayState(
            origin_epoch=origin,
            half_life_seconds=cfg.half_life_days * 86400.0,
            floor=cfg.floor,
            reinforce_count=int(row.get("reinforce_count", 0)),
        )
        return state.score(now_epoch)

    def is_expired(self, row: dict, score: float, cfg: PendingSpeechConfig) -> bool:
        """鮮度切れ or 参照先 supersede で失効と判定する。"""
        if score < cfg.expire_threshold:
            return True
        return row.get("superseded_by") is not None
