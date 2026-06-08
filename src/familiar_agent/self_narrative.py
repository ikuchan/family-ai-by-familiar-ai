"""Self-narrative — persistent first-person diary of Kokone's sessions."""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import psycopg2.extras

from .db import get_db

if TYPE_CHECKING:
    from .workspace import Coalition

logger = logging.getLogger(__name__)


class NarrativeEntry(NamedTuple):
    date: str
    text: str
    mood: str
    trigger: str


class SelfNarrative:
    """Persists a rolling diary of session-closing self-descriptions in PostgreSQL."""

    def __init__(self, path: Path | None = None):
        pass  # path ignored; kept for call-site compatibility

    def write(self, text: str, mood: str = "neutral", trigger: str = "session_close") -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        today = date.today().isoformat()
        recent = self.read_recent(n=1)
        if recent and recent[-1].date == today and recent[-1].text == cleaned:
            return
        try:
            db = get_db()
            now = datetime.utcnow().isoformat()
            with db.lock:
                conn = db.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO self_narrative_log (date, text, mood, trigger, created_at)"
                        " VALUES (%s, %s, %s, %s, %s)",
                        (today, cleaned, mood, trigger, now),
                    )
                conn.commit()
        except Exception as e:
            logger.warning("Could not write self narrative: %s", e)

    def read_recent(self, n: int = 3) -> list[NarrativeEntry]:
        if n <= 0:
            return []
        try:
            db = get_db()
            with db.lock:
                conn = db.conn()
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        "SELECT date, text, mood, trigger FROM self_narrative_log"
                        " ORDER BY id DESC LIMIT %s",
                        (n,),
                    )
                    rows = cur.fetchall()
            return [
                NarrativeEntry(date=r["date"], text=r["text"], mood=r["mood"], trigger=r["trigger"])
                for r in reversed(rows)
            ]
        except Exception as e:
            logger.warning("Could not read self narrative: %s", e)
            return []

    def context_for_prompt(self) -> str | None:
        entries = self.read_recent(n=3)
        if not entries:
            return None
        lines = [f"[{e.date}] {e.text}" for e in entries]
        return "過去のウチからの続き:\n" + "\n".join(lines)

    def as_coalition(self) -> Coalition | None:
        from .workspace import Coalition

        context = self.context_for_prompt()
        if not context:
            return None
        entries = self.read_recent(n=3)
        latest = entries[-1] if entries else None
        summary = latest.text[:80] if latest else "self-narrative"
        return Coalition(
            source="narrative",
            summary=summary,
            activation=0.4,
            urgency=0.1,
            novelty=0.1,
            context_block=context,
        )
