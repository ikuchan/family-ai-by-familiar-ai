"""Capability manifest loader and AI self-understanding storage.

The agent periodically reads capabilities.yaml, asks the LLM to write
a first-person capability summary, and stores it in agent_state.
That summary is injected into the variable system prompt each turn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import psycopg2.extras

from .db import get_db

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent.parent / "capabilities.yaml"
_STATE_KEY = "capability_summary"
_REFRESH_EVERY_N_TURNS = 50


def load_manifest() -> str:
    """Return raw YAML text of capabilities.yaml, or empty string if missing."""
    try:
        return _MANIFEST_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read capabilities.yaml: %s", e)
        return ""


def load_summary() -> str:
    """Return the AI-written capability summary from agent_state, or ''."""
    try:
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT value_json FROM agent_state WHERE state_key = %s",
                    (_STATE_KEY,),
                )
                row = cur.fetchone()
        if row:
            import json
            return str(json.loads(row["value_json"]))
    except Exception as e:
        logger.warning("Could not load capability summary: %s", e)
    return ""


def save_summary(text: str) -> None:
    """Persist the AI-written capability summary to agent_state."""
    try:
        import json
        now = datetime.now(timezone.utc).isoformat()
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at)"
                    " VALUES (%s, %s, %s)"
                    " ON CONFLICT (state_key) DO UPDATE"
                    "   SET value_json = EXCLUDED.value_json,"
                    "       updated_at = EXCLUDED.updated_at",
                    (_STATE_KEY, json.dumps(text), now),
                )
            conn.commit()
    except Exception as e:
        logger.warning("Could not save capability summary: %s", e)


def should_refresh(turn_index: int) -> bool:
    """True on turn 0 (no summary yet) or every N turns thereafter."""
    if turn_index == 0:
        return not bool(load_summary())
    return turn_index % _REFRESH_EVERY_N_TURNS == 0
