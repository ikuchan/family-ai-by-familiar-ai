"""World model — persistent scene entity tracking and change detection.

Phase 1 of the familiar-ai roadmap.

Architecture:
- extract_entities(): calls the utility backend to parse a scene description
  into a structured list of {label, category, confidence} dicts.
- SceneTracker: PostgreSQL-backed tracker that maintains the current entity set,
  diffs it against the previous set on every update(), emits change events,
  and formats a compact context block for LLM injection.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .core.helpers import strip_code_fence
from .db import Database

if TYPE_CHECKING:
    from .prediction import PredictionEngine
    from .workspace import Coalition

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """\
You are a scene-analysis assistant. Given a description of what an AI agent sees,
extract the distinct entities (people, objects, locations/features) and return them
as JSON with the key "entities".

Each entity must have:
  - "label": short singular noun ("person", "chair", "window")
  - "category": one of "person", "object", "location"
  - "confidence": float 0.0–1.0

Return ONLY the JSON object. Example:
{
  "entities": [
    {"label": "chair", "category": "object", "confidence": 0.9},
    {"label": "person", "category": "person", "confidence": 0.8}
  ]
}
"""


async def extract_entities(
    description: str, backend: Any, image_b64: str | None = None
) -> list[dict]:
    """Call the utility backend to extract structured entities from a scene description.

    When image_b64 is provided and the backend supports complete_with_image(),
    the raw camera image is sent directly to the backend (useful for local VLMs via
    Ollama) and description is used only as fallback.

    Returns a list of dicts with keys: label, category, confidence.
    Returns [] on any parse error.
    """
    try:
        if image_b64 and hasattr(backend, "complete_with_image"):
            vision_prompt = (
                f"{_EXTRACT_SYSTEM}\n\n"
                "Analyze this camera image directly and extract all visible entities."
            )
            raw = await backend.complete_with_image(vision_prompt, image_b64)
            if not raw:
                # Fall back to text description if vision call returned nothing
                raw = await backend.complete(
                    f"{_EXTRACT_SYSTEM}\n\nScene description:\n{description}"
                )
        else:
            raw = await backend.complete(
                f"{_EXTRACT_SYSTEM}\n\nScene description:\n{description}"
            )
        # 返答は素の JSON とは限らない。実機の VLM は同じ画像でも回ごとに
        # ```json で包んだり包まなかったりする。
        data = json.loads(strip_code_fence(str(raw)))
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            _log_unusable(raw, "entities が配列でない")
            return []
        return [e for e in entities if isinstance(e, dict) and "label" in e]
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        _log_unusable(raw, str(exc))
        return []


# 記録に載せる返答の長さ。情景の説明は長く、全文を warning で出すとログが埋まる。
_REPLY_HEAD_CHARS = 200


def _log_unusable(raw: Any, reason: str) -> None:
    """意味づけに使えない返答を、先頭を添えて残す。

    実機で `see` が「意味づけは何も返さなかった」を出し続けたとき、失敗が debug で
    例外の型しか出ておらず、**VLM が何を返したのかが分からなかった**。空文字なのか、
    説明文なのか、JSON もどきなのかで対応が変わる。
    """
    text = "" if raw is None else str(raw)
    head = text[:_REPLY_HEAD_CHARS] if text.strip() else "（空）"
    logger.warning("情景の意味づけに使えない返答（%s）：%s", reason, head)


class SceneTracker:
    """Persistent scene entity tracker backed by PostgreSQL.

    Usage:
        tracker = SceneTracker(db)
        events = await tracker.update(description, backend)
        context = tracker.context_for_prompt()
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._current_entities: dict[str, dict] = {}  # label → entity dict
        self._load_current_entities()

    @staticmethod
    def _init_schema(db: Database) -> None:
        """Create tables if not present (idempotent). Used by tests and __init__."""
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_entities (
                    entity_id   TEXT PRIMARY KEY,
                    label       TEXT NOT NULL,
                    category    TEXT NOT NULL DEFAULT 'object',
                    first_seen  TEXT NOT NULL,
                    last_seen   TEXT NOT NULL,
                    confidence  REAL NOT NULL DEFAULT 0.8,
                    bbox_hint   TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_events (
                    event_id     TEXT PRIMARY KEY,
                    event_type   TEXT NOT NULL,
                    entity_id    TEXT,
                    entity_label TEXT NOT NULL,
                    timestamp    TEXT NOT NULL
                )
                """
            )
        conn.commit()

    def _load_current_entities(self) -> None:
        """Restore current entity set from DB (last-seen entities)."""
        try:
            with self._db.lock:
                conn = self._db.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT label, category, confidence, entity_id FROM scene_entities"
                    )
                    rows = cur.fetchall()
            self._current_entities = {
                r["label"]: {
                    "label": r["label"],
                    "category": r["category"],
                    "confidence": r["confidence"],
                    "entity_id": r["entity_id"],
                }
                for r in rows
            }
        except Exception:
            self._current_entities = {}

    async def update(
        self,
        description: str,
        backend: Any,
        prediction_engine: PredictionEngine | None = None,
        action_name: str | None = None,
        action_input: dict[str, Any] | None = None,
        image_b64: str | None = None,
    ) -> list[dict]:
        """Extract entities from description, detect changes, persist, return events.

        Returns a list of event dicts: {event_type, entity_label, entity_id}.

        When image_b64 is provided, it is passed to extract_entities() so that
        a vision-capable backend (e.g. local Ollama VLM) can process the raw
        camera frame instead of relying on the main backend's text description.

        If prediction_engine is provided, compute prediction error against
        the new observation and update the model.
        """
        if prediction_engine is not None and action_name:
            prediction_engine.record_action(action_name, action_input or {})

        new_entities = await extract_entities(description, backend, image_b64=image_b64)
        events = self._diff_entities(new_entities)
        self._persist_entities(new_entities)
        self._persist_events(events)
        self._current_entities = {e["label"]: e for e in new_entities}

        if prediction_engine is not None:
            labels = [e["label"] for e in new_entities]
            prediction_engine.compute_error(labels)
            prediction_engine.update(labels)
        return events

    def _diff_entities(self, new_entities: list[dict]) -> list[dict]:
        new_labels = {e["label"] for e in new_entities}
        old_labels = set(self._current_entities.keys())

        events = []
        for label in new_labels - old_labels:
            events.append({"event_type": "appeared", "entity_label": label, "entity_id": None})
        for label in old_labels - new_labels:
            old = self._current_entities[label]
            events.append(
                {
                    "event_type": "disappeared",
                    "entity_label": label,
                    "entity_id": old.get("entity_id"),
                }
            )
        return events

    def _persist_entities(self, entities: list[dict]) -> None:
        now = datetime.now().isoformat()
        with self._db.lock:
            conn = self._db.conn()
            with conn.cursor() as cur:
                for entity in entities:
                    label = entity["label"]
                    existing = self._current_entities.get(label)
                    if existing and existing.get("entity_id"):
                        entity_id = existing["entity_id"]
                        cur.execute(
                            "UPDATE scene_entities SET last_seen=%s, confidence=%s"
                            " WHERE entity_id=%s",
                            (now, entity.get("confidence", 0.8), entity_id),
                        )
                    else:
                        entity_id = str(uuid.uuid4())
                        entity["entity_id"] = entity_id
                        cur.execute(
                            """
                            INSERT INTO scene_entities
                                (entity_id, label, category, first_seen, last_seen, confidence)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                entity_id,
                                label,
                                entity.get("category", "object"),
                                now,
                                now,
                                entity.get("confidence", 0.8),
                            ),
                        )
            conn.commit()

    def _persist_events(self, events: list[dict]) -> None:
        if not events:
            return
        now = datetime.now().isoformat()
        with self._db.lock:
            conn = self._db.conn()
            with conn.cursor() as cur:
                for event in events:
                    cur.execute(
                        """
                        INSERT INTO scene_events
                            (event_id, event_type, entity_id, entity_label, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            event["event_type"],
                            event.get("entity_id"),
                            event["entity_label"],
                            now,
                        ),
                    )
            conn.commit()

    def context_for_prompt(self, n: int = 10) -> str:
        """Return a compact scene summary for LLM system prompt injection."""
        if not self._current_entities:
            return ""

        lines = ["[Current scene]"]
        people = [e for e in self._current_entities.values() if e.get("category") == "person"]
        objects = [e for e in self._current_entities.values() if e.get("category") == "object"]
        locations = [e for e in self._current_entities.values() if e.get("category") == "location"]

        if people:
            labels = ", ".join(e["label"] for e in people[:n])
            lines.append(f"  People: {labels}")
        if objects:
            labels = ", ".join(e["label"] for e in objects[:n])
            lines.append(f"  Objects: {labels}")
        if locations:
            labels = ", ".join(e["label"] for e in locations[:n])
            lines.append(f"  Locations: {labels}")

        return "\n".join(lines)

    def recent_events(self, n: int = 5) -> list[dict]:
        """Return the n most recent scene events from the DB."""
        with self._db.lock:
            conn = self._db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_type, entity_label, timestamp
                    FROM scene_events
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (n,),
                )
                rows = cur.fetchall()
        return [
            {
                "event_type": r["event_type"],
                "entity_label": r["entity_label"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def as_coalition(self) -> Coalition | None:
        """Return a workspace Coalition from the current scene state."""
        from .workspace import Coalition

        if not self._current_entities:
            return None

        # Urgency rises when people appeared recently
        events = self.recent_events(n=3)
        appeared_people = [
            e
            for e in events
            if e["event_type"] == "appeared"
            and any(
                ent.get("category") == "person" and ent["label"] == e["entity_label"]
                for ent in self._current_entities.values()
            )
        ]
        urgency = 0.8 if appeared_people else 0.2
        novelty = min(1.0, len(events) * 0.2)

        entity_count = len(self._current_entities)
        summary = f"{entity_count} entities in scene"
        context = self.context_for_prompt()
        avg_conf = (
            sum(e.get("confidence", 0.8) for e in self._current_entities.values()) / entity_count
        )

        return Coalition(
            source="scene",
            summary=summary,
            dynamism=avg_conf,
            urgency=urgency,
            novelty=novelty,
            context_block=context,
        )
