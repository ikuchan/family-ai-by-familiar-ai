"""撤去予定の意味・信念層と明示リンク（Phase 6 で撤去）。

`semantic_facts`／`behavior_policies`／`memory_revisions`／`memory_links` を触る
コードをここへ隔離する。設計上、意味と信念は O へ一元化され（[D-記憶単一化]）、
明示リンクは WR 拡散想起へ置き換わる（[D-WR拡散想起]）。いずれも新しい経路が通って
実証されてから撤去する（課題8 Phase 6）。

隔離の狙いは、撤去を「このファイルを消して、`ObservationMemory` の基底クラスから
1語外す」に近づけることにある。ここに新しい機能を足さない。

`ObservationMemory` に mixin として混ぜているのは、呼び出し側（`agent.py`）を
書き換えずに移すためである。移動と呼び出し側の変更を同時にやると、挙動不変の
検証が難しくなる。
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from ..store import clock
from ..store.db_compat import _RealDictConnWrapper

from ..store.context import StoreContext

logger = logging.getLogger(__name__)


class LegacySemanticLayer:
    """撤去予定の層（Phase 6）。

    使うものは文脈（`StoreContext`）から受け取る。ここに新しい機能を足さない。
    """

    def __init__(self, ctx: StoreContext) -> None:
        self._ctx = ctx

    def recall_semantic_facts(self, query: str, n: int = 5) -> list[dict]:
        like = f"%{query.strip()}%" if query.strip() else "%"
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT fact_key,fact_text,source_memory_id,confidence,tags,last_seen_at "
                        "FROM semantic_facts "
                        "WHERE person_id=%s AND (%s='%%' OR fact_text LIKE %s OR tags LIKE %s) "
                        "ORDER BY CASE WHEN fact_text LIKE %s THEN 0 ELSE 1 END, last_seen_at DESC "
                        "LIMIT %s",
                        (self._ctx.person_id, like, like, like, like, n),
                    )
                    return [
                        {"key":r["fact_key"],"summary":r["fact_text"],
                         "source_memory_id":r["source_memory_id"],
                         "confidence":float(r["confidence"]),"tags":r["tags"],
                         "last_seen_at":r["last_seen_at"]}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_semantic_facts failed: %s", e); return []

    async def recall_semantic_facts_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall_semantic_facts, *a, **kw)

    def _upsert_semantic_fact_locked(
        self,
        conn: "_RealDictConnWrapper",
        key: str,
        text: str,
        confidence: float = 0.5,
        source_memory_id: str | None = None,
        tags: str = "",
    ) -> None:
        """Upsert a semantic fact inside an already-held lock; record a revision if the text changes."""
        now = clock.now_local_iso()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, fact_text, confidence FROM semantic_facts "
                "WHERE person_id=%s AND fact_key=%s",
                (self._ctx.person_id, key),
            )
            existing = cur.fetchone()
        if existing and existing["fact_text"] != text:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_revisions "
                    "(id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                    "VALUES (%s,'semantic_fact',%s,%s,%s,%s,%s,%s,'upsert',%s)",
                    (str(uuid.uuid4()), key,
                     existing["fact_text"], text,
                     float(existing["confidence"]), confidence,
                     source_memory_id, now),
                )
        if existing:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE semantic_facts SET fact_text=%s,confidence=%s,"
                    "source_memory_id=%s,tags=%s,updated_at=%s,last_seen_at=%s "
                    "WHERE person_id=%s AND fact_key=%s",
                    (text, confidence, source_memory_id, tags, now, now,
                     self._ctx.person_id, key),
                )
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO semantic_facts "
                    "(id,fact_key,fact_text,source_memory_id,confidence,tags,"
                    "last_seen_at,created_at,updated_at,person_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), key, text, source_memory_id,
                     confidence, tags, now, now, now, self._ctx.person_id),
                )

    def _upsert_behavior_policy_locked(
        self,
        conn: "_RealDictConnWrapper",
        key: str,
        text: str,
        trigger_context: str = "",
        action_hint: str = "",
        confidence: float = 0.5,
        source_memory_id: str | None = None,
    ) -> None:
        now = clock.now_local_iso()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, policy_text, confidence FROM behavior_policies "
                "WHERE policy_key=%s AND person_id=%s",
                (key, self._ctx.person_id),
            )
            existing = cur.fetchone()
        if existing and existing["policy_text"] != text:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_revisions "
                    "(id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                    "VALUES (%s,'behavior_policy',%s,%s,%s,%s,%s,%s,'upsert',%s)",
                    (str(uuid.uuid4()), key,
                     existing["policy_text"], text,
                     float(existing["confidence"]), confidence,
                     source_memory_id, now),
                )
        if existing:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE behavior_policies "
                    "SET policy_text=%s,trigger_context=%s,action_hint=%s,"
                    "confidence=%s,source_memory_id=%s,updated_at=%s,last_seen_at=%s "
                    "WHERE policy_key=%s AND person_id=%s",
                    (text, trigger_context, action_hint, confidence, source_memory_id,
                     now, now, key, self._ctx.person_id),
                )
        else:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO behavior_policies "
                    "(id,policy_key,policy_text,trigger_context,action_hint,"
                    "source_memory_id,confidence,last_seen_at,created_at,updated_at,person_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (str(uuid.uuid4()), key, text, trigger_context, action_hint,
                     source_memory_id, confidence, now, now, now, self._ctx.person_id),
                )

    def project_observation(
        self, conn: "_RealDictConnWrapper", obs_id: str, content: str, kind: str, emotion: str
    ) -> None:
        try:
            if kind == "self_model":
                self._upsert_semantic_fact_locked(
                    conn, "self_model:core", content,
                    confidence=0.85, source_memory_id=obs_id, tags="self_model",
                )
            elif kind == "curiosity":
                self._upsert_behavior_policy_locked(
                    conn, "curiosity:active", content,
                    trigger_context="idle", action_hint="look_around",
                    confidence=0.75, source_memory_id=obs_id,
                )
            elif kind == "conversation" and emotion == "moved":
                self._upsert_behavior_policy_locked(
                    conn, "conversation:support", content,
                    trigger_context="conversation", action_hint="respond_supportively",
                    confidence=0.80, source_memory_id=obs_id,
                )
        except Exception as e:
            logger.warning("_project_observation failed: %s", e)

    def adjust_behavior_policy_confidence(
        self, key: str, delta: float, reason: str = ""
    ):
        try:
            now = clock.now_local_iso()
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, policy_text, confidence FROM behavior_policies "
                        "WHERE policy_key=%s AND person_id=%s",
                        (key, self._ctx.person_id),
                    )
                    row = cur.fetchone()
                if not row:
                    return None
                new_conf = max(0.0, min(1.0, float(row["confidence"]) + delta))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_revisions "
                        "(id,entity_type,entity_key,previous_text,new_text,"
                        "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                        "VALUES (%s,'behavior_policy',%s,%s,%s,%s,%s,NULL,%s,%s)",
                        (str(uuid.uuid4()), key,
                         row["policy_text"], row["policy_text"],
                         float(row["confidence"]), new_conf, reason, now),
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE behavior_policies SET confidence=%s,updated_at=%s "
                        "WHERE policy_key=%s AND person_id=%s",
                        (new_conf, now, key, self._ctx.person_id),
                    )
                conn.commit()
            return new_conf
        except Exception as e:
            logger.warning("adjust_behavior_policy_confidence failed: %s", e); return None

    async def adjust_behavior_policy_confidence_async(
        self,
        key: str,
        delta: float,
        reason: str = "",
        policy_text: str = "",
        trigger_context: str = "",
        action_hint: str = "",
    ):
        """Async wrapper: adjust confidence, upserting the policy if policy_text is given."""
        def _run():
            if policy_text:
                try:
                    with self._ctx.lock:
                        conn = self._ctx.conn()
                        self._upsert_behavior_policy_locked(
                            conn, key, policy_text,
                            trigger_context=trigger_context,
                            action_hint=action_hint,
                        )
                        conn.commit()
                except Exception as e:
                    logger.warning("adjust_behavior_policy_confidence_async upsert failed: %s", e)
            return self.adjust_behavior_policy_confidence(key, delta, reason)
        return await asyncio.to_thread(_run)

    def adjust_semantic_fact_confidence(
        self, key: str, delta: float, reason: str = ""
    ):
        try:
            now = clock.now_local_iso()
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, fact_text, confidence FROM semantic_facts "
                        "WHERE fact_key=%s AND person_id=%s",
                        (key, self._ctx.person_id),
                    )
                    row = cur.fetchone()
                if not row:
                    return None
                new_conf = max(0.0, min(1.0, float(row["confidence"]) + delta))
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_revisions "
                        "(id,entity_type,entity_key,previous_text,new_text,"
                        "previous_confidence,new_confidence,source_memory_id,reason,created_at) "
                        "VALUES (%s,'semantic_fact',%s,%s,%s,%s,%s,NULL,%s,%s)",
                        (str(uuid.uuid4()), key,
                         row["fact_text"], row["fact_text"],
                         float(row["confidence"]), new_conf, reason, now),
                    )
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE semantic_facts SET confidence=%s,updated_at=%s "
                        "WHERE fact_key=%s AND person_id=%s",
                        (new_conf, now, key, self._ctx.person_id),
                    )
                conn.commit()
            return new_conf
        except Exception as e:
            logger.warning("adjust_semantic_fact_confidence failed: %s", e); return None

    async def adjust_semantic_fact_confidence_async(self, key: str, delta: float, reason: str = ""):
        return await asyncio.to_thread(self.adjust_semantic_fact_confidence, key, delta, reason)

    def recall_revisions(
        self,
        entity_type: str = "semantic_fact",
        entity_key: str | None = None,
        n: int = 50,
    ) -> list[dict]:
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                params: list = [entity_type]
                sql = (
                    "SELECT id,entity_type,entity_key,previous_text,new_text,"
                    "previous_confidence,new_confidence,source_memory_id,reason,created_at "
                    "FROM memory_revisions WHERE entity_type=%s"
                )
                if entity_key:
                    sql += " AND entity_key=%s"
                    params.append(entity_key)
                sql += " ORDER BY created_at DESC LIMIT %s"
                params.append(n)
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning("recall_revisions failed: %s", e); return []

    def recall_behavior_policies(self, query: str, n: int = 5) -> list[dict]:
        like = f"%{query.strip()}%" if query.strip() else "%"
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT policy_key,policy_text,trigger_context,action_hint,"
                        "source_memory_id,confidence,last_seen_at "
                        "FROM behavior_policies "
                        "WHERE person_id=%s AND (%s='%%' OR policy_text LIKE %s "
                        "   OR trigger_context LIKE %s OR action_hint LIKE %s) "
                        "ORDER BY CASE WHEN policy_text LIKE %s THEN 0 ELSE 1 END, last_seen_at DESC "
                        "LIMIT %s",
                        (self._ctx.person_id, like, like, like, like, like, n),
                    )
                    return [
                        {"key":r["policy_key"],"summary":r["policy_text"],
                         "trigger_context":r["trigger_context"],"action_hint":r["action_hint"],
                         "source_memory_id":r["source_memory_id"],
                         "confidence":float(r["confidence"]),"last_seen_at":r["last_seen_at"]}
                        for r in cur.fetchall()
                    ]
        except Exception as e:
            logger.warning("recall_behavior_policies failed: %s", e); return []

    async def recall_behavior_policies_async(self, *a, **kw):
        return await asyncio.to_thread(self.recall_behavior_policies, *a, **kw)

    def link_memories(self, src: str, tgt: str, link_type: str = "related", note: str | None = None) -> bool:
        try:
            with self._ctx.lock:
                conn = self._ctx.conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_links (id,source_id,target_id,link_type,note,created_at) "
                        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (str(uuid.uuid4()), src, tgt, link_type, note, clock.now_local_iso()),
                    )
                conn.commit()
            return True
        except Exception as e:
            logger.warning("link_memories failed: %s", e); return False

    async def link_memories_async(self, *a, **kw):
        return await asyncio.to_thread(self.link_memories, *a, **kw)

    def get_linked_memories(self, memory_id: str, direction: str = "both") -> list[dict]:
        try:
            results = []
            with self._ctx.lock:
                conn = self._ctx.conn()
                if direction in ("out", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.timestamp,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.target_id "
                            "WHERE ml.source_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend(
                            {**dict(r), "date": clock.ts_to_date(r["timestamp"]), "time": clock.ts_to_time(r["timestamp"]), "link_direction": "→"}
                            for r in cur.fetchall()
                        )
                if direction in ("in", "both"):
                    with conn.cursor() as cur:
                        cur.execute(
                            "SELECT o.id,o.content,o.timestamp,o.emotion,o.kind,"
                            "ml.link_type,ml.note FROM memory_links ml "
                            "JOIN observations o ON o.id=ml.source_id "
                            "WHERE ml.target_id=%s AND o.superseded_by IS NULL",
                            (memory_id,),
                        )
                        results.extend(
                            {**dict(r), "date": clock.ts_to_date(r["timestamp"]), "time": clock.ts_to_time(r["timestamp"]), "link_direction": "←"}
                            for r in cur.fetchall()
                        )
            return results
        except Exception as e:
            logger.warning("get_linked_memories failed: %s", e); return []

    async def get_linked_memories_async(self, *a, **kw):
        return await asyncio.to_thread(self.get_linked_memories, *a, **kw)

    def format_semantic_facts_for_context(self, facts: list[dict]) -> str:
        if not facts: return ""
        lines = ["[安定した事実（semantic memory）]:"]
        for f in facts:
            lines.append(f"- conf:{float(f.get('confidence',0)):.2f} key:{str(f.get('key','?'))[:24]}: {str(f.get('summary',''))[:140]}")
        return "\n".join(lines)

    def format_behavior_policies_for_context(self, policies: list[dict]) -> str:
        if not policies: return ""
        lines = ["[行動方針（policy memory）]:"]
        for p in policies:
            lines.append(f"- conf:{float(p.get('confidence',0)):.2f} trigger:{str(p.get('trigger_context',''))[:24]} action:{str(p.get('action_hint',''))[:32]}: {str(p.get('summary',''))[:140]}")
        return "\n".join(lines)
