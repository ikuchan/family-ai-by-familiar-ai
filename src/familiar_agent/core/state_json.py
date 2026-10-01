"""`agent_state` の 1 つの鍵に JSON を読み書きする口（2026-10-01）。

機械が持つ状態は DB に置く（開発ルール「保存は PostgreSQL のみ・ファイルは既定値と人の入力だけ」）。家族のいまの
様子（知-ad）・音楽の目録とおすすめ（知-aa）が、同じ読み書きを書き写さないように 1 か所にした。読めない・
書けないときは例外を外へ出さず、読み出しは `None`、書き込みは `False` を返す（呼び手は前の状態のまま続ける）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def read(key: str) -> object:
    """その鍵の JSON。無い・読めなければ None。"""
    try:
        from ..db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("SELECT value_json FROM agent_state WHERE state_key = %s", (key,))
                row = cur.fetchone()
        if not row:
            return None
        raw = row["value_json"] if isinstance(row, dict) else row[0]
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_state の %s を読めなかった: %s", key, e)
        return None


def write(key: str, value: object) -> bool:
    """その鍵に JSON を置く（上書き）。"""
    try:
        from ..db import get_db

        now = datetime.now(timezone.utc).isoformat()
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, "
                    "updated_at = EXCLUDED.updated_at",
                    (key, json.dumps(value, ensure_ascii=False), now),
                )
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_state の %s を保存できなかった: %s", key, e)
        return False


def clear(key: str) -> bool:
    """その鍵を消す。"""
    try:
        from ..db import get_db

        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_state WHERE state_key = %s", (key,))
            conn.commit()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("agent_state の %s を消せなかった: %s", key, e)
        return False
