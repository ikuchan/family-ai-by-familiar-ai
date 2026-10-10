"""家族の予定を先読みし、変わったら O に記録する（情-q・2026-10-10・本人の決定・`設計方針_家の予定との接続` v0.2）。

BOND で話しかけるとき、W が家族の予定に寄るように（`core/bond_cue`）、予定を O に置く。T（自律機構）が起動直後に
1 回、そのあと `INTERVAL_SEC` に 1 回 `get_family_schedule(days=DAYS)`（家族ティア・`family-calendar`）を読み、
時刻の行（`【いま】`・`【出典】`）を外した本文を前回（`agent_state.family_schedule`）と比べる。変わっていれば機器の
記録「予定」として O に書く。**初回も書く**——メモ（`notes_watch`）は初回は覚えるだけだが、予定は O に無いと手がかりで
引けない。日付が進むと 7 日の範囲が動くので、少なくとも 1 日 1 回は書き直す。話しかけはしない。道具が無い・読めない
ときは何もしない。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import psycopg2.extras

from ..db import get_db

logger = logging.getLogger(__name__)

TOOL = "get_family_schedule"
DAYS = 7  # 本人の決定（2026-10-10）
INTERVAL_SEC = 3600.0  # 1 時間に 1 回（本人の決定・メモと同じ）
_STATE_KEY = "family_schedule"
_RECORD_MAX = 1500


def body_of(text: str) -> str:
    """道具の返りから、比べる本文（時刻の行を外したもの）だけを取り出す。"""
    lines = [ln for ln in (text or "").splitlines() if not ln.startswith(("【いま】", "【出典】"))]
    return "\n".join(lines).strip()


def stored() -> "str | None":
    """前回読んだ本文。まだ読んでいない・読めなければ None。"""
    try:
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT value_json FROM agent_state WHERE state_key = %s", (_STATE_KEY,)
                )
                row = cur.fetchone()
        if row is None:
            return None
        data = json.loads(row["value_json"])
        return str(data.get("body", "")) if isinstance(data, dict) else str(data)
    except Exception:  # noqa: BLE001
        logger.warning("家族の予定の前回値を読めない", exc_info=True)
        return None


def _save_state(body: "str | None") -> None:
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            if body is None:
                cur.execute("DELETE FROM agent_state WHERE state_key = %s", (_STATE_KEY,))
            else:
                now = datetime.now(timezone.utc).isoformat()
                cur.execute(
                    "INSERT INTO agent_state (state_key, value_json, updated_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT (state_key) DO UPDATE SET value_json = EXCLUDED.value_json, "
                    "updated_at = EXCLUDED.updated_at",
                    (
                        _STATE_KEY,
                        json.dumps({"body": body, "read_at": now}, ensure_ascii=False),
                        now,
                    ),
                )
        conn.commit()


async def check_schedule(ip) -> bool:
    """1 回読む。O に書いたら True。道具が無い・失敗・変化なしは False。`ip` はループ（DIF はループが持つ）。"""
    dif = ip._dif
    if not dif.tool_defs(TOOL):
        return False
    text, ok = await dif.call_tool(TOOL, {"days": DAYS})
    if not ok:
        logger.warning("家族の予定を読めなかった：%.120s", text)
        return False
    body = body_of(text)
    if body == stored():
        return False
    content = f"家族のこれから {DAYS} 日の予定：\n{body or '予定は入っていない。'}"
    # **先に記録してから覚える**（`notes_watch` と同じ。逆だと記録で落ちたときに変化が失われる）。
    await ip.record_device("予定", content[:_RECORD_MAX])
    _save_state(body)
    logger.info("家族の予定を先読みした（%d 字）→ 記憶に記録した", len(body))
    return True
