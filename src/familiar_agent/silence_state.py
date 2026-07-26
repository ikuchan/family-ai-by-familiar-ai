"""「黙っていて」と頼まれた状態。

止めるのは**発話すべて**（自発だけでなく、話しかけられても話さない）。解けるのは
**退室**（頼んだ人が居なくなる）と**時間**（Config・既定60分）の2つ。黙っているあいだの
言葉は捨てず `pending_speech` へ溜め、解けたときに配る。

状態は `agent_state` に置く。再起動で消えると、頼んだ本人からは「勝手に喋り出した」と
しか見えない。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .db import get_db

logger = logging.getLogger(__name__)

_STATE_KEY = "silence_request"


@dataclass(frozen=True)
class SilenceRequest:
    """誰が・いつまで黙っていてほしいと言ったか。"""

    person: str
    until: float          # epoch 秒


def is_silenced(req: SilenceRequest | None, *, present: set[str], now: float) -> bool:
    """いま黙っているべきか。

    頼んだ人が**居て**、かつ**期限内**のときだけ真。居なくなれば（退室）その時点で解け、
    期限を過ぎても解ける。どちらも判定だけで済むので、解除の処理を別に持たない。
    """
    if req is None:
        return False
    if now >= req.until:
        return False
    return req.person in present


def load_silence() -> SilenceRequest | None:
    """保存された依頼を読む。無ければ None。"""
    try:
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT value_json FROM agent_state WHERE state_key = %s", (_STATE_KEY,)
                )
                row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row["value_json"] if isinstance(row, dict) else row[0])
        return SilenceRequest(person=str(data["person"]), until=float(data["until"]))
    except Exception as e:  # noqa: BLE001
        logger.warning("黙っている依頼を読めなかった: %s", e)
        return None


def save_silence(req: SilenceRequest) -> None:
    """依頼を保存する。"""
    try:
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
                    (_STATE_KEY, json.dumps({"person": req.person, "until": req.until}), now),
                )
            conn.commit()
        logger.info("黙っているよう頼まれた：%s（%.0f 分）",
                    req.person, max(0.0, (req.until - datetime.now(timezone.utc).timestamp())) / 60)
    except Exception as e:  # noqa: BLE001
        logger.warning("黙っている依頼を保存できなかった: %s", e)


def clear_silence() -> None:
    """依頼を消す。"""
    try:
        db = get_db()
        with db.lock:
            conn = db.conn()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_state WHERE state_key = %s", (_STATE_KEY,))
            conn.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("黙っている依頼を消せなかった: %s", e)
    return None
