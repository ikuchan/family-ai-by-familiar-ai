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
    until: float  # epoch 秒
    # 依頼の由来。空＝人が「黙って」と頼んだ。"timer:<id>"＝タイマーが鳴るまで黙る（2026-09-16）。
    reason: str = ""


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
        return SilenceRequest(
            person=str(data["person"]),
            until=float(data["until"]),
            reason=str(data.get("reason") or ""),
        )
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
                    (
                        _STATE_KEY,
                        json.dumps(
                            {"person": req.person, "until": req.until, "reason": req.reason}
                        ),
                        now,
                    ),
                )
            conn.commit()
        logger.info(
            "黙っているよう頼まれた：%s（%.0f 分）",
            req.person,
            max(0.0, (req.until - datetime.now(timezone.utc).timestamp())) / 60,
        )
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


def hush_for_timer(
    current: SilenceRequest | None, *, person: str, until: float, tid: int, now: float | None = None
) -> SilenceRequest:
    """タイマーを掛けたときの次の依頼（純関数・`TIMER_SILENCE`・2026-09-16）。

    鳴る時刻＝期限。人が明示的に頼んだ依頼（`reason` が空）が生きていれば上書きしない。
    タイマー由来の依頼が生きていれば、遅いほうの鳴る時刻まで（先のタイマーの id のまま）。
    """
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    if current is not None and now < current.until:
        if not current.reason.startswith("timer:"):
            return current
        if current.until >= until:
            return current
    return SilenceRequest(person=person, until=until, reason=f"timer:{tid}")


def unhush_timer(current: SilenceRequest | None, *, tid: "int | str") -> SilenceRequest | None:
    """タイマーを止めたときの次の依頼（純関数）。そのタイマー由来なら解く（`all` はタイマー由来すべて）。"""
    if current is None or not current.reason.startswith("timer:"):
        return current
    if tid == "all" or current.reason == f"timer:{tid}":
        return None
    return current


def resolve_minutes(asked: int, *, default: int, maximum: int) -> int:
    """調停が返した分数を、実際に黙る分数へ直す。

    `-1` は「頼まれたが長さの指定なし」で、既定を当てる（軽量LLM は既定値を知らない）。
    上限を超える指定は**弾かずに丸める**。「3時間黙って」に黙らないより、上限まで黙るほうが
    意図に近い。`0` はそのまま「黙らない」。
    """
    if asked == 0:
        return 0
    minutes = default if asked < 0 else asked
    return min(minutes, maximum)
