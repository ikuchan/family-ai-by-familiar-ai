"""パジュ宛てのメモを読みに行く（知-g-ろ・2026-09-15・`設計方針_家の記録との接続` §9）。

本人が Vault の「パジュへ」のページに書いたことを、T（自律機構）が `INTERVAL_SEC` に 1 回
`get_notes_for_paju()`（家族ティア・`obsidian-memo`）で読み、前回の本文（`agent_state.paju_notes`）と
違えば `機器` の求め「パジュへのメモが変わった」を積む。**何をするかは主LLM が決める**（覚える・
予定に入れる・伝える）。在席ゲート・静穏時間は既存のまま。道具が無ければ黙って何もしない。

比べるのは `## メモ` の本文だけ（先頭の `【いま】`（時刻）と `【出典】` は毎回変わるので外す）。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import psycopg2.extras

from ..db import get_db

logger = logging.getLogger(__name__)

TOOL = "get_notes_for_paju"
INTERVAL_SEC = 3600.0  # 1 時間に 1 回〔仮〕
_STATE_KEY = "paju_notes"
_NONE = "（まだ無い）"


def body_of(text: str) -> str:
    """道具の返りから、比べる本文（`## メモ` の下）だけを取り出す。"""
    lines = (text or "").splitlines()
    if "## メモ" in lines:
        lines = lines[lines.index("## メモ") + 1 :]
    else:
        lines = [ln for ln in lines if not ln.startswith(("【いま】", "【出典】"))]
    body = "\n".join(ln for ln in lines).strip()
    return "" if body == _NONE else body


def _load_state() -> "str | None":
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
    except Exception as e:  # noqa: BLE001
        logger.warning("パジュへのメモの前回値を読めない: %s", e)
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


def _added_lines(before: str, after: str) -> list[str]:
    old = set(before.splitlines())
    return [ln for ln in after.splitlines() if ln.strip() and ln not in old]


async def check_notes(ip) -> bool:
    """1 回読む。変わっていて求めを積んだら True。道具が無い・失敗・変化なしは False。

    `ip` はループ（`InformationProcessing`）。DIF はループが持つ（`agent` には無い——実機で
    `'EmbodiedAgent' object has no attribute '_dif'` と落ちた・2026-09-15）。
    """
    dif = ip._dif
    if not dif.tool_defs(TOOL):
        return False  # 繋がっていない（相手側がまだ足していない）
    text, ok = await dif.call_tool(TOOL, {})
    if not ok:
        logger.warning("パジュへのメモを読めなかった：%.120s", text)
        return False
    body = body_of(text)
    before = _load_state()
    if before is None:
        _save_state(body)  # 初回は覚えるだけ（起動のたびに「変わった」と言わない）
        logger.info("パジュへのメモを初めて読んだ（%d 字）", len(body))
        return False
    if body == before:
        return False
    added = _added_lines(before, body)
    content = "パジュへのメモが変わった。" + (
        "新しく書かれたこと：\n" + "\n".join(added) if added else "書かれていたことが消えた"
    )
    if body:
        content += "\n\nいまのメモ全文：\n" + body
    # **メモは記憶に入れるだけ。話しかけない**（出-as §2.7・2026-09-26）。以前は求めを立てていた。
    # **先に記録してから覚える。** 逆だと、記録で落ちたときに差分が失われる（実機 19:40・
    # `device()` が落ちて前回値だけ進み、次の読みで「変わっていない」になった）。
    await ip.record_device("メモ", content[:1500])
    _save_state(body)
    logger.info(
        "パジュへのメモが変わった（足された行 %d・全 %d 字）→ 記憶に記録した", len(added), len(body)
    )
    return True
