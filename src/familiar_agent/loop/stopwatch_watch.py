"""ストップウォッチの寿命（知-u・2026-09-18・`設計方針_ストップウォッチ` v0.1）。T が毎 tick 呼ぶ。

`STOPWATCH_MAX_SEC`（6 時間）を超えて動いているものを止め、O に `予定`「…のストップウォッチを 6 時間で止めた」を
残す。声は出さない（枠に「寿命で自動で止めた」が残るだけ）。09-16 に始めた 2 本が 2 日間動き続けた。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def expire(store, oif, *, now: "datetime | None" = None, max_sec: float) -> int:
    """止めた本数を返す。"""
    from ..io.oif import MI
    from ..person_memory_manager import AGENT_SELF_ID

    now = now or datetime.now(timezone.utc)
    rows = store.expire(now=now, max_sec=max_sec)
    for r in rows:
        hours = max(1, int(max_sec // 3600))
        logger.info(
            "ストップウォッチを寿命で止めた id=%s %s（%d 時間）", r["id"], r["label"], hours
        )
        try:
            await oif.write(
                MI(
                    id="",
                    content=f"「{r['label']}」のストップウォッチを {hours} 時間で止めた（止められないまま動いていた）",
                    timestamp=None,
                    direction="予定",
                ),
                writer_id=AGENT_SELF_ID,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("予定の記録を書けなかった: %s", e)
    return len(rows)
