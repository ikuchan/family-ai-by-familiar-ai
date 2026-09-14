"""REST 内省・層 1 の前段「$I$ を測る → $\\Delta$ 減らす」（記-a-ろ-ろ・2026-09-15・`出来事を畳む` §4）。

毎晩、①日次の畳み込みの前に：

1. 核（根づき $n \\ge 1$ の面・全ての視点）と今日の分（前回の内省以降の記録）の使われる情報量
   $I=\\sum b_i u_i$ を測り、計測ログに `層1計測` を 1 行。
2. $I>I^\\*$ なら $\\Delta=\\lceil\\log_2(I/I^\\*)\\rceil$ だけ、**前回の内省以降に参照されなかった核**の $n$ を
   減らす（1 未満にしない・`ObservationStore.decay_groundedness`）。計測ログに `層1減り` を 1 行。

減らすのは記録でなく使われやすさ $u$（柔らかい削ぎ落とし）。記録を畳んで隠すのは②核の固め
（ろ-に）。$I^\\*$・$HL$ は層 3 の設定値。式は `core/info_measure.py`（純関数）。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..core import info_measure as im
from ..core import measure

logger = logging.getLogger(__name__)


async def measure_and_decay(agent, *, now: "datetime | None" = None) -> str:
    """$I$ を測り、超えていれば $n$ を減らす。何をしたかの短い文を返す（`内省` の記録に載る）。"""
    now = now or datetime.now(timezone.utc)
    mem = agent.config.memory
    hl = float(mem.recall_half_life_days)
    w_t, w_g = float(mem.recall_w_t), float(mem.recall_w_g)
    target = float(mem.info_target_bits)

    core = agent._oif.core_faces()
    fresh = agent._oif.fresh_since_last_rest()
    total = im.total(core, now=now, hl=hl, w_t=w_t, w_g=w_g)
    today = im.total(fresh, now=now, hl=hl, w_t=w_t, w_g=w_g)
    faces = len({r.get("person_id") for r in core})
    measure.record(
        "層1計測",
        I=int(total),
        目標=int(target),
        核=len(core),
        今日=len(fresh),
        今日I=int(today),
        面=faces,
    )
    logger.info(
        "rest 層 1 計測：I=%d bit（目標 %d・核 %d 件・面 %d）今日 %d 件 %d bit",
        total,
        target,
        len(core),
        faces,
        len(fresh),
        today,
    )
    delta = im.delta_for(total, target)
    if delta <= 0:
        return f"使われる情報量 {int(total)} bit（目標 {int(target)}）・根づきは減らさなかった"
    moved = int(agent._oif.decay_groundedness(delta) or 0)
    measure.record("層1減り", Δ=delta, 動かした=moved)
    logger.info("rest 層 1 減り：Δ=%d で %d 面の根づきを下げた", delta, moved)
    return f"使われる情報量 {int(total)} bit（目標 {int(target)}）・Δ={delta} で根づきを {moved} 件下げた"
