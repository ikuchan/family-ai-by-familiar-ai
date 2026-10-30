"""経験繋がり——REST 内省・層 1 の④（出-bq・2026-10-29・本人の決定）。

会話の流れで生まれた**意味の似ていない飛び**を、将来、話題を膨らませるための記憶の糸として関係に持つ
（`課題8` 出-bq・`用語_略語一覧`「継起の拡張」）。いまは書くだけで、使う想起は無い。

回るたびに直近 7 日（今日の途中までを 1 日目と数える）を調べ直す。日ごとの塊・3 件の選び出し・
書くかどうかは純関数（`core/experience_links.py`）が決め、ここは材料を読み、決まったことを書く。
同じ塊を何度調べても、同じ 3 件なら何も書かない（1 つの塊に 1 本・よりよい 3 件なら同じ関係の項を入れ替える）。
関係の時刻は書いた時刻でなく持ち主の塊の日（正午）にし、7 日の窓から出た日の関係には触らない。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core import experience_links as xl
from ..store import clock
from ..store.relations import KIND_EXPERIENCE, ROLE_TERM

logger = logging.getLogger(__name__)

DAYS = 7  # 調べ直す日数（今日を含む・本人の決定）
GATE = 0.6  # いちばん近い対の類似度がこれ以下なら書く（本人の決定・記-s の後の 16 塊で 6 本）
MIN_CLUSTER = 5  # これ未満の塊は何もしない
EXACT_MAX = 100  # すべての組を調べる塊の大きさの上限（超えたら近似・本人の決定）


@dataclass(frozen=True)
class Outcome:
    """書いた・入れ替えた・外した関係の本数。"""

    written: int
    replaced: int
    removed: int


async def link_experiences(agent, *, now: "datetime | None" = None) -> Outcome:
    """直近 7 日の日ごとの塊から経験繋がりを書き、何本動かしたかを返す。

    DB の読み書きと 3 件の総当たりは同期なので、イベントループを塞がないよう別スレッドで回す。
    """
    return await asyncio.to_thread(_link, agent, now)


def _link(agent, now: "datetime | None") -> Outcome:
    tz = clock.local_tz()
    now = now or clock.now_utc()
    since = (now.astimezone(tz) - timedelta(days=DAYS - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    m = agent._oif.experience_material(since)
    edges = [xl.Edge(e["kind"], e["created_at"], tuple(e["members"])) for e in m["edges"]]
    clusters = xl.day_clusters(edges, m["parents"], tz)
    links = {
        int(rid): xl.Link(lk["at"].astimezone(tz).date().isoformat(), frozenset(lk["ids"]))
        for rid, lk in m["links"].items()
    }
    wanted = {o for cls in clusters.values() for c in cls if len(c) >= MIN_CLUSTER for o in c}
    # 比べる既存の関係は塊にまるごと入るものだけなので、その項の埋め込みは塊のぶんで足りる。
    emb = agent._oif.embeddings(sorted(wanted))
    vecs = {i: v for i, (_ts, v) in emb.items()}
    actions = xl.plan(clusters, links, vecs, gate=GATE, min_size=MIN_CLUSTER, exact_max=EXACT_MAX)
    done = {"add": 0, "replace": 0, "remove": 0}
    for a in actions:
        if a.op == "remove":
            ok = bool(agent._oif.unlink(KIND_EXPERIENCE, a.relation_id))
        else:
            ordered = sorted(a.ids, key=lambda i: (emb[i][0], i))  # 位置は記録の時刻の古い順
            members = [(i, ROLE_TERM, p) for p, i in enumerate(ordered)]
            if a.op == "add":
                # 関係の時刻は持ち主の塊の日（正午）。窓から出た日の関係に、後の日の塊が触らない
                noon = datetime.fromisoformat(a.day).replace(hour=12, tzinfo=tz)
                ok = agent._oif.link(KIND_EXPERIENCE, members, at=noon) is not None
            else:
                ok = bool(agent._oif.relink(KIND_EXPERIENCE, a.relation_id, members))
        done[a.op] += ok  # 決めた数でなく、実際に動いた数を数える
    logger.info(
        "rest 経験繋がり：%s から %d 日・塊 %d（%d 件以上 %d）・書いた %d・入れ替えた %d・外した %d",
        since.date(),
        DAYS,
        sum(len(c) for c in clusters.values()),
        MIN_CLUSTER,
        sum(len(x) >= MIN_CLUSTER for c in clusters.values() for x in c),
        done["add"],
        done["replace"],
        done["remove"],
    )
    return Outcome(done["add"], done["replace"], done["remove"])
