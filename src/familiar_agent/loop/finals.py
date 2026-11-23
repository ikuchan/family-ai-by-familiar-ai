"""最終版を書く（2026-11-23 に `event_loop.py` から移した）。

求めが閉じたときの最終版を 1 件書いて前の版を畳む（`write_final`）、長すぎれば主LLM に要約させる（`shorten_final`）、
前に閉じた最終版の続きなら継起で結ぶ（`link_succession`）。どれも `agent` を受け取るだけで状態を持たない。背景で書く
タスクを立てるのはループ（`event_loop.InformationProcessing._spawn_final`）で、ここではループもタスクも持たない。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from itertools import zip_longest

from ..core import final_version, jev_judges
from ..io.oif import MI
from ..store.relations import KIND_REVISION, KIND_SUCCESSION

# ループの記録と同じ名前で書く（`iteration.py` と同じ。試験は `caplog` をこの名前で絞って見る）。
logger = logging.getLogger("familiar_agent.loop.event_loop")


async def write_final(
    agent,
    content: str,
    *,
    parent: "str | None",
    live: "str | None",
    at: "datetime | None" = None,
    fold: str = "",
) -> "str | None":
    """最終版を 1 件書き、前の版を改訂で畳む。`at` は閉じた時刻（背景で書くときだけ渡す）。

    `fold` は前の窓の最終版（環-ar 段 E）。これも改訂で畳む。
    """
    extra = {"at": at} if at is not None else {}
    final_id = await agent._oif.write(
        MI(
            id="",
            content=content,
            timestamp=None,
            direction=final_version.DIRECTION,
            parent_id=parent,
        ),
        **agent._observation_perspective(),
        **extra,
    )
    folded = [
        old
        for old in ((live, fold) if final_id else ())
        if old and old != final_id and agent._oif.supersede(old, final_id, kind=KIND_REVISION)
    ]
    logger.info(
        "event-loop 最終版を書いた（%d 字%s）",
        len(content),
        "・前の窓の最終版を畳んだ" if fold and fold in folded else "",
    )
    return final_id


async def shorten_final(
    agent, head: str, body: str, course: str, limit: int, prompt: str, window: bool
) -> str:
    """長すぎる最終版を主LLM に要約させる。だめなら切る（窓の最終版は古い行から落とす）。"""
    room = max(1, limit - len(head))
    logger.info(
        "event-loop 最終版を要約する（%s・%d 字 → %d 字まで）",
        "窓の会話" if window else "1 回のやりとり",
        len(body + course),
        room,
    )
    summary = ""
    try:
        summary = str(
            await agent.backend.complete(prompt.format(n=room, text=body + course), room) or ""
        ).strip()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("event-loop 最終版の要約に失敗した（切って書く）", exc_info=True)
    if summary and len(summary) <= room:
        return head + summary
    if window:
        return final_version.cut_window(head, body, course, limit)
    return final_version.cut(head + body + course, limit)


async def link_succession(
    agent, final_id: str, content: str, start: datetime, *, prev: str = ""
) -> "str | None":
    """いま書いた最終版を、その前に閉じた最終版の続きなら継起（前 → 後）で結ぶ（記-t 段 2・背景）。

    結ぶのは最終版どうし（本人の決定）。起点どうしでは、想起しても人の言葉しか出てこず意味のない網になる。
    候補は開始時刻の前 `succession_window_sec`（5 分〔仮〕）のうちに閉じた最終版で、Jev が「どれの続きか」を
    「どれの続きでもない」と並べて選ぶ。候補が無ければ聞かない。失敗・確信度が低いときは書かない。
    結んだ「前」の id を返す。`prev` を渡されたら聞かずにそれと結ぶ（書き直した窓の最終版に、窓の 1 つ目が
    結んでいた「前」を付け直す・環-ar 段 E）。
    """
    try:
        if not prev:
            window = timedelta(seconds=int(agent.config.memory.succession_window_sec))
            rows = agent._oif.recent_finals(start - window, start, exclude_id=final_id)
            if not rows:
                logger.info("event-loop 最終版の継起を書かない：候補なし")
                return None
            limit = int(getattr(agent.config, "jev_state_max_chars", 28_000))
            current, cands = final_version.fit(
                content, {str(r["id"]): str(r["content"]) for r in rows}, limit
            )
            prev = (
                await jev_judges.judge_succession(
                    getattr(agent, "_jev", None),
                    current=current,
                    candidates=cands,
                    min_conf=float(agent.config.jev_confidence_min),
                )
                or ""
            )
            if not prev:
                logger.info(
                    "event-loop 最終版の継起を書かない：どれの続きでもない（候補 %d 件）",
                    len(cands),
                )
                return None
        agent._oif.link(KIND_SUCCESSION, [(prev, "前", 0), (final_id, "後", 1)])
        logger.info("event-loop 最終版の継起を書いた：前 %.8s → 後 %.8s", prev, final_id)
        return prev
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("event-loop 最終版の継起を書けなかった", exc_info=True)
        return None


#: 記憶で答えたときに結果の頭に付ける印（その回の最終版を、次の「最近の答え」の候補から外すのに使う）。
REUSED_MARK = "（記憶の答え・調べ直していない）"


async def recent_answer(
    agent,
    *,
    question: str,
    query: str,
    recalled: "list | None" = None,
    now: "datetime | None" = None,
) -> "str | None":
    """いまの問いに、記憶の最終版がもう答えていて、その答えがいまも使えれば、その本文（2026-11-23・本人の決め方）。

    候補は、直近 `answer_recent_hours` の最終版（新しい順・読むだけ）と、この求めの W に載った最終版（`recalled`・W に載った順・
    引き直さないので「思い出した」を書かない）を 1 件ずつ交互に並べたもの。続けて聞いたときの前の答え（直近）と、古い結果
    （W）のどちらも、もう片方に押し出されない。本文で印と畳みを見て落とし（記憶で答えた回・改訂で畳まれた記録）、残りの頭から
    `answer_candidates_max` 件。Jev に 2 回聞く：どれが答えているか（`judge_answered`）、選ばれた記録の種類といまから出来事
    まで（`judge_freshness`）。使うのは、過去の結果か、予報・予定で調べてからの経過がいまから出来事までの下限より短いとき
    （＝調べた時から出来事までの半分を過ぎていない）。進行中・過ぎた・読めない・確信が低い・失敗なら None（いまどおり調べる）。
    答えに使う本文は切らない（Jev へ送る文だけを上限に合わせる）。
    """
    if not (question or "").strip():
        return None
    try:
        now = now or datetime.now(timezone.utc)
        mem = agent.config.memory
        since = now - timedelta(hours=float(mem.answer_recent_hours))
        rows = await asyncio.to_thread(agent._oif.recent_finals, since, now, exclude_id="")
        recent = {str(r["id"]): r["timestamp"] for r in rows}
        on_w: "dict[str, datetime]" = {}
        for r in recalled or []:
            mi = getattr(r, "mi", None)
            if (
                mi is not None
                and mi.direction == final_version.DIRECTION
                and mi.timestamp is not None
            ):
                on_w.setdefault(str(mi.obs_id or mi.id), mi.timestamp)
        stamps = {**on_w, **recent}
        order = dict.fromkeys(k for pair in zip_longest(recent, on_w) for k in pair if k)
        bodies = await asyncio.to_thread(agent._oif.bodies, list(order)) if order else {}
        usable = [k for k in order if k in bodies and REUSED_MARK not in bodies[k]]
        cands = {k: bodies[k] for k in usable[: int(mem.answer_candidates_max)]}
        if not cands:
            logger.info("event-loop 記憶の答えの候補なし（調べる）")
            return None
        limit = int(getattr(agent.config, "jev_state_max_chars", 28_000))
        asked = f"{question}（調べる語：{query}）" if query and query not in question else question
        client, min_conf = getattr(agent, "_jev", None), float(agent.config.jev_confidence_min)
        pick = await jev_judges.judge_answered(
            client, question=asked, candidates=cands, now=now, min_conf=min_conf, limit=limit
        )
        if pick is None:
            logger.info("event-loop 記憶に答えは無い（候補 %d 件・調べる）", len(cands))
            return None
        fresh = await jev_judges.judge_freshness(
            client, record=cands[pick], now=now, min_conf=min_conf, limit=limit
        )
        hours_ago = (now - stamps[pick]).total_seconds() / 3600
        if fresh is None or not _still_usable(
            *fresh, hours_ago=hours_ago, lower=mem.answer_remain_hours
        ):
            logger.info(
                "event-loop 記憶の答えは使わない（%.8s・%s・%.1f 時間前・調べ直す）",
                pick,
                fresh,
                hours_ago,
            )
            return None
        logger.info(
            "event-loop 記憶の答えで答える（%.8s・%s・%.1f 時間前・調べない）",
            pick,
            fresh,
            hours_ago,
        )
        return cands[pick]
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001
        logger.warning("event-loop 記憶の答えを探せなかった（調べる）", exc_info=True)
        return None


def _still_usable(kind: str, remain: str, *, hours_ago: float, lower: "dict[str, float]") -> bool:
    """過去の結果なら使える。予報・予定は、調べてからの経過が、いまから出来事までの下限より短いとき（本人の決め方）。

    調べた時から出来事までの長さ＝経過＋残り なので、「経過がその半分より短い」は「経過 < 残り」と同じ。残りは下限で見積もる。
    進行中は調べ直す（仮）。
    """
    if kind == jev_judges.KIND_RESULT:
        return True
    if kind != jev_judges.KIND_FORECAST:
        return False
    return hours_ago < float(lower.get(remain, 0))
