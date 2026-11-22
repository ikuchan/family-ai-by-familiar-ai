"""話題にされた人の面（about）を足す——REST 内省（記-ae 段 2・2026-11-22）。

「ママが『パパ、最近忙しそう』と話したことを、パジュはパパと話すときに思い出す」（README）。想起は人の面で引くので、
話題にされた人に about の面が無いと、その人と話すときに上がらない。面の生成の段②（REST が本文を読んで意味役割を足す・
`MIデータモデル`）のうち、about だけを作り直したもの（段②のコードは 2026-08 に失われた・記-a-ほ）。

**面は最終版に立て、判定は起点の言葉でする。** 人の言葉（起点）はその晩の畳み込みで隠れて想起に上がらないので、会話の
記録で見え続ける最終版（起点を親に持つ）に立てる（`store/observations.about_pending`）。`about_lookback_days`（2 日）より
新しい最終版を新しい順に見る（調べた印は持たない）。候補はコードで出す：起点の言葉に呼び名が出る登録の人から、話した本人と
パジュを除く。
決めるのは Jev（候補ごとに 1 問・`core/jev_judges.judge_about`）。今ある記録にさかのぼって足すのは埋める道具（段 3）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta

from ..config import MemoryConfig
from ..core.jev_judges import judge_about
from ..core.speaker_claim import aliases_of
from ..person_memory_manager import AGENT_SELF_ID
from ..store import clock

logger = logging.getLogger(__name__)

#: 話した人が分からない発話（規則 048 で actor が自分の印に寄ったもの）の、Jev に渡す呼び方。
_UNKNOWN_SPEAKER = "（分からない人）"


@dataclass(frozen=True)
class AboutResult:
    """調べた発話の数、聞いた問いの数、足した面の数、Jev が答えなかった問いの数、「はい」でも書けなかった数（埋め込みが無い）。

    `skipped` は何もしなかった理由。
    """

    sources: int
    asked: int
    added: int
    unanswered: int
    unwritten: int = 0
    skipped: str = ""


def candidates(content: str, actor: "str | None", persons: "list[dict]", done=()) -> "list[dict]":
    """本文に呼び名が出る登録の人（話した本人・パジュ・面がもう立っている人を除く）。

    呼び名の一部が別の語の中にある（パパイヤ）ことは Jev が見分ける。
    """
    out = []
    for p in persons:
        pid = str(p.get("id", ""))
        if not pid or pid in (AGENT_SELF_ID, actor) or pid in done:
            continue
        if any(a and a in content for a in aliases_of(p)):
            out.append(p)
    return out


async def add_about(
    oif, jev, *, limit: int, min_p: float, since, until=None, rows: "int | None" = None
) -> AboutResult:
    """`since` より後（`until` より前）の最終版を新しい順に見て、`limit` 問まで聞き、about の面を足す。

    読む発話は `rows` 件まで（無ければ `about_rows_max`）。問いの上限 `limit` とは別にする（名前の出ない発話を数えない）。
    """
    if jev is None or not getattr(jev, "available", False):
        return AboutResult(0, 0, 0, 0, skipped="Jev が使えない")
    persons = [
        p for p in await asyncio.to_thread(oif.persons) if str(p.get("id", "")) != AGENT_SELF_ID
    ]
    names = await asyncio.to_thread(oif.person_names, [str(p["id"]) for p in persons])
    cap = rows if rows is not None else MemoryConfig().about_rows_max
    pending = await asyncio.to_thread(oif.about_pending, cap, since=since, until=until)
    examined = asked = added = unanswered = unwritten = 0
    for row in pending:
        if asked >= limit:
            break
        obs_id = str(row["obs_id"])
        actor = row.get("actor")
        examined += 1
        content = str(row.get("content") or "")
        speaker = names.get(str(actor)) or _UNKNOWN_SPEAKER
        for p in candidates(content, actor, persons, done=set(row.get("done") or ())):
            if asked >= limit:
                break
            person = names.get(str(p["id"]))
            if not person:
                continue  # 呼び名の無い人は、Jev に「この人」を渡せない
            asked += 1
            got = await judge_about(jev, text=content, speaker=speaker, person=person, min_p=min_p)
            if got is None:
                unanswered += 1
            elif got:
                if await asyncio.to_thread(oif.add_about, obs_id, str(p["id"])):
                    added += 1
                else:
                    unwritten += 1  # 埋め込みの無い記録
    logger.info(
        "rest about の面：発話 %d 件・問い %d・足した %d・答えなし %d・書けなかった %d",
        examined,
        asked,
        added,
        unanswered,
        unwritten,
    )
    return AboutResult(examined, asked, added, unanswered, unwritten)


async def add_about_tonight(agent) -> AboutResult:
    """REST の毎晩の口。`about_lookback_days` 日より新しい発話を見る（調べた印は持たない）。"""
    cfg = MemoryConfig()
    since = clock.now_utc() - timedelta(days=cfg.about_lookback_days)
    return await add_about(
        agent._oif,
        getattr(agent, "_jev", None),
        limit=cfg.about_per_night,
        min_p=cfg.about_min_p,
        since=since,
    )
