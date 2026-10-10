"""REST の音楽の仕事（知-aa 段 3・段 4・2026-10-01〜02・本人の決定）。

- **目録の読み直し**：晩に、自分の Spotify のプレイリストの中身と、保存したアルバムと曲を読み直して DB に置く
  （`core/music_catalog`）。読めなかったプレイリストは前の晩の中身を残す（Spotify は落ちる前提のもの）。
- **おすすめの候補**：晩に 1 件だけ用意する（`core/music_suggestion`）。フルLLM がプレイリストのアーティストを材料に、
  プレイリストに入っていない曲を 1 つ理由つきで選び、機械が検索で実在を確かめる。候補が残っていれば作らない。

音楽の器が無い機体では何もしない。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..core.structured_ask import read_json_merged
from ..core import music_catalog as mc
from ..core import music_rules
from ..core import music_suggestion as ms

logger = logging.getLogger(__name__)


@dataclass
class CatalogResult:
    playlists: int = 0
    unread: int = 0  # 中身を読めず、前の晩の中身を残したプレイリスト（自分のもの）
    unreadable: int = 0  # 中身を読めない他人のプレイリスト（知-ao）
    tracks: int = 0


async def refresh_catalog(agent) -> CatalogResult:
    """目録を読み直す。Spotify を呼ぶのはスレッドで（ネットワークの待ちでループを止めない）。"""
    tool = getattr(agent, "_music_tool", None)
    web = getattr(tool, "_web", None) if tool is not None else None
    if web is None:
        return CatalogResult()
    return await asyncio.to_thread(_refresh, web)


#: 中身を読めない他人のプレイリストを読み直すまでの日数〔仮・知-ao〕。
UNREADABLE_RETRY_DAYS = 7


def _days_between(since: str, today: str) -> int:
    from datetime import date

    try:
        return (date.fromisoformat(today) - date.fromisoformat(since)).days
    except ValueError:
        return UNREADABLE_RETRY_DAYS  # 読めない日付なら読み直す


def _refresh(web, *, today: "str | None" = None) -> CatalogResult:
    """目録を読み直す。他人のもので中身を読めないものは、覚えて `UNREADABLE_RETRY_DAYS` 日は読まない（知-ao）。

    Spotify のプレイリストの中身を読む口は、自分が持っているか共同編集者のものだけで、ほかは 403 を返す（公式の
    リファレンス）。毎晩 403 を叩いていた。どちらも読めなければ前の中身は残す。自分のものは一時の失敗として毎晩読み直す。
    """
    if today is None:
        from ..store.clock import local_tz
        from datetime import datetime

        today = datetime.now(local_tz()).date().isoformat()
    before = {p.id: p for p in mc.stored().playlists}
    result = CatalogResult()
    playlists: list[mc.Playlist] = []
    for p in web.my_playlists():
        old = before.get(p["id"])
        since = old.unreadable_since if old is not None and not p["mine"] else ""
        if since and _days_between(since, today) < UNREADABLE_RETRY_DAYS:
            result.unreadable += 1
            kept = list(old.tracks) if old is not None else []
            playlists.append(mc.Playlist(p["id"], p["name"], p["uri"], kept, p["mine"], since))
            result.tracks += len(kept)
            continue
        items = web.playlist_items(p["id"])
        mark = ""
        if items is None:
            # 読めなくても前の中身は残す（データを失わない）。他人のものは印を付けて数え分ける。
            items = old.tracks if old is not None else []
            if p["mine"]:
                result.unread += 1
            else:
                result.unreadable += 1
                mark = today
        playlists.append(mc.Playlist(p["id"], p["name"], p["uri"], list(items), p["mine"], mark))
        result.tracks += len(items)
    result.playlists = len(playlists)
    if playlists or not before:
        mc.store(
            mc.Catalog(playlists=playlists, albums=web.saved_albums(), tracks=web.saved_tracks())
        )
    logger.info(
        "rest 音楽の目録：プレイリスト %d・曲 %d（読めず前のまま %d・中身を読めない他人のもの %d）",
        result.playlists,
        result.tracks,
        result.unread,
        result.unreadable,
    )
    return result


_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、家族に 1 曲すすめたい。家族のプレイリストに入っている
アーティストを、多く入っている順に並べる（括弧の数は入っている曲の数）。

この好みに合いそうで、**プレイリストに入っていない曲**を、合いそうな順に {count} つ選び、それぞれ選んだ理由を
添える。プレイリストのアーティストの別の曲でも、似たテイストの別のアーティストの曲でもよい。実在する曲だけを
選ぶ（曲名とアーティスト名は、Spotify で検索して見つかる正確な表記で）。前に気に入ってもらえた曲は好みの
手がかりに、断られた曲は避ける手がかりにする。無かった曲は、前に挙げたが Spotify に無かった曲なので挙げない。

出力は次の JSON だけ（ほかには何も書かない）：
{{"candidates": [{{"title": "曲名", "artist": "アーティスト名", "reason": "選んだ理由（1〜2 文）"}}]}}

[プレイリストのアーティスト]
{artists}

[前に気に入ってもらえた曲]
{liked}

[断られた曲]
{declined}

[無かった曲]
{missing}
"""

#: 1 回の頼みで挙げてもらう候補の数（知-aq・本人の決定ア）。
CANDIDATES = 5


def _parse(raw: str) -> "dict | None":
    data = read_json_merged(raw)  # 2 つに分けた・書き直した返りも読む（記-p）
    if data is None:
        return None
    out = {k: str(data.get(k, "")).strip() for k in ("title", "artist", "reason")}
    return out if all(out.values()) else None


def _candidates(raw: str) -> "list[dict]":
    """`candidates` の並びを読む。前の形（1 つの物体）が返っても 1 件として読む。欄が欠けたものは外す。"""
    data = read_json_merged(raw)
    if data is None:
        return []
    items = data.get("candidates")
    if not isinstance(items, list):
        one = _parse(raw)
        return [one] if one else []
    out = []
    for item in items:
        if isinstance(item, dict):
            c = {k: str(item.get(k, "")).strip() for k in ("title", "artist", "reason")}
            if all(c.values()):
                out.append(c)
    return out


def _songs(rows: "list[dict]") -> str:
    return "\n".join(f"- {r.get('title', '')}（{r.get('artist', '')}）" for r in rows) or "（なし）"


async def prepare_suggestion(agent) -> bool:
    """おすすめの候補を 1 件用意する。用意できたら True。"""
    state = ms.stored()
    if state.candidate is not None and state.candidate.offered >= ms.MAX_OFFERS:
        ms.drop_if_unanswered(state)  # 返事が無いまま 2 回すすめた（もう一度だけ・本人の決定）
        ms.store(state)
    if state.candidate is not None:
        return False  # 1 件を使い切ってから次
    tool = getattr(agent, "_music_tool", None)
    web = getattr(tool, "_web", None) if tool is not None else None
    catalog = mc.stored()
    artists = catalog.playlist_artists()
    if web is None or not artists:
        return False
    prompt = _PROMPT.format(
        artists="\n".join(f"- {a}（{n}）" for a, n in artists),
        liked=_songs(state.liked),
        declined=_songs(state.declined),
        missing=_songs(state.missing),
        count=CANDIDATES,
    )
    try:
        raw = await agent.backend.complete(prompt, max_tokens=1000)
    except Exception as e:  # noqa: BLE001
        logger.warning("rest おすすめの依頼に失敗（次の晩に持ち越す）: %s", e)
        return False
    got_all = _candidates(str(raw or ""))[:CANDIDATES]
    if not got_all:
        logger.warning("rest おすすめの返りを読めなかった")
        return False
    known = {str(t.get("uri") or "") for t in catalog.playlist_tracks()}
    known |= set(state.offered_uris) | {str(d.get("uri") or "") for d in state.declined}
    absent = already = 0
    for n, got in enumerate(got_all, 1):
        # 上から順に確かめ、最初に「実在して、まだ知らない曲」を採る（知-aq）。無かった曲は覚えて次の晩に避けさせる。
        hits = await asyncio.to_thread(web.search, f"{got['title']} {got['artist']}", "track")
        pick = next(
            (
                h
                for h in hits or []
                if music_rules.same_name(h.get("title", ""), got["title"])
                and music_rules.same_name(h.get("artist", ""), got["artist"])
            ),
            None,
        )
        if pick is None:
            absent += 1
            state.missing = [{"title": got["title"], "artist": got["artist"]}] + [
                m
                for m in state.missing
                if (m.get("title"), m.get("artist")) != (got["title"], got["artist"])
            ]
            state.missing = state.missing[: ms.MISSING_MAX]
            continue
        if pick["uri"] in known:
            already += 1
            continue
        state.candidate = ms.Candidate(got["title"], got["artist"], str(pick["uri"]), got["reason"])
        ms.store(state)
        logger.info(
            "rest おすすめを用意した：「%s」（%s）・%d 件中 %d 件目",
            got["title"],
            got["artist"],
            len(got_all),
            n,
        )
        return True
    ms.store(state)
    logger.info(
        "rest おすすめを用意できなかった：%d 件中 Spotify に無かった %d・知っている曲 %d",
        len(got_all),
        absent,
        already,
    )
    return False
