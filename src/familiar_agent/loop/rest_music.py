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
    unread: int = 0  # 中身を読めず、前の晩の中身を残したプレイリスト
    tracks: int = 0


async def refresh_catalog(agent) -> CatalogResult:
    """目録を読み直す。Spotify を呼ぶのはスレッドで（ネットワークの待ちでループを止めない）。"""
    tool = getattr(agent, "_music_tool", None)
    web = getattr(tool, "_web", None) if tool is not None else None
    if web is None:
        return CatalogResult()
    return await asyncio.to_thread(_refresh, web)


def _refresh(web) -> CatalogResult:
    before = {p.id: p for p in mc.stored().playlists}
    result = CatalogResult()
    playlists: list[mc.Playlist] = []
    for p in web.my_playlists():
        items = web.playlist_items(p["id"])
        if items is None:
            result.unread += 1
            items = before[p["id"]].tracks if p["id"] in before else []
        playlists.append(mc.Playlist(p["id"], p["name"], p["uri"], list(items), p["mine"]))
        result.tracks += len(items)
    result.playlists = len(playlists)
    if playlists or not before:
        mc.store(
            mc.Catalog(playlists=playlists, albums=web.saved_albums(), tracks=web.saved_tracks())
        )
    logger.info(
        "rest 音楽の目録：プレイリスト %d・曲 %d（読めず前のまま %d）",
        result.playlists,
        result.tracks,
        result.unread,
    )
    return result


_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、家族に 1 曲すすめたい。家族のプレイリストに入っている
アーティストを、多く入っている順に並べる（括弧の数は入っている曲の数）。

この好みに合いそうで、**プレイリストに入っていない曲**を 1 つ選び、選んだ理由を添える。プレイリストの
アーティストの別の曲でも、似たテイストの別のアーティストの曲でもよい。実在する曲だけを選ぶ（曲名と
アーティスト名は、Spotify で検索して見つかる正確な表記で）。前に気に入ってもらえた曲は好みの手がかりに、
断られた曲は避ける手がかりにする。

出力は次の JSON だけ（ほかには何も書かない）：
{{"title": "曲名", "artist": "アーティスト名", "reason": "選んだ理由（1〜2 文）"}}

[プレイリストのアーティスト]
{artists}

[前に気に入ってもらえた曲]
{liked}

[断られた曲]
{declined}
"""


def _parse(raw: str) -> "dict | None":
    data = read_json_merged(raw)  # 2 つに分けた・書き直した返りも読む（記-p）
    if data is None:
        return None
    out = {k: str(data.get(k, "")).strip() for k in ("title", "artist", "reason")}
    return out if all(out.values()) else None


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
    )
    try:
        raw = await agent.backend.complete(prompt, max_tokens=400)
    except Exception as e:  # noqa: BLE001
        logger.warning("rest おすすめの依頼に失敗（次の晩に持ち越す）: %s", e)
        return False
    got = _parse(str(raw or ""))
    if got is None:
        logger.warning("rest おすすめの返りを読めなかった")
        return False
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
        logger.info(
            "rest おすすめ「%s」（%s）は実在を確かめられなかった", got["title"], got["artist"]
        )
        return False
    known = {str(t.get("uri") or "") for t in catalog.playlist_tracks()}
    known |= set(state.offered_uris) | {str(d.get("uri") or "") for d in state.declined}
    if pick["uri"] in known:
        logger.info("rest おすすめ「%s」はもう知っている曲なので見送る", got["title"])
        return False
    state.candidate = ms.Candidate(got["title"], got["artist"], str(pick["uri"]), got["reason"])
    ms.store(state)
    logger.info("rest おすすめを用意した：「%s」（%s）", got["title"], got["artist"])
    return True
