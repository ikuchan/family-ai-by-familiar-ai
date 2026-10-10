"""音楽の目録（知-aa 段 3 の土台・2026-10-01・本人の決定ア）。

あなたの Spotify のプレイリストの中身（曲・アーティスト・URI）と、保存したアルバムと曲を、REST の晩に読み直して
DB（`agent_state` の鍵 `music_catalog`）に持つ。会話のとき（「〇〇かけて」）と、おすすめの候補を作るときは、ここを
引くだけにする（頼まれるたびに Spotify を数十回呼ぶと、鳴るまで待たせる）。`MUSIC.md` は人の入力のまま触らない。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import state_json

STATE_KEY = "music_catalog"


@dataclass
class Playlist:
    id: str
    name: str
    uri: str
    tracks: "list[dict]" = field(default_factory=list)  # {"title","artist","uri"}
    mine: bool = True
    # 中身を読めないと分かった日（YYYY-MM-DD・空なら読める）。他人のプレイリストは Spotify の決まりで中身を読めない
    # ことがある（403）ので、覚えて `UNREADABLE_RETRY_DAYS` 日は読みに行かない（知-ao）。
    unreadable_since: str = ""


@dataclass
class Catalog:
    playlists: "list[Playlist]" = field(default_factory=list)
    albums: "list[dict]" = field(default_factory=list)  # {"title","artist","uri"}
    tracks: "list[dict]" = field(default_factory=list)  # 保存した曲

    def playlist_artists(self) -> "list[tuple[str, int]]":
        """プレイリストに入っているアーティストと、入っている曲の数。多い順。"""
        counts: dict[str, int] = {}
        for p in self.playlists:
            for t in p.tracks:
                a = str(t.get("artist") or "").strip()
                if a:
                    counts[a] = counts.get(a, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    def playlist_tracks(self) -> "list[dict]":
        """プレイリストに入っている曲（重なりは 1 つに）。"""
        seen: set[str] = set()
        out: list[dict] = []
        for p in self.playlists:
            for t in p.tracks:
                uri = str(t.get("uri") or "")
                if uri and uri not in seen:
                    seen.add(uri)
                    out.append(t)
        return out


def _to_json(cat: Catalog) -> dict:
    return {
        "playlists": [
            {
                "id": p.id,
                "name": p.name,
                "uri": p.uri,
                "tracks": p.tracks,
                "mine": p.mine,
                "unreadable_since": p.unreadable_since,
            }
            for p in cat.playlists
        ],
        "albums": cat.albums,
        "tracks": cat.tracks,
    }


def _from_json(raw: object) -> Catalog:
    if not isinstance(raw, dict):
        return Catalog()
    pls = []
    for p in raw.get("playlists") or []:
        if isinstance(p, dict) and p.get("uri"):
            pls.append(
                Playlist(
                    id=str(p.get("id", "")),
                    name=str(p.get("name", "")),
                    uri=str(p["uri"]),
                    tracks=list(p.get("tracks") or []),
                    mine=bool(p.get("mine", True)),
                    unreadable_since=str(p.get("unreadable_since", "") or ""),
                )
            )
    return Catalog(
        playlists=pls, albums=list(raw.get("albums") or []), tracks=list(raw.get("tracks") or [])
    )


def stored() -> Catalog:
    """DB にある目録。無ければ空の目録。"""
    return _from_json(state_json.read(STATE_KEY))


def store(cat: Catalog) -> bool:
    return state_json.write(STATE_KEY, _to_json(cat))


def clear() -> bool:
    return state_json.clear(STATE_KEY)
