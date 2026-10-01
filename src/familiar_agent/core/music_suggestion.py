"""音楽のおすすめ（知-aa 段 4・2026-10-02・本人の決定）。

パジュは自分で音楽を**かけない**（本人の決定）。REST の晩に候補を 1 件だけ用意し、話しかけてよいときに 1 日 1 回まで
「こんな曲あるけどどう？」と聞く。「いいね」ならかけ、「いらない」ならその曲は二度と勧めない（アーティストは避けない）、
返事が無ければ別の日にもう一度だけ勧める。

DB（`agent_state` の鍵 `music_suggestion`）に持つのは、いまの候補（勧めた回数つき）、最後に勧めた日、気に入った曲、
断った曲、前に勧めた曲の URI。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import state_json

STATE_KEY = "music_suggestion"
#: 返事が無いまま勧めてよい回数。2 回目でも返事が無ければ、その候補は捨てる（本人の決定：もう一度だけ）。
MAX_OFFERS = 2


@dataclass
class Candidate:
    title: str
    artist: str
    uri: str
    reason: str
    offered: int = 0  # 勧めた回数


@dataclass
class Suggestions:
    candidate: "Candidate | None" = None
    last_offered_on: str = ""  # YYYY-MM-DD（1 日 1 回まで）
    liked: "list[dict]" = field(default_factory=list)
    declined: "list[dict]" = field(default_factory=list)
    offered_uris: "list[str]" = field(default_factory=list)


def _to_json(s: Suggestions) -> dict:
    d = asdict(s)
    return d


def _from_json(raw: object) -> Suggestions:
    if not isinstance(raw, dict):
        return Suggestions()
    c = raw.get("candidate")
    cand = None
    if isinstance(c, dict) and c.get("uri"):
        cand = Candidate(
            title=str(c.get("title", "")),
            artist=str(c.get("artist", "")),
            uri=str(c["uri"]),
            reason=str(c.get("reason", "")),
            offered=int(c.get("offered", 0) or 0),
        )
    return Suggestions(
        candidate=cand,
        last_offered_on=str(raw.get("last_offered_on", "") or ""),
        liked=list(raw.get("liked") or []),
        declined=list(raw.get("declined") or []),
        offered_uris=[str(u) for u in raw.get("offered_uris") or []],
    )


def stored() -> Suggestions:
    return _from_json(state_json.read(STATE_KEY))


def store(s: Suggestions) -> bool:
    return state_json.write(STATE_KEY, _to_json(s))


def clear() -> bool:
    return state_json.clear(STATE_KEY)
