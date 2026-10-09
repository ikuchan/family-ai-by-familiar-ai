"""音楽のおすすめ（知-aa 段 4・2026-10-02・本人の決定）。

パジュは自分で音楽を**かけない**（本人の決定）。REST の晩に候補を 1 件だけ用意し、話しかけてよいときに 1 日 1 回まで
「こんな曲あるけどどう？」と聞く。「いいね」ならかけ、「いらない」ならその曲は二度と勧めない（アーティストは避けない）、
返事が無ければ別の日にもう一度だけ勧める。

DB（`agent_state` の鍵 `music_suggestion`）に持つのは、いまの候補（勧めた回数つき）、最後に勧めた日、気に入った曲、
断った曲、前に勧めた曲の URI。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

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


HEADING = "[音楽のおすすめ]"


def today() -> str:
    """いまの日付（その日に勧めたかを見る）。試験で差し替えられるよう口にしておく。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def frame(s: Suggestions, *, today: str, talking: bool, conversation: bool) -> str:
    """主LLM のシステム文に載せる枠。載せなければ空。

    - 話しかけてよいとき（bond・esteem の発火）で、その日にまだ勧めていなければ、勧める材料を載せる。
    - 勧めた日の会話では、何を勧めたかと、返事の受け方を載せる。
    """
    c = s.candidate
    if c is None:
        return ""
    song = f"「{c.title}」（{c.artist}）"
    if talking and s.last_offered_on != today and c.offered < MAX_OFFERS:
        return (
            f"{HEADING}\n家族にすすめたい曲がある：{song}。選んだ理由：{c.reason}\n"
            "話しかけるなら、「こんな曲あるけどどう？」と、曲名とアーティストと理由を添えて聞いてよい"
            "（聞かなくてもよい。かけるのは返事を聞いてから）。"
        )
    if conversation and awaiting_reply(s, today=today):
        # 返事は調停（Jev が意味と動作で）が受ける（出-ay 段 4-4f）。主LLM には何をすすめたかだけを渡す。
        return f"{HEADING}\nさっき {song} をすすめた。"
    return ""


def awaiting_reply(s: Suggestions, *, today: str) -> bool:
    """すすめた曲への返事を待っているか（候補があり、今日すすめた）。調停が返事の意味を並べるのに使う（出-ay 段 4-4f）。"""
    c = s.candidate
    return c is not None and c.offered > 0 and s.last_offered_on == today


def mark_offered(s: Suggestions, said: str, *, today: str) -> bool:
    """声にした返事に曲名が入っていたら「勧めた」と印をつける。つけたら True。"""
    from .music_rules import _norm

    c = s.candidate
    if (
        c is None
        or s.last_offered_on == today
        or not _norm(c.title)
        or _norm(c.title) not in _norm(said)
    ):
        return False
    c.offered += 1
    s.last_offered_on = today
    if c.uri not in s.offered_uris:
        s.offered_uris.append(c.uri)
    return True


def drop_if_unanswered(s: Suggestions) -> None:
    """返事が無いまま `MAX_OFFERS` 回勧めた候補は捨てる（晩に次を用意する）。"""
    if s.candidate is not None and s.candidate.offered >= MAX_OFFERS:
        s.candidate = None
