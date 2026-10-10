"""いま鳴っている曲を答える文（出-bf・2026-10-10・本人の決定ア）。

「いま何の曲かけてる？」に、機械が決まった形で答える。曲名を取り違えず、軽量LLM を待たない。アーティストは
Spotify から来た表記のまま。様子は `io/music.status` の辞書（鳴っているか・曲名・アーティスト）。読めなければ None。
"""

from __future__ import annotations


def text_for(status: "dict | None") -> str:
    """いまの様子 → 答えの文。"""
    if status is None:
        return "いま何が鳴ってるか、分からなかった"
    title = str(status.get("title") or "").strip()
    if not status.get("playing") or not title:
        return "いまは何も鳴ってないよ"
    artist = str(status.get("artist") or "").strip()
    return f"いまは『{title}』、{artist}だよ" if artist else f"いまは『{title}』だよ"
