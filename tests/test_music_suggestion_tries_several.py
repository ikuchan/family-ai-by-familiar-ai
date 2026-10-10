"""おすすめは 1 回の頼みで 5 候補を挙げてもらい、上から確かめて最初に通ったものを採る（知-aq・2026-10-10・本人の決定ア）。

晩に 1 候補だけ頼み、外れたらその晩は終わりだった。10/09・10/10 の 2 晩とも「灯」（Mr.Children）を挙げ、Spotify に
その曲は無く（検索は「Again」「産声」「しるし」…）、おすすめが一度も用意されなかった。外れを覚えていないので、同じ頼み
で同じ答えが返る。候補を 5 つ挙げてもらい、上から「実在して、まだ知らない曲」を確かめる。Spotify に無かった曲は
`missing` に覚え（20 件〔仮〕まで）、次の晩の頼みに `[無かった曲]` として渡して避けさせる。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import music_catalog as mc
from familiar_agent.core import music_suggestion as ms
from familiar_agent.loop.rest_music import prepare_suggestion

CATALOG = mc.Catalog(
    playlists=[
        mc.Playlist(
            "p1",
            "夜のドライブ",
            "spotify:playlist:p1",
            [{"title": "新宝島", "artist": "サカナクション", "uri": "spotify:track:shin"}],
        )
    ]
)


@pytest.fixture
def state(monkeypatch):
    data: dict = {"s": ms.Suggestions()}
    monkeypatch.setattr(ms, "stored", lambda: data["s"])
    monkeypatch.setattr(ms, "store", lambda s: data.__setitem__("s", s) or True)
    monkeypatch.setattr(mc, "stored", lambda: CATALOG)
    return data


def _reply(*titles: str) -> str:
    return json.dumps(
        {
            "candidates": [
                {"title": t, "artist": "サカナクション", "reason": f"{t} が合う"} for t in titles
            ]
        },
        ensure_ascii=False,
    )


def _agent(reply: str, found: "dict[str, str]"):
    """`found` は Spotify に有る曲名 → URI。無い曲名の検索は空を返す。"""
    a = MagicMock()
    a.backend.complete = AsyncMock(return_value=reply)

    def search(query, kind):
        title = query.rsplit(" ", 1)[0]
        uri = found.get(title)
        return [{"title": title, "artist": "サカナクション", "uri": uri}] if uri else []

    a._music_tool._web.search = MagicMock(side_effect=search)
    return a


def test_the_first_real_and_new_one_is_taken(state):
    a = _agent(
        _reply("灯", "新宝島", "忘れられないの"),
        {"新宝島": "spotify:track:shin", "忘れられないの": "spotify:track:wasure"},
    )
    assert asyncio.run(prepare_suggestion(a)) is True
    s = state["s"]
    assert s.candidate is not None and s.candidate.uri == "spotify:track:wasure"
    assert s.missing == [{"title": "灯", "artist": "サカナクション"}]


def test_all_off_prepares_nothing_but_remembers_the_missing(state):
    a = _agent(_reply("灯", "幻の曲", "新宝島"), {"新宝島": "spotify:track:shin"})
    assert asyncio.run(prepare_suggestion(a)) is False
    assert state["s"].candidate is None
    assert [m["title"] for m in state["s"].missing] == ["幻の曲", "灯"]  # 新しい順


def test_the_missing_are_told_to_the_llm(state):
    state["s"] = ms.Suggestions(missing=[{"title": "灯", "artist": "Mr.Children"}])
    a = _agent(_reply("忘れられないの"), {"忘れられないの": "spotify:track:wasure"})
    asyncio.run(prepare_suggestion(a))
    prompt = a.backend.complete.await_args.args[0]
    assert "[無かった曲]" in prompt and "灯（Mr.Children）" in prompt


def test_the_missing_are_kept_up_to_twenty(state):
    state["s"] = ms.Suggestions(missing=[{"title": f"古い{i}", "artist": "X"} for i in range(20)])
    a = _agent(_reply("灯"), {})
    asyncio.run(prepare_suggestion(a))
    missing = state["s"].missing
    assert len(missing) == 20 and missing[0]["title"] == "灯" and missing[-1]["title"] == "古い18"


def test_the_missing_survive_a_round_trip():
    s = ms.Suggestions(missing=[{"title": "灯", "artist": "Mr.Children"}])
    assert ms._from_json(json.loads(json.dumps(ms._to_json(s)))).missing == s.missing
