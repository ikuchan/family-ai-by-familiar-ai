"""REST 内省・季節の層（知-ac 段 2・2026-09-26・`設計方針_季節の層` v0.1）。

晩に 1 回、`ME.md` の住所で検索し、検索結果とその晩の出来事から、天気・まわり・家の話題を
フル LLM に書かせる。機械が検査し、通れば保存する。通らなければ前の値を残す。1 日 1 回まで。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.core import season_env as se
from familiar_agent.loop import rest_season as rs
from familiar_agent.mcp_client import CallResult

TODAY = date(2026, 9, 26)
ME = "# パジュ\n名前：パジュ\n\n## 住んでいるところ\n\n茨城県守谷市本町の家。1 階の和室に居る。\n"
SEARCH = "守谷市の天気。この数日は最高24℃前後で、朝晩は15℃ほど。守谷では金木犀が咲き始めました。"


@dataclass
class _Rec:
    obs_id: str
    kind: str
    text: str


RECORDS = [_Rec("obs-1", "day_summary", "たいきが運動会の練習の話をした")]


def _reply(**over):
    body = {
        "天気": [
            {
                "text": "この数日は最高 24℃前後、朝晩は 15℃ほど。",
                "quote": "最高24℃前後で、朝晩は15℃ほど",
            }
        ],
        "まわり": [{"text": "守谷では金木犀が咲き始めた。", "quote": "金木犀が咲き始めました"}],
        "家の話題": [{"text": "たいきの運動会が近いと話していた。", "source": "obs-1"}],
    }
    body.update(over)
    return json.dumps(body, ensure_ascii=False)


@pytest.fixture
def box(monkeypatch):
    """DB の代わり（置き場の往復は `test_season_env.py` がテスト DB で見る）。"""
    held: dict = {"env": None}
    monkeypatch.setattr(se, "stored", lambda: held["env"])
    monkeypatch.setattr(se, "store", lambda env: held.update(env=env) or True)
    return held


def _agent(reply: str, *, me: str = ME, search_ok: bool = True):
    a = MagicMock()
    a._me_md = me
    a._mcp_search = AsyncMock(return_value=CallResult(SEARCH, None, search_ok))
    a.backend.complete = AsyncMock(return_value=reply)
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-x", True))
    a._observation_perspective = MagicMock(return_value={"writer_id": "__self__"})
    return a


def _run(agent, records=RECORDS):
    return asyncio.run(rs.update_season(agent, records, today=TODAY))


# ── 住所から検索の語 ───────────────────────────────────────────────────────


def test_the_place_is_the_prefecture_and_the_city():
    assert rs.place_of(ME) == "茨城県守谷市"


def test_no_place_when_me_md_has_none():
    assert rs.place_of("# パジュ\n名前：パジュ\n") == ""


# ── 通れば保存する ─────────────────────────────────────────────────────────


def test_a_good_reply_is_stored(box):
    a = _agent(_reply())
    out = _run(a)
    assert out.applied is True
    env = box["env"]
    assert env.written_on == TODAY
    assert env.rows["まわり"][0].text == "守谷では金木犀が咲き始めた。"
    assert env.rows["家の話題"][0].source == "obs-1"
    query = a._mcp_search.await_args.args[1]["query"]
    assert query.startswith("茨城県守谷市")
    a._memory.save_async_with_id.assert_awaited()  # `内省` の記録


def test_a_quote_not_in_the_search_keeps_the_old_value(box):
    old = se.SeasonEnv(written_on=date(2026, 9, 20))
    box["env"] = old
    bad = _reply(天気=[{"text": "最高 30℃。", "quote": "最高30℃の真夏日"}])
    out = _run(_agent(bad))
    assert out.applied is False and "検索" in out.reason
    assert box["env"] is old


def test_it_writes_only_once_a_day(box):
    box["env"] = se.SeasonEnv(written_on=TODAY)
    a = _agent(_reply())
    out = _run(a)
    assert out.applied is False and "今日" in out.reason
    a._mcp_search.assert_not_awaited()
    a.backend.complete.assert_not_awaited()


def test_no_place_means_no_search(box):
    a = _agent(_reply(天気=[], まわり=[]), me="# パジュ\n")
    out = _run(a)
    a._mcp_search.assert_not_awaited()
    assert out.applied is True
    assert "天気" not in box["env"].rows or not box["env"].rows["天気"]


def test_a_failed_search_tries_the_other_tool(box):
    a = _agent(_reply())
    a._mcp_search = AsyncMock(
        side_effect=[CallResult("鍵が無い", None, False), CallResult(SEARCH, None, True)]
    )
    assert _run(a).applied is True
    tools = [c.args[0] for c in a._mcp_search.await_args_list]
    assert tools == ["brave_web_search", "tavily_search"]


def test_an_unreadable_reply_keeps_the_old_value(box):
    out = _run(_agent("わかりません"))
    assert out.applied is False and box["env"] is None
