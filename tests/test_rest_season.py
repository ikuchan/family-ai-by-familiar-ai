"""REST 内省・季節の層（知-ac・2026-10-24・`設計方針_季節の層` v0.4）。

晩に 1 回、決まったページを fetch で読む。天気は `ME.md` の「天気のページ」の表を機械が読み（明日と明後日）、
まわりはウェザーニュースと Yahoo!のお天気コラムから Jev が選んだ記事と節気のページを、家の話題はその晩の
出来事を、フル LLM に書かせる。何も読めなければ書かず、前の値を残す。1 日 1 回まで。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import season_env as se
from familiar_agent.loop import rest_season as rs
from familiar_agent.mcp_client import CallResult

TODAY = date(2026, 10, 24)
WEATHER_URL = "https://weather.example/jp/1/2/3.html"
ME = (
    "# パジュ\n名前：パジュ\n\n## 住んでいるところ\n\n茨城県守谷市本町の家。1 階の和室に居る。\n"
    f"天気のページ：{WEATHER_URL}\n"
)
PINPOINT = """\
### 明日の天気 - 10月25日(日)

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| 時刻 | 0時 | 6時 | 12時 | 18時 |
| 天気 | 曇り  曇り | 雨  雨 | 雨  雨 | 曇り  曇り |
| 気温（℃） | 15 | 14 | 19 | 21 |
| 降水量（mm） | 0 | 1 | 3 | 0 |

## 週間天気

| 日付 | 10月26日  (月) |
| 天気 | 曇時々雨  曇時々雨 |
| 気温（℃） | 23  14 |
| 降水 確率（％） | 100 |
"""
NEWS_LIST = (
    '<a href="/news/202610/220281/"><p data-text="title">来週後半は強い寒気</p>'
    '<p data-text="date">2026.10.22 18:30</p></a>'
)
COLUMNS = """\
## 天気の知識

* [紅葉が色づく条件](https://weather.yahoo.co.jp/weather/promo/column/knowledge/kouyou.html)

## 二十四節気

* [霜降](https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-soukou.html)
"""
PAGES = {
    WEATHER_URL: PINPOINT,
    rs.NEWS_LIST_URL: NEWS_LIST,
    rs.COLUMN_LIST_URL: COLUMNS,
    "https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-soukou.html": "霜が降り始めるころ。",
    "https://weathernews.jp/news/202610/220281/": "来週後半は強い寒気が南下します。",
    "https://weather.yahoo.co.jp/weather/promo/column/knowledge/kouyou.html": "紅葉は朝の冷え込みで色づく。",
}


@dataclass
class _Rec:
    obs_id: str
    kind: str
    text: str


RECORDS = [_Rec("obs-1", "day_summary", "たいきが運動会の練習の話をした")]


def _reply(**over):
    body = {
        "まわり": [{"text": "来週後半は強い寒気が来るらしい。"}],
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


def _pages(pages: dict):
    async def fetch(tool, args):
        assert tool == "fetch"
        text = pages.get(args["url"])
        return CallResult(text or "読めない", None, text is not None)

    return AsyncMock(side_effect=fetch)


def _jev(news: str = "n1", column: str = "c1"):
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True,
            answers={
                "news": {"probabilities": {news: 0.7, "none": 0.3}},
                "column": {"probabilities": {column: 0.6, "none": 0.4}},
            },
        )
    )
    return jev


def _agent(reply: str, *, me: str = ME, pages: "dict | None" = None, jev=None):
    a = MagicMock()
    a._me_md = me
    a._mcp_search = _pages(PAGES if pages is None else pages)
    a._jev = jev if jev is not None else _jev()
    a.backend.complete = AsyncMock(return_value=reply)
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-x", True))
    a._observation_perspective = MagicMock(return_value={"writer_id": "__self__"})
    return a


def _run(agent, records=RECORDS):
    return asyncio.run(rs.update_season(agent, records, today=TODAY))


def _read_urls(agent):
    return [c.args[1]["url"] for c in agent._mcp_search.await_args_list]


# ── `ME.md` から住所と天気のページ ─────────────────────────────────────────


def test_the_place_is_the_prefecture_and_the_city():
    assert rs.place_of(ME) == "茨城県守谷市"


def test_no_place_when_me_md_has_none():
    assert rs.place_of("# パジュ\n名前：パジュ\n") == ""


def test_the_weather_page_is_read_from_me_md():
    assert rs.weather_url_of(ME) == WEATHER_URL
    assert rs.weather_url_of("# パジュ\n") == ""


# ── 読んで書く ──────────────────────────────────────────────────────────────


def test_a_good_night_is_stored(box):
    a = _agent(_reply(天気=[{"text": "LLM が書いた天気"}]))
    out = _run(a)
    assert out.applied is True
    env = box["env"]
    assert env.written_on == TODAY
    assert [r.text for r in env.rows["天気"]] == [
        "10/25（日）：曇り→雨→曇り、14〜21℃、雨は6〜12時",
        "10/26（月）：曇時々雨、14〜23℃、雨100%",
    ]  # 天気は機械が書き、LLM の天気は使わない
    assert env.rows["まわり"][0].text == "来週後半は強い寒気が来るらしい。"
    assert env.rows["家の話題"][0].source == "obs-1"
    a._memory.save_async_with_id.assert_awaited()  # `内省` の記録


def test_the_picked_pages_reach_the_llm(box):
    a = _agent(_reply())
    _run(a)
    urls = _read_urls(a)
    assert urls[:3] == [WEATHER_URL, rs.NEWS_LIST_URL, rs.COLUMN_LIST_URL]
    assert set(urls[3:]) == {
        "https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-soukou.html",
        "https://weathernews.jp/news/202610/220281/",
        "https://weather.yahoo.co.jp/weather/promo/column/knowledge/kouyou.html",
    }
    news_list = a._mcp_search.await_args_list[1].args[1]
    assert news_list.get("raw") is True  # 一覧は元の文から題名を取る
    prompt = a.backend.complete.await_args.args[0]
    for text in ("霜が降り始めるころ", "強い寒気が南下", "朝の冷え込みで色づく", "obs-1"):
        assert text in prompt


def test_without_jev_only_the_solar_term_page_is_read(box):
    down = MagicMock()
    down.available = False
    a = _agent(_reply(), jev=down)
    assert _run(a).applied is True
    assert _read_urls(a)[3:] == [
        "https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-soukou.html"
    ]


def test_no_weather_page_means_no_weather(box):
    a = _agent(_reply(), me="# パジュ\n茨城県守谷市本町の家。\n")
    assert _run(a).applied is True
    assert WEATHER_URL not in _read_urls(a)
    assert not box["env"].rows.get("天気")


def test_an_unreadable_weather_page_leaves_weather_empty(box, caplog):
    pages = {**PAGES, WEATHER_URL: "## ログインしてください"}
    with caplog.at_level("INFO"):
        assert _run(_agent(_reply(), pages=pages)).applied is True
    assert not box["env"].rows.get("天気")
    assert "天気を読めなかった" in caplog.text


def test_nothing_read_keeps_the_old_value_and_retries(box):
    old = se.SeasonEnv(written_on=date(2026, 10, 20))
    box["env"] = old
    a = _agent(_reply(), pages={})
    out = _run(a)
    assert out.applied is False and "読めなかった" in out.reason
    assert box["env"] is old  # 書いた印が付かないので、次の REST でやり直す
    a.backend.complete.assert_not_awaited()


def test_it_writes_only_once_a_day(box):
    box["env"] = se.SeasonEnv(written_on=TODAY)
    a = _agent(_reply())
    out = _run(a)
    assert out.applied is False and "今日" in out.reason
    a._mcp_search.assert_not_awaited()
    a.backend.complete.assert_not_awaited()


def test_an_unreadable_reply_keeps_the_old_value(box):
    out = _run(_agent("わかりません"))
    assert out.applied is False and box["env"] is None
