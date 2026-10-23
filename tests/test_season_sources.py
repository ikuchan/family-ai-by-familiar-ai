"""季節の層が毎晩読むページを、機械が読む部品（知-ac・2026-10-24・`設計方針_季節の層` v0.4）。

返りの文は 10/23〜10/24 に fetch で読んだ本物のページを短く切ったもの。試験から外へは出ない。
"""

from __future__ import annotations

from datetime import date

from familiar_agent.core import season_env as se
from familiar_agent.core import season_sources as ss

PINPOINT = """\
## ピンポイント天気

2026年10月23日　8時00分発表

### 今日の天気 - 10月23日(金)

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 時刻 | 0時 | 3時 | 6時 | 9時 | 12時 | 15時 | 18時 | 21時 |
| 天気 | 晴れ  晴れ | 晴れ  晴れ | 晴れ  晴れ | 晴れ  晴れ | 晴れ  晴れ | 曇り  曇り | 曇り  曇り | 曇り  曇り |
| 気温（℃） | 13 | 11 | 11 | 16 | 19 | 19 | 17 | 15 |
| 降水量（mm） | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## AI解説

昼間は穏やかな晴天で過ごしやすく、夕方からは次第に雲が広がる見込みです。

### 明日の天気 - 10月24日(土)

|  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 時刻 | 0時 | 3時 | 6時 | 9時 | 12時 | 15時 | 18時 | 21時 |
| 天気 | 曇り  曇り | 曇り  曇り | 曇り  曇り | 雨  雨 | 雨  雨 | 雨  雨 | 雨  雨 | 曇り  曇り |
| 気温（℃） | 15 | 15 | 14 | 16 | 16 | 17 | 16 | 16 |
| 湿度（％） | 80 | 83 | 84 | 79 | 91 | 95 | 100 | 98 |
| 降水量（mm） | 0 | 0 | 0.2 | 0.8 | 3 | 3 | 1 | 0.7 |

## 週間天気

2026年10月23日 8時00分発表

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| 日付 | 10月25日  (日) | 10月26日  (月) | 10月27日  (火) | 10月28日  (水) | 10月29日  (木) | 10月30日  (金) |
| 天気 | 曇時々雨  曇時々雨 | 曇時々雨  曇時々雨 | 曇時々晴  曇時々晴 | 晴れ  晴れ | 曇時々晴  曇時々晴 | 曇一時雨  曇一時雨 |
| 気温（℃） | 21  15 | 23  14 | 21  12 | 20  9 | 17  8 | 20  11 |
| 降水 確率（％） | 60 | 100 | 30 | 10 | 40 | 70 |
"""

NEWS_LIST = (
    '<h2>お天気ニュース</h2><div><a href="/news/202610/220291/" class=""><figure>'
    '<img src="x.jpg" alt="北海道で初冠雪"><div><p data-text="title">北海道の手稲山で初冠雪</p>'
    '<div><p data-tag="season">季節・暦</p></div><p data-text="date">2026.10.22 18:59</p>'
    '</div></figure></a><a href="/news/202610/220281/" class=""><div>'
    '<p data-text="title">週末から週明けは再び蒸し暑い　来週後半は強い寒気</p>'
    '<p data-text="date">2026.10.22 18:30</p></div></a>'
    '<a href="/news/202610/210101/" class=""><div><p data-text="title">＜PR＞当たり前を守る「インフラ」</p>'
    '<p data-text="date">2026.10.21 17:57</p></div></a>'
    '<a href="/news/202610/220281/" class="" data-v-b4907be8><div data-v-b4907be8>'
    '<p data-text="title" data-v-b4907be8>週末から週明けは再び蒸し暑い　来週後半は強い寒気</p>'
    '<p data-text="date" data-v-b4907be8>2026.10.22 18:30</p></div></a>'
)

COLUMN_LIST = """\
## お天気コラム

## 天気の知識

* [![](https://s.yimg.jp/a.jpg)紅葉が色づく条件](https://weather.yahoo.co.jp/weather/promo/column/knowledge/kouyou.html)
* [![](https://s.yimg.jp/b.jpg)木枯らし1号とは](https://weather.yahoo.co.jp/weather/promo/column/knowledge/kogarashi.html)

## 二十四節気

* [寒露](https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-kanro.html)
* [霜降](https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-soukou.html)
* [小雪](https://weather.yahoo.co.jp/weather/promo/column/season/solarterm-shousetu.html)
* [![](https://s.yimg.jp/c.jpg)二十四節気で季節に親しむ](https://weather.yahoo.co.jp/weather/promo/column/season/nijuushisekki.html)

## 歳時記

* [![](https://s.yimg.jp/d.jpg)十三夜](https://weather.yahoo.co.jp/weather/promo/column/calendar/jusanya.html)
"""


# ── 天気：守谷市のページの表を機械が読む ─────────────────────────────────────


def test_weather_rows_are_tomorrow_and_the_day_after():
    rows = ss.weather_rows(PINPOINT)
    assert [r.text for r in rows] == [
        "10/24（土）：曇り→雨→曇り、14〜17℃、雨は6〜21時",
        "10/25（日）：曇時々雨、15〜21℃、雨60%",
    ]


def test_a_dry_day_says_no_rain():
    dry = PINPOINT.replace(
        "| 0 | 0 | 0.2 | 0.8 | 3 | 3 | 1 | 0.7 |", "| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
    )
    assert ss.weather_rows(dry)[0].text.endswith("雨なし")


def test_a_changeable_day_still_fits_in_a_row():
    changeable = PINPOINT.replace(
        "| 曇り  曇り | 曇り  曇り | 曇り  曇り | 雨  雨 | 雨  雨 | 雨  雨 | 雨  雨 | 曇り  曇り |",
        "| 晴れ  晴れ | 曇り  曇り | 雨  雨 | 雪  雪 | みぞれ  みぞれ | 雨  雨 | 曇り  曇り | 晴れ  晴れ |",
    )
    row = ss.weather_rows(changeable)[0].text
    assert len(row) <= se.MAX_CHARS
    assert row.startswith("10/24（土）：晴れ→曇り→…→晴れ")


def test_an_unknown_page_gives_no_weather():
    assert ss.weather_rows("## ログインしてください") == ()


# ── ウェザーニュースの一覧 ───────────────────────────────────────────────────


def test_news_items_have_title_url_and_date_without_pr_or_repeats():
    items = ss.news_items(NEWS_LIST)
    assert [(i.title, i.url, i.date) for i in items] == [
        (
            "北海道の手稲山で初冠雪",
            "https://weathernews.jp/news/202610/220291/",
            "2026.10.22 18:59",
        ),
        (
            "週末から週明けは再び蒸し暑い　来週後半は強い寒気",
            "https://weathernews.jp/news/202610/220281/",
            "2026.10.22 18:30",
        ),
    ]


# ── Yahoo!のお天気コラムの一覧 ────────────────────────────────────────────────


def test_column_links_are_grouped_by_section():
    got = ss.column_links(COLUMN_LIST)
    assert [i.title for i in got["天気の知識"]] == ["紅葉が色づく条件", "木枯らし1号とは"]
    assert [i.title for i in got["歳時記"]] == ["十三夜"]
    assert got["歳時記"][0].url.endswith("/calendar/jusanya.html")


def test_the_solar_term_page_is_found_by_its_name():
    got = ss.column_links(COLUMN_LIST)
    assert ss.solar_term_page(got, "霜降").endswith("solarterm-soukou.html")
    assert ss.solar_term_page(got, "小雪").endswith("solarterm-shousetu.html")
    assert ss.solar_term_page(got, "二十四節気で季節に親しむ") is None
    assert ss.solar_term_page(got, "大寒") is None


# ── 直前の節気 ──────────────────────────────────────────────────────────────


def test_the_last_solar_term():
    assert se.last_term(date(2026, 10, 23)) == "寒露"
    assert se.last_term(date(2026, 10, 24)) == "霜降"
    assert se.last_term(date(2027, 1, 3)) == "冬至"


# ── Jev が読む記事を選ぶ ─────────────────────────────────────────────────────

import asyncio  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from familiar_agent.backends.jev import JevAnswer  # noqa: E402
from familiar_agent.core import jev_judges  # noqa: E402

NEWS = [ss.Link(f"記事{n}", f"https://weathernews.jp/news/202610/{n}/") for n in (1, 2, 3)]
COLUMNS = [
    ss.Link("紅葉が色づく条件", "https://k/kouyou.html"),
    ss.Link("節分", "https://k/s.html"),
]


def _jev(answers: dict, ok: bool = True):
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(return_value=JevAnswer(ok=ok, answers=answers))
    return jev


def _reads(jev, news=NEWS, columns=COLUMNS):
    return asyncio.run(
        jev_judges.judge_season_reads(
            jev, today=date(2026, 10, 24), place="茨城県守谷市", news=news, columns=columns
        )
    )


def _probs(**p):
    return {"choice": max(p, key=p.get), "confidence": max(p.values()), "probabilities": p}


def test_jev_sees_the_date_the_term_the_place_and_the_titles():
    jev = _jev({"news": _probs(n1=0.6, none=0.4), "column": _probs(c1=0.7, none=0.3)})
    _reads(jev)
    state, questions = jev.ask.await_args.args
    assert "2026-10-24" in state and "霜降" in state and "茨城県守谷市" in state
    assert questions["news"]["criteria"]["n2"] == "記事2"
    assert "none" in questions["news"]["criteria"] and "none" in questions["column"]["criteria"]


def test_the_second_article_is_read_only_above_none():
    jev = _jev(
        {"news": _probs(n2=0.5, n3=0.3, n1=0.05, none=0.15), "column": _probs(none=0.9, c1=0.1)}
    )
    news, columns = _reads(jev)
    assert [n.title for n in news] == ["記事2", "記事3"]
    assert columns == []
    jev = _jev({"news": _probs(n2=0.5, n3=0.1, none=0.4), "column": _probs(c2=0.6, none=0.4)})
    news, columns = _reads(jev)
    assert [n.title for n in news] == ["記事2"]
    assert [c.title for c in columns] == ["節分"]


def test_none_on_top_reads_nothing():
    jev = _jev({"news": _probs(none=0.5, n1=0.3, n2=0.2), "column": _probs(none=0.8, c1=0.2)})
    assert _reads(jev) == ([], [])


def test_without_jev_nothing_is_picked():
    """**反証の側**：使えない・失敗・確率の無い答えは何も選ばない（倒し先：節気のページだけ読む）。"""
    down = MagicMock()
    down.available = False
    assert _reads(down) == ([], [])
    assert _reads(_jev({}, ok=False)) == ([], [])
    assert _reads(_jev({"news": {"choice": "n1", "confidence": 0.9}})) == ([], [])


def test_no_titles_means_no_question():
    jev = _jev({})
    assert _reads(jev, news=[], columns=[]) == ([], [])
    jev.ask.assert_not_awaited()
