"""REST 内省・季節の層（知-ac・2026-09-26・2026-10-24 に材料を替えた・`設計方針_季節の層` v0.4）。

層 1（出来事を畳む）の後・層 2（自己像）の前に回り、**いまの季節と家のまわり**を書き直す。

1. 今日もう書いていれば何もしない（1 日 1 回まで）。
2. 決まったページを fetch で読む（本人の決定 2026-10-24）：
   - 天気：`ME.md` の「天気のページ」（Yahoo!天気の住まいの市）。表を**機械が**読み、明日と明後日を日付つきで書く。
   - まわり：ウェザーニュースのニュース一覧と Yahoo!のお天気コラムの一覧から、Jev が題名で選んだ記事
     （ウェザーニュース 2 本まで・コラム 1 本まで）と、いまの節気のページ（機械が名前で決める）。
3. フル LLM に読んだページとその晩の出来事を渡し、まわりと家の話題を書かせる（暦と天気は機械）。
4. 機械が検査する（`core.season_env.check`・欄・字数・家の話題の出典）。通れば保存し `内省` の記録と計測ログに
   残す。**通らなければ・何も読めなければ前の値を残す**（書いた印が付かないので次の REST でやり直す）。

まとめ方は LLM、通すかどうかは機械（層 1・層 2 と同じ分担）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

from ..core.structured_ask import read_json_merged
from ..core import jev_judges, measure
from ..core import season_env as se
from ..core import season_sources as ss

logger = logging.getLogger(__name__)

NEWS_LIST_URL = "https://weathernews.jp/news/"
COLUMN_LIST_URL = "https://weather.yahoo.co.jp/weather/promo/column/"
#: fetch の返りの上限。既定（5,000 字）では表や一覧が切れる。一覧の元の文は約 18 万字（10/22）。
PAGE_CHARS = 20_000
RAW_LIST_CHARS = 200_000
#: プロンプトへ渡す読み物の上限（読んだページの数で等分する）。
READ_CHARS = 4000

_PLACE = re.compile(r"(\S{2,3}?[都道府県])(\S+?[市区町村])")
#: `ME.md` の「天気のページ：URL」の行（見出しの語は〔仮〕）。
_WEATHER_URL = re.compile(r"天気のページ\s*[:：]\s*(https?://\S+)")

_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、一日の終わりに、いまの季節と家のまわりの様子を書き留める。
下に、今夜読んだ天気と季節の読み物（全国や一般の話を含む）と、今夜まとめた出来事（各行の先頭が id）を示す。

次の 2 つの欄を書く。各欄 0〜{rows} 行、1 行 {chars} 字まで。書けることが無ければ空の配列にする。
- まわり：いまの季節の様子（花・虫・行事・寒さ暑さの移り変わりなど）。読み物にあることだけ。
  遠い土地の話は、そうとわかるように書く（例：北海道では初冠雪）。
- 家の話題：家族が話していた季節や行事の話。今夜の出来事にあることだけ。

家の話題の行には、根拠にした出来事の id を source に書く。暦と天気は書かない。

[住んでいる土地]
{place}

[今夜読んだ読み物]
{reads}

[今夜まとめた出来事]
{materials}

JSON だけを返す：
{{"まわり": [{{"text": "…"}}], "家の話題": [{{"text": "…", "source": "id"}}]}}
"""


@dataclass(frozen=True)
class Outcome:
    applied: bool
    reason: "str | None" = None


def place_of(me_md: str) -> str:
    """`ME.md` から都道府県と市（例：「茨城県守谷市」）。見つからなければ空。"""
    m = _PLACE.search(me_md or "")
    return m.group(1) + m.group(2) if m else ""


def weather_url_of(me_md: str) -> str:
    """`ME.md` の「天気のページ」の URL。無ければ空（天気の欄は書かない）。"""
    m = _WEATHER_URL.search(me_md or "")
    return m.group(1) if m else ""


async def _fetch(agent, url: str, *, raw: bool = False) -> str:
    """1 ページ読む。読めなければ空。"""
    args: dict = {"url": url, "max_length": RAW_LIST_CHARS if raw else PAGE_CHARS}
    if raw:
        args["raw"] = True
    try:
        r = await agent._mcp_search("fetch", args)
    except Exception:  # noqa: BLE001
        logger.warning("rest 季節の層：%s を読めなかった", url, exc_info=True)
        return ""
    if not r.ok:
        logger.warning("rest 季節の層：%s を読めなかった（%.120s）", url, r.text)
        return ""
    return str(r.text or "")


async def _weather(agent, me_md: str) -> "tuple[se.Row, ...]":
    url = weather_url_of(me_md)
    if not url:
        logger.info("rest 季節の層：ME.md に天気のページが無い")
        return ()
    rows = ss.weather_rows(await _fetch(agent, url))
    if not rows:
        logger.info("rest 季節の層：天気を読めなかった")
    return rows


async def _reads(agent, today: date, place: str) -> "list[str]":
    """まわりの材料：いまの節気のページと、Jev が選んだ記事。"""
    news = ss.news_items(await _fetch(agent, NEWS_LIST_URL, raw=True))
    columns = ss.column_links(await _fetch(agent, COLUMN_LIST_URL))
    picked_news, picked_columns = await jev_judges.judge_season_reads(
        getattr(agent, "_jev", None),
        today=today,
        place=place,
        news=news,
        columns=[c for name in ss.PICK_SECTIONS for c in columns.get(name, [])],
    )
    term = ss.solar_term_page(columns, se.last_term(today))
    urls = ([term] if term else []) + [i.url for i in (*picked_news, *picked_columns)]
    pages = [await _fetch(agent, u) for u in urls]
    return [p for p in pages if p]


def _parse(text: str) -> "dict | None":
    return read_json_merged(text)  # 2 つに分けた・書き直した返りも読む（記-p）


def _rows(data: dict) -> "dict[str, tuple[se.Row, ...]] | None":
    """返りを欄ごとの行へ。形が違えば None。天気は機械が書くので、LLM の天気は使わない。"""
    out: dict[str, tuple[se.Row, ...]] = {}
    for name, items in data.items():
        if name == "天気":
            continue
        if not isinstance(items, list):
            return None
        rows = []
        for it in items:
            if not isinstance(it, dict):
                return None
            source = str(it.get("source", "")).strip() if name == "家の話題" else ""
            rows.append(se.Row(str(it.get("text", "")).strip(), source))
        if rows:
            out[str(name)] = tuple(rows)
    return out


async def update_season(agent, records: list, *, today: "date | None" = None) -> Outcome:
    """季節の層の 1 段：確かめる → 読む → 頼む → 検査 → 通れば保存する。"""
    today = today or date.today()
    old = se.stored()
    if old is not None and old.written_on == today:
        measure.record("季節", 書いた="-", 見送り="今日は書いた")
        return Outcome(False, "今日はもう書いた")
    me_md = str(getattr(agent, "_me_md", "") or "")
    place = place_of(me_md)
    weather = await _weather(agent, me_md)
    reads = await _reads(agent, today, place)
    if not weather and not reads:
        return _skip("ページを読めなかった")
    share = READ_CHARS // len(reads) if reads else 0
    prompt = _PROMPT.format(
        rows=se.MAX_ROWS,
        chars=se.MAX_CHARS,
        place=place or "（分からない）",
        reads="\n\n".join(f"---\n{p[:share]}" for p in reads) or "（無し）",
        materials="\n".join(f"- {r.obs_id} [{r.direction}] {r.text}" for r in records)
        or "（無し）",
    )
    try:
        raw = str(await agent.backend.complete(prompt, max_tokens=1000) or "")
    except Exception as e:  # noqa: BLE001
        return _skip(f"依頼に失敗：{e}")
    data = _parse(raw)
    rows = _rows(data) if data is not None else None
    if rows is None:
        return _skip("返りを読めなかった")
    if weather:
        rows = {"天気": weather, **rows}
    env = se.SeasonEnv(written_on=today, rows=rows)
    reason = se.check(env, material_ids={r.obs_id for r in records})
    if reason:
        return _skip(reason)
    se.store(env)
    content = "季節とまわりを書いた：" + "／".join(
        f"{k}：{r.text}" for k in se.FIELDS for r in rows.get(k, ())
    )
    await agent._memory.save_async_with_id(
        content[:500],
        direction="内省",
        kind="observation",
        materialize_now=True,
        **agent._observation_perspective(),
    )
    logger.info("rest 季節の層：書いた（天気 %d 行・読み物 %d 本）", len(weather), len(reads))
    measure.record("季節", 書いた=sum(len(v) for v in rows.values()), 見送り="-")
    return Outcome(True)


def _skip(reason: str) -> Outcome:
    logger.info("rest 季節の層：書かなかった（%s）", reason)
    measure.record("季節", 書いた=0, 見送り=reason)
    return Outcome(False, reason)
