"""季節の層が毎晩読むページを、機械が読む（知-ac・2026-10-24・本人の決定・`設計方針_季節の層` v0.4）。

- 天気：Yahoo!天気の住まいの市のページ（URL は `ME.md` の「天気のページ」）。表を機械が読み、**明日と明後日**の
  2 行を日付つきで書く。REST は夜に回るので今日の表はほぼ過ぎている。季節の層は 7 日そのまま渡るので、
  「明日」と書くと 3 日目には嘘になる。表の形が違えば行は作らない（天気の欄は空）。
- まわり：ウェザーニュースのニュース一覧（元の文）と、Yahoo!のお天気コラムの一覧から題名と URL を取る。
  どれを読むかは Jev が題名で選ぶ（`jev_judges.judge_season_reads`）。節気のページだけは名前で機械が決める。

外へは出ない（fetch は呼び手が持つ）。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from . import season_env as se

NEWS_BASE = "https://weathernews.jp"
#: Yahoo!のお天気コラムの節。二十四節気は機械が決め、ほかの 2 つから Jev が選ぶ。
SOLAR_SECTION = "二十四節気"
PICK_SECTIONS = ("歳時記", "天気の知識")
_PR = ("＜PR＞", "<PR>", "【PR】")


@dataclass(frozen=True)
class Link:
    title: str
    url: str
    date: str = ""


# ── 天気 ───────────────────────────────────────────────────────────────────

_TOMORROW = re.compile(r"^###\s*明日の天気\s*-\s*(\d+)月(\d+)日\((.)\)", re.M)
_WEEK = re.compile(r"^##\s*週間天気", re.M)
_DAY = re.compile(r"(\d+)月(\d+)日\s*\((.)\)")


def _table(text: str, start: int) -> "dict[str, list[str]]":
    """`start` の後の最初の表を、行の見出し → 欄の並びにする。"""
    rows: dict[str, list[str]] = {}
    for line in text[start:].splitlines()[1:]:
        s = line.strip()
        if not s.startswith("|"):
            if rows:
                break
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells[0]:
            rows[cells[0]] = cells[1:]
    return rows


def _first(cell: str) -> str:
    """「雨  雨」（絵の代わりの文字と本文）の 1 つ目。"""
    words = cell.split()
    return words[0] if words else ""


def _num(cell: str) -> str:
    return f"{float(cell):g}"


def _sky(seq: "list[str]") -> str:
    kept = [s for i, s in enumerate(seq) if s and (i == 0 or s != seq[i - 1])]
    if len(kept) > 3:
        kept = [kept[0], kept[1], "…", kept[-1]]
    return "→".join(kept)


def _tomorrow(text: str) -> "str | None":
    m = _TOMORROW.search(text)
    if not m:
        return None
    t = _table(text, m.end())
    try:
        sky = _sky([_first(c) for c in t["天気"]])
        temps = [float(c) for c in t["気温（℃）"]]
        wet = [h.rstrip("時") for h, mm in zip(t["時刻"], t["降水量（mm）"]) if float(mm or 0) > 0]
    except (KeyError, ValueError):
        return None
    if not sky or not temps:
        return None
    if not wet:
        rain = "雨なし"
    elif len(wet) == 1:
        rain = f"雨は{wet[0]}時ごろ"
    else:
        rain = f"雨は{wet[0]}〜{wet[-1]}時"
    head = f"{m.group(1)}/{m.group(2)}（{m.group(3)}）：{sky}、{min(temps):g}〜{max(temps):g}℃"
    row = f"{head}、{rain}"
    return row if len(row) <= se.MAX_CHARS else (head if len(head) <= se.MAX_CHARS else None)


def _day_after(text: str) -> "str | None":
    m = _WEEK.search(text)
    if not m:
        return None
    t = _table(text, m.end())
    rain_key = next((k for k in t if k.startswith("降水")), "")
    try:
        day = _DAY.search(t["日付"][0])
        sky = _first(t["天気"][0])
        hi, lo = t["気温（℃）"][0].split()[:2]
        rain = t[rain_key][0] if rain_key else ""
        hi, lo = _num(hi), _num(lo)
    except (KeyError, IndexError, ValueError):
        return None
    if day is None or not sky:
        return None
    row = f"{day.group(1)}/{day.group(2)}（{day.group(3)}）：{sky}、{lo}〜{hi}℃"
    if rain:
        row += f"、雨{rain}%"
    return row if len(row) <= se.MAX_CHARS else None


def weather_rows(page: str) -> "tuple[se.Row, ...]":
    """守谷市のページ（fetch の返り）から、明日と明後日の天気の行。読めなければ空。"""
    return tuple(se.Row(r) for r in (_tomorrow(page), _day_after(page)) if r)


# ── ウェザーニュースの一覧 ───────────────────────────────────────────────────

_NEWS = re.compile(r'<a href="(/news/\d{6}/\d+/)"[^>]*>((?:(?!</a>).)*?)</a>', re.S)
_TITLE = re.compile(r'<p data-text="title"[^>]*>(.*?)</p>', re.S)
_DATE = re.compile(r'<p data-text="date"[^>]*>(.*?)</p>', re.S)


def news_items(raw_html: str) -> "list[Link]":
    """一覧の元の文から、記事の題名・URL・日時。PR と重なりは除く。"""
    out: list[Link] = []
    seen: set[str] = set()
    for m in _NEWS.finditer(raw_html or ""):
        title = _TITLE.search(m.group(2))
        if title is None or m.group(1) in seen:
            continue
        name = html.unescape(title.group(1)).strip()
        if not name or name.startswith(_PR):
            continue
        seen.add(m.group(1))
        when = _DATE.search(m.group(2))
        out.append(Link(name, NEWS_BASE + m.group(1), when.group(1).strip() if when else ""))
    return out


# ── Yahoo!のお天気コラムの一覧 ────────────────────────────────────────────────

_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_LINK = re.compile(r"^\*\s*\[(?:!\[[^\]]*\]\([^)]*\))?([^\]]+)\]\((https?://[^)\s]+)\)")


def column_links(page: str) -> "dict[str, list[Link]]":
    """一覧（fetch の返り）の節ごとの題名と URL。"""
    out: dict[str, list[Link]] = {}
    section = ""
    for line in (page or "").splitlines():
        s = _SECTION.match(line)
        if s:
            section = s.group(1)
            continue
        link = _LINK.match(line.strip())
        if link and section:
            out.setdefault(section, []).append(Link(link.group(1).strip(), link.group(2)))
    return out


def solar_term_page(links: "dict[str, list[Link]]", name: str) -> "str | None":
    """節気の名前（リンクの文字）でページを決める。URL のローマ字は不揃いなので使わない。"""
    if name not in {n for _, _, n in se.SOLAR_TERMS}:
        return None
    return next((i.url for i in links.get(SOLAR_SECTION, ()) if i.title == name), None)
