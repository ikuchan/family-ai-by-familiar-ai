"""REST 内省・季節の層（知-ac・2026-09-26・`設計方針_季節の層` v0.1）。

層 1（出来事を畳む）の後・層 2（自己像）の前に回り、**いまの季節と家のまわり**を書き直す。

1. 今日もう書いていれば何もしない（1 日 1 回まで）。
2. `ME.md` の住所（都道府県と市）で 1 回検索する。Brave が使えなければ Tavily（調べものと同じ）。
   住所が無ければ検索しない（天気とまわりは空）。
3. フル LLM に検索結果とその晩の出来事を渡し、天気・まわり・家の話題を書かせる（暦は機械が計算する）。
4. 機械が検査する（`core.season_env.check`）。通れば保存し `内省` の記録と計測ログに残す。
   **通らなければ前の値を残す**。

まとめ方は LLM、通すかどうかは機械（層 1・層 2 と同じ分担）。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from ..core import measure
from ..core import season_env as se

logger = logging.getLogger(__name__)

#: 検索の道具（調べものと同じ 2 つ）。先頭が使えなければ次。
SEARCH_TOOLS = ("brave_web_search", "tavily_search")
#: プロンプトへ渡す検索結果の上限。出典の照合は全文で行う。
SEARCH_CHARS = 4000

_PLACE = re.compile(r"(\S{2,3}?[都道府県])(\S+?[市区町村])")

_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、一日の終わりに、いまの季節と家のまわりの様子を書き留める。
下に、住んでいる土地の検索結果と、今夜まとめた出来事（各行の先頭が id）を示す。

次の 3 つの欄を書く。各欄 0〜{rows} 行、1 行 {chars} 字まで。書けることが無ければ空の配列にする。
- 天気：この数日の天気と気温。検索結果にあることだけ。
- まわり：土地の季節の様子（花・虫・行事など）。検索結果にあることだけ。
- 家の話題：家族が話していた季節や行事の話。今夜の出来事にあることだけ。

天気とまわりの行には、根拠にした検索結果の文を**一字も変えずに** quote に写す（{chars} 字まで）。
家の話題の行には、根拠にした出来事の id を source に書く。暦（節気）は書かない。

[住んでいる土地]
{place}

[検索結果]
{search}

[今夜まとめた出来事]
{materials}

JSON だけを返す：
{{"天気": [{{"text": "…", "quote": "…"}}], "まわり": [{{"text": "…", "quote": "…"}}], "家の話題": [{{"text": "…", "source": "id"}}]}}
"""


@dataclass(frozen=True)
class Outcome:
    applied: bool
    reason: "str | None" = None


def place_of(me_md: str) -> str:
    """`ME.md` から都道府県と市（例：「茨城県守谷市」）。見つからなければ空。"""
    m = _PLACE.search(me_md or "")
    return m.group(1) + m.group(2) if m else ""


async def _search(agent, query: str) -> str:
    for tool in SEARCH_TOOLS:
        try:
            r = await agent._mcp_search(tool, {"query": query})
        except Exception as e:  # noqa: BLE001
            logger.warning("rest 季節の層：%s が失敗した: %s", tool, e)
            continue
        if r.ok:
            return str(r.text or "")
        logger.warning("rest 季節の層：%s が使えなかった（%.120s）", tool, r.text)
    return ""


def _parse(text: str) -> "dict | None":
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _rows(data: dict) -> "dict[str, tuple[se.Row, ...]] | None":
    """返りを欄ごとの行へ。形が違えば None。"""
    out: dict[str, tuple[se.Row, ...]] = {}
    for name, items in data.items():
        if not isinstance(items, list):
            return None
        key = "source" if name == "家の話題" else "quote"
        rows = []
        for it in items:
            if not isinstance(it, dict):
                return None
            rows.append(se.Row(str(it.get("text", "")).strip(), str(it.get(key, "")).strip()))
        if rows:
            out[str(name)] = tuple(rows)
    return out


async def update_season(agent, records: list, *, today: "date | None" = None) -> Outcome:
    """季節の層の 1 段：確かめる → 検索する → 頼む → 検査 → 通れば保存する。"""
    today = today or date.today()
    old = se.stored()
    if old is not None and old.written_on == today:
        measure.record("季節", 書いた="-", 見送り="今日は書いた")
        return Outcome(False, "今日はもう書いた")
    place = place_of(str(getattr(agent, "_me_md", "") or ""))
    search = await _search(agent, f"{place} 今週の天気 季節の話題") if place else ""
    prompt = _PROMPT.format(
        rows=se.MAX_ROWS,
        chars=se.MAX_CHARS,
        place=place or "（分からない）",
        search=search[:SEARCH_CHARS] or "（無し）",
        materials="\n".join(f"- {r.obs_id} [{r.kind}] {r.text}" for r in records) or "（無し）",
    )
    try:
        raw = str(await agent.backend.complete(prompt, max_tokens=1000) or "")
    except Exception as e:  # noqa: BLE001
        return _skip(f"依頼に失敗：{e}")
    data = _parse(raw)
    rows = _rows(data) if data is not None else None
    if rows is None:
        return _skip("返りを読めなかった")
    env = se.SeasonEnv(written_on=today, rows=rows)
    reason = se.check(env, search_text=search, material_ids={r.obs_id for r in records})
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
    logger.info("rest 季節の層：書いた（%s）", place or "住所なし")
    measure.record("季節", 書いた=sum(len(v) for v in rows.values()), 見送り="-")
    return Outcome(True)


def _skip(reason: str) -> Outcome:
    logger.info("rest 季節の層：書かなかった（%s）", reason)
    measure.record("季節", 書いた=0, 見送り=reason)
    return Outcome(False, reason)
