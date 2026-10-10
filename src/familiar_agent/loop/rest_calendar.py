"""REST 内省・層 1 ③「暦のまとめ」（記-m 段 3・2026-10-01）。月と年の要約を書いて、記憶の木をつくる。

日次の畳み込み（①）が書く日ごとの要約（`day_summary`）と人ごとのまとめ（`person_summary`）の上に、
月と年の要約を重ねて「人 → 年 → 月 → 日」の木にする。主LLM は想起の道具 `recall_tree` で節を指して読む。

- **何を書くか**（機械が決める）：家族全体の月は日ごとの要約がある月、人ごとの月はその人の人ごとのまとめが
  ある月。**いまの月とまだ書いていない月は除き**、古い順に並べる。年はいまの年より前で、その年の月の要約が
  そろっている年。同じ時期なら家族全体を人ごとより先に。
- **1 晩に 6 本まで**（本人の決定）。1 か月ぶんは家族全体 1 本＋人ごと。過去が溜まっていれば数晩かけて埋める。
- **書き手はフルLLM**（本人の決定・日次の畳み込みと同じ担い手）。材料は 1 つ下の段の要約（月なら日ごと、
  年なら月）で、`{"summary": …}` だけを返させる。空・読めない・字数の超過は書かず、次の晩に持ち越す。
- **置き方**：向き「月のまとめ／年のまとめ／人の月のまとめ／人の年のまとめ」、時刻はその月・年の終わり
  （UTC 23:59・日ごとの要約がその日の 23:59 に置かれるのと同じ）、人ごとのものはその人の面。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..core.structured_ask import read_json_merged
from ..core import measure, parsing
from ..core.memory_tree import parse_node
from ..io.oif import MI
from .rest_fold import EPISODE_MAX_CHARS, PERSON_MAX_CHARS, Written

logger = logging.getLogger(__name__)

#: 1 晩に書く本数の上限（本人の決定 2026-10-01）。最初の晩の「残り」を見て見直す。
CALENDAR_MAX_PER_NIGHT = 6

#: (段, 人か) → 向き。向きから種類が決まる（`io/oif._KIND_OF_DIRECTION`）。
_DIRECTION = {
    ("月", False): "月のまとめ",
    ("年", False): "年のまとめ",
    ("月", True): "人の月のまとめ",
    ("年", True): "人の年のまとめ",
}
#: 材料になる種類（月の材料は日ごと、年の材料は月）と、書いた要約の種類。
_SOURCE = {False: "day_summary", True: "person_summary"}
_MONTH = {False: "month_summary", True: "person_month_summary"}
_YEAR = {False: "year_summary", True: "person_year_summary"}

_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、{period}を振り返って{whose}のまとめを書く。
下に、その{unit}ごとの{source}を時間順に並べる。

一人称（ぼく）で、その{span}に何があったか・誰と・何が続いていたか・何が変わったかを、{limit} 字以内で書く。
並べた{source}に無いことは書かない。日付を足すなら、並べたものに書いてある日だけを使う。

出力は次の JSON だけ（ほかには何も書かない）：
{{"summary": "…"}}

[{period}の{source}]
{records}
"""


@dataclass(frozen=True)
class Task:
    """暦のまとめ 1 本。`period` は節（YYYY-MM か YYYY）。"""

    level: str  # 月 / 年
    period: str
    person_name: "str | None" = None
    person_id: "str | None" = None


@dataclass
class CalendarResult:
    written: int = 0
    skipped: int = 0  # 検査に通らず書かなかった本数（次の晩に持ち越す）
    left: int = 0  # 上限で次の晩へ回した本数
    records: "tuple[Written, ...]" = field(default_factory=tuple)


def _people(agent) -> "list[tuple[str, str]]":
    """家族ひとりずつ（名前, 人物 id）。人物表に無い名前は除く。"""
    out: list[tuple[str, str]] = []
    for m in parsing.parse_family_md(str(getattr(agent, "_family_md", "") or "")):
        name = str(m.get("name") or "").strip()
        pid = agent._pmm.find_person_id_by_name(name) if name else None
        if pid and isinstance(pid, str):
            out.append((name, pid))
    return out


def _months(agent, kind: str, pid: "str | None") -> "set[str]":
    return set(agent._oif.tree_months(kind, person_id=pid) or [])


def pending(agent, *, now: datetime) -> "list[Task]":
    """まだ書いていない暦のまとめを、古い順に並べる（家族全体が人ごとより先）。"""
    this_month = now.astimezone(timezone.utc).strftime("%Y-%m")
    this_year = this_month[:4]
    trees: list[tuple[str | None, str | None]] = [(None, None), *_people(agent)]
    tasks: list[tuple[tuple, Task]] = []
    for order, (name, pid) in enumerate(trees):
        person = pid is not None
        sources = {m for m in _months(agent, _SOURCE[person], pid) if m < this_month}
        done_months = _months(agent, _MONTH[person], pid)
        done_years = {m[:4] for m in _months(agent, _YEAR[person], pid)}
        for m in sorted(sources - done_months):
            tasks.append(((m, 0, order), Task("月", m, name, pid)))
        for y in sorted({m[:4] for m in sources}):
            year_months = {m for m in sources if m.startswith(y + "-")}
            if y < this_year and y not in done_years and year_months <= done_months:
                tasks.append(((y + "-13", 1, order), Task("年", y, name, pid)))
    return [t for _, t in sorted(tasks, key=lambda kt: kt[0])]


def _period_end(period: str) -> datetime:
    """その月・年の終わり（UTC 23:59）。"""
    got = parse_node(period)
    assert got is not None
    return got[2] - timedelta(minutes=1)


def _prompt(task: Task, rows: "list[dict]") -> str:
    month = task.level == "月"
    records = "\n".join(
        f"- {r['timestamp']:%Y-%m-%d} {r.get('content', '')}"
        if isinstance(r.get("timestamp"), datetime)
        else f"- {r.get('content', '')}"
        for r in rows
    )
    return _PROMPT.format(
        period=f"{task.period[:4]} 年 {int(task.period[5:])} 月" if month else f"{task.period} 年",
        whose=f"{task.person_name}との" if task.person_name else "家族の",
        unit="日" if month else "月",
        source="日ごとの記録" if month else "月のまとめ",
        span="月" if month else "年",
        limit=PERSON_MAX_CHARS if task.person_id else EPISODE_MAX_CHARS,
        records=records or "（なし）",
    )


def _parse(raw: str, limit: int) -> "str | None":
    data = read_json_merged(raw)  # 2 つに分けた・書き直した返りも読む（記-p）
    text = str(data.get("summary", "") if data else "").strip()
    if not text or len(text) > limit:
        return None
    return text


async def _write_one(agent, task: Task) -> "str | None":
    """1 本書く。書けたら記録の id、検査に通らなければ None。"""
    rows = agent._oif.tree_children(task.period, person_id=task.person_id) or []
    limit = PERSON_MAX_CHARS if task.person_id else EPISODE_MAX_CHARS
    try:
        raw = await agent.backend.complete(_prompt(task, rows), max_tokens=1200)
    except Exception as e:  # noqa: BLE001
        logger.warning("rest 暦のまとめの依頼に失敗（%s・次の晩に持ち越す）: %s", task.period, e)
        return None
    text = _parse(str(raw or ""), limit)
    if text is None:
        logger.warning(
            "rest 暦のまとめを見送った（%s・%s）", task.period, task.person_name or "家族"
        )
        return None
    mi = MI(
        id="",
        content=text,
        timestamp=_period_end(task.period),
        direction=_DIRECTION[(task.level, task.person_id is not None)],
    )
    kw = dict(agent._observation_perspective())
    if task.person_id:
        kw["participants"] = [task.person_id]  # その人の面に立てる（人ごとのまとめと同じ）
    return str(await agent._oif.write(mi, now=True, **kw) or "")


async def write_calendar(agent, *, now: "datetime | None" = None) -> CalendarResult:
    """暦のまとめを 1 晩ぶん書く（層 1 の ③）。上限を超えたぶんは次の晩へ。"""
    now = now or datetime.now(timezone.utc)
    tasks = pending(agent, now=now)
    todo, rest = tasks[:CALENDAR_MAX_PER_NIGHT], tasks[CALENDAR_MAX_PER_NIGHT:]
    result = CalendarResult(left=len(rest))
    records: list[Written] = []
    for task in todo:
        obs_id = await _write_one(agent, task)
        if obs_id is None:
            result.skipped += 1
            continue
        result.written += 1
        kind = "month_summary" if task.level == "月" else "year_summary"
        records.append(Written(obs_id, kind, f"{task.period}：{task.person_name or '家族'}"))
    result.records = tuple(records)
    measure.record("層1暦", 書いた=result.written, 見送り=result.skipped, 残り=result.left)
    logger.info(
        "rest 暦のまとめ：%d 本書いた（残り %d・見送り %d）",
        result.written,
        result.left,
        result.skipped,
    )
    return result
