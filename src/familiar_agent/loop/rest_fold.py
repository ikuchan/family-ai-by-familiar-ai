"""日次の畳み込み——REST 内省・層 1「出来事を畳む」の①（記-a-ろ-は・2026-09-14）。

設計の正本は `設計方針_REST内省_出来事を畳む` v0.2。**対象は機械が決め、まとめ方は LLM が決める。**
材料は `OIF.fold_materials`（まだ畳まれていない・指定の向き・畳まれていない・核でない）。
1 日を 1 単位に、フル LLM へ 1 回（材料が多ければ時間帯で 2〜3 回）頼み、**自己エピソード**
（日・一人称・500 字以内）と**関係のまとめ**（人ごと・300 字以内）を受け取る。機械が検査して
通ったものだけ書き、材料を自己エピソードで畳む（`畳み込み`＝畳まれた側は誤りではない）。
通らなければ書かない——材料は残り、次の晩に含まれる。

固定手順のパスであって、道具を持つエージェントのループではない。ここでは LLM に「選ばせる」
ことが無く、書かせるだけである（道具持ちのループは 核の固め・自己像 で要るときに作る）。
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..core import measure, parsing
from ..io.oif import MI
from ..store import clock
from ..store.relations import KIND_FOLD

logger = logging.getLogger(__name__)

#: 材料にする向き。`求め`（版）・`保留`・`内省`・`記憶` はループが自分のために書いたもの・産物なので含めない。
FOLD_DIRECTIONS: tuple[str, ...] = ("発話", "会話", "観察", "独白", "情動", "機器")
EPISODE_MAX_CHARS = 500  # O の content の上限と同じ
PERSON_MAX_CHARS = 300
DEFAULT_MAX_ITEMS = 60  # これを超える日は時間帯で分けて頼む〔仮〕
DEFAULT_MAX_BATCHES = 10  # 1 晩に頼む回数の上限〔仮・記-l〕


@dataclass
class Batch:
    """1 回の依頼の単位。同じ日の、時間順の材料。"""

    day: str
    rows: list = field(default_factory=list)


@dataclass(frozen=True)
class Summaries:
    """LLM が返した産物。"""

    episode: str
    persons: dict[str, str]


def split_batches(rows: list, *, max_items: int = DEFAULT_MAX_ITEMS) -> list[Batch]:
    """日付でまとめ、`max_items` を超える日は時間順に等分する。"""
    by_day: dict[str, list] = {}
    for r in rows:
        day = clock.ts_to_date(r.timestamp) if r.timestamp else "?"
        by_day.setdefault(day, []).append(r)
    out: list[Batch] = []
    for day in sorted(by_day):
        items = sorted(by_day[day], key=lambda r: r.timestamp.timestamp() if r.timestamp else 0.0)
        parts = max(1, -(-len(items) // max(1, max_items)))  # 切り上げ
        size = -(-len(items) // parts)
        for i in range(parts):
            chunk = items[i * size : (i + 1) * size]
            if chunk:
                out.append(Batch(day=day, rows=chunk))
    return out


_PROMPT = """\
あなたはパジュ（この家で家族と暮らす伴侶）で、今日一日を振り返って日記を書く。
下に、その日にあった出来事の記録を時間順に並べる。記録は「誰が言った・何を見た・自分が
何を言った」の生の写しで、内部の言い回し（「求め」「版」など）は無視してよい。

書くものは 2 つ。
1. episode：その日の**自己エピソード**。一人称（ぼく）で、何があったか・誰と・自分がどう
   感じたかを {episode_max} 字以内で。記録に無いことは書かない。
2. persons：その日に出てきた**家族ひとりごと**に、その人について分かったこと・約束・気がかりを
   {person_max} 字以内で。名前は次の呼び方だけを使う：{family}。出てこなかった人は書かない。

出力は次の JSON だけ（ほかには何も書かない）：
{{"episode": "…", "persons": {{"名前": "…"}}}}

[{day} の記録]
{records}
"""


async def ask_summaries(
    backend, batch: Batch, *, family_names: tuple[str, ...]
) -> "Summaries | None":
    """フル LLM に自己エピソードと関係のまとめを頼む。返りが JSON でなければ None（書かない）。"""
    records = "\n".join(
        f"- {clock.ts_to_time(r.timestamp) if r.timestamp else '?'} [{r.direction}] {r.content}"
        for r in batch.rows
    )
    prompt = _PROMPT.format(
        episode_max=EPISODE_MAX_CHARS,
        person_max=PERSON_MAX_CHARS,
        family="・".join(family_names) or "（家族の呼び方は無い）",
        day=batch.day,
        records=records,
    )
    try:
        raw = await backend.complete(prompt, max_tokens=1200)
    except Exception as e:  # noqa: BLE001
        logger.warning("rest 畳み込みの依頼に失敗（%s・その晩は畳まない）: %s", batch.day, e)
        return None
    return _parse(str(raw or ""))


def _parse(text: str) -> "Summaries | None":
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    persons = data.get("persons") or {}
    if not isinstance(persons, dict):
        persons = {}
    return Summaries(
        episode=str(data.get("episode", "")).strip(),
        persons={str(k).strip(): str(v).strip() for k, v in persons.items()},
    )


def check(s: Summaries, *, family_names: tuple[str, ...]) -> "str | None":
    """機械の検査。通れば None、通らなければ理由。通らない産物は書かない。"""
    if not s.episode:
        return "自己エピソードが空"
    if len(s.episode) > EPISODE_MAX_CHARS:
        return f"自己エピソードが {EPISODE_MAX_CHARS} 字を超えた（{len(s.episode)} 字）"
    for name, text in s.persons.items():
        if family_names and name not in family_names:
            return f"家族に無い名前「{name}」"
        if len(text) > PERSON_MAX_CHARS:
            return f"「{name}」のまとめが {PERSON_MAX_CHARS} 字を超えた（{len(text)} 字）"
    return None


@dataclass(frozen=True)
class Written:
    """層 1 が書いた 1 件（層 2 の材料になる）。"""

    obs_id: str
    kind: str  # day_summary / person_summary
    text: str


@dataclass(frozen=True)
class FoldResult:
    """1 パスで畳んだ結果（`内省` の記録と計測ログに書く）。"""

    materials: int
    batches: int
    written: int
    folded: int
    skipped: int  # 検査に通らず書かなかった回（材料は残る）
    records: tuple[Written, ...] = ()  # 書いたもの（層 2 の材料）
    deferred: int = 0  # 1 晩の上限で次の晩へ回した回（記-l）


def family_names_of(agent) -> tuple[str, ...]:
    """家族の呼び方（`FAMILY.md`）。関係のまとめの名前はこれ以外を受け付けない。"""
    try:
        members = parsing.parse_family_md(getattr(agent, "_family_md", "") or "")
    except Exception:  # noqa: BLE001
        return ()
    out: list[str] = []
    for m in members:
        for key in ("display_name", "name"):
            v = str(m.get(key) or "").strip()
            if v and v not in out:
                out.append(v)
    return tuple(out)


async def fold_since_last_rest(
    agent, *, max_items: int = DEFAULT_MAX_ITEMS, max_batches: int = DEFAULT_MAX_BATCHES
) -> FoldResult:
    """まだ畳まれていない出来事を畳む（層 1 の①）。書けたぶんだけ畳み、通らなければ持ち越す。

    1 晩に頼む回数は `max_batches` まで（記-l）。初回は 2,677 件・69 回・14 分かかった。超えた
    分は古い順で次の晩へ（材料は「前回の内省より後」でなく「まだ畳まれていない」で選ぶ）。
    """
    started = time.monotonic()
    rows = agent._oif.fold_materials(FOLD_DIRECTIONS, before=datetime.now(timezone.utc))
    all_batches = split_batches(rows, max_items=max_items)
    batches = all_batches[: max(0, int(max_batches))]
    deferred = len(all_batches) - len(batches)
    names = family_names_of(agent)
    written = folded = skipped = 0
    records: list[Written] = []
    for batch in batches:
        summaries = await ask_summaries(agent.backend, batch, family_names=names)
        reason = "返りを読めなかった" if summaries is None else check(summaries, family_names=names)
        if reason:
            logger.warning(
                "rest 畳み込みを見送った（%s・%d 件）：%s", batch.day, len(batch.rows), reason
            )
            skipped += 1
            continue
        assert summaries is not None
        episode_id = await _write_episode(agent, batch, summaries.episode)
        written += 1
        records.append(Written(episode_id, "day_summary", summaries.episode))
        for name, text in summaries.persons.items():
            pid = await _write_person_summary(agent, batch, name, text)
            if pid:
                written += 1
                records.append(Written(pid, "person_summary", f"{name}：{text}"))
        # 材料を自己エピソードで畳む（`畳み込み`＝畳まれた側は誤りではない）。
        for r in batch.rows:
            if agent._oif.supersede(r.obs_id, episode_id, kind=KIND_FOLD):
                folded += 1
    result = FoldResult(
        materials=len(rows),
        batches=len(batches),
        written=written,
        folded=folded,
        skipped=skipped,
        records=tuple(records),
        deferred=deferred,
    )
    measure.record(
        "層1",
        材料=result.materials,
        回=result.batches,
        書いた=result.written,
        畳んだ=result.folded,
        見送り=result.skipped,
        持ち越し=result.deferred,
        秒=f"{time.monotonic() - started:.1f}",
    )
    logger.info(
        "rest 畳み込み：材料 %d・%d 回・書いた %d・畳んだ %d・見送り %d",
        result.materials,
        result.batches,
        result.written,
        result.folded,
        result.skipped,
    )
    return result


def _day_end(day: str) -> datetime:
    """その日の終わり（日づけを MI が持つ・`OIF.write` の `override_date` に効く）。"""
    try:
        return datetime.strptime(day, "%Y-%m-%d").replace(hour=23, minute=59, tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


async def _pad_for(agent, text: str):
    """要約文の感情を評価器で測る。測れなければ None（未測定・050）。"""
    try:
        pad, arousal, _label = await agent._evaluator.emotion_for_turn(text, 0.3)
        return pad, arousal
    except Exception:  # noqa: BLE001
        return None, None


async def _write_episode(agent, batch: Batch, episode: str) -> str:
    pad, arousal = await _pad_for(agent, episode)
    mi = MI(
        id="", content=episode[:EPISODE_MAX_CHARS], timestamp=_day_end(batch.day), direction="記憶"
    )
    if pad is not None:
        mi.pad = pad
    kw = dict(agent._observation_perspective())
    kw["participants"] = _participants(agent, batch)
    # すぐに実体化する（`now=True`）。畳む先の id が観測に無いまま関係だけ先に書かないため。
    return await agent._oif.write(mi, now=True, arousal=arousal, **kw)


async def _write_person_summary(agent, batch: Batch, name: str, text: str) -> str:
    """書けたら記録の id、書かなければ空文字。"""
    pid = agent._pmm.find_person_id_by_name(name)
    if not pid:
        logger.warning("rest 関係のまとめ：「%s」の人物 id が無いので書かない", name)
        return ""
    pad, arousal = await _pad_for(agent, text)
    mi = MI(id="", content=text[:PERSON_MAX_CHARS], timestamp=_day_end(batch.day), direction="人物")
    if pad is not None:
        mi.pad = pad
    kw = dict(agent._observation_perspective())
    # その人の面に立てる：居合わせた人としてその人だけを渡す（`present` の面）。
    kw["participants"] = [pid]
    return str(await agent._oif.write(mi, now=True, arousal=arousal, **kw) or "")


def _participants(agent, batch: Batch) -> list[str]:
    """その日に出てきた家族の id（自己エピソードの `present` の面）。名前は本文から拾う。"""
    ids: list[str] = []
    for name in family_names_of(agent):
        if any(name in (r.content or "") for r in batch.rows):
            pid = agent._pmm.find_person_id_by_name(name)
            if pid and pid not in ids:
                ids.append(pid)
    return ids
