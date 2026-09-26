"""REST 内省・層 1 ②「核の固め」（記-a-ろ-に・2026-09-15・`設計方針_REST内省_出来事を畳む` §3c）。

核（根づき n ≥ 1）が I* を超えたぶんを、意味の近い束ごとに**まとめ知識**（週・月・人ごとの
「いつもの」）へ固め、出典を畳む。**対象と束は機械が決め、LLM は 1 束 1 文を書くだけ。**

1. 同一：面ベクトルのコサイン ≥ τ同 でつながる群を、個数上限なし・LLM なしで 1 件に。代表は
   いちばん新しい 1 件、他は代表で supersede（畳み込み）、代表の面の n は群の最大。
2. 測り直し：同一のあとの核で I を測り、I ≤ I* なら終わり。
3. 束ね：`core/bundling.bundle`（τ の覆いで自然な群 → 容量 k=⌈N/K_τ⌉ で同じ大きさに）。
4. 選ぶ：平均 u の低い束から、Σb·u の累計が超過 E=I−I* に達するまで（1 晩の上限あり）。
5. 書く：1 束 1 依頼。JSON {text, sources, people}。検査（出典が束の中・字数・家族の名前）に落ちた
   束はその晩は固めない。
6. 産物：向き `まとめ`（kind core_summary）・時刻は出典の最新・出典の面の人が居合わせた人・
   面の n は出典の最大。各出典を産物で supersede。
7. 記録：計測ログに `層1同一` と `層1固め`。`内省` の記録には一文。

材料は `OIF.core_records()`（`内省`・`保留` 以外の全向き）。τ同・τ・m・上限は層 3 の設定値。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..core import bundling, measure
from ..core import info_measure as im
from ..io.oif import MI
from ..store.relations import KIND_FOLD
from .rest_fold import Written, family_names_of

logger = logging.getLogger(__name__)

SUMMARY_MAX_CHARS = 300  # 1 束 1 文の上限〔仮・①の関係のまとめと同じ〕
K_MAX = 12  # 1 束の容量の上限（300 字に写せる出典の数の目安・25 字/件）〔仮〕

_PROMPT = """\
あなた自身の記憶を、ひとつの「いつものこと」に固める。出力は JSON だけ。前置きも説明も書かない。

下の記録は、意味の近いものを機械が束ねたものである。これらに共通していること・繰り返し起きて
いることを、**一人称で 1 文〜数文（{max_chars} 字以内）**にまとめる。

- 記録に書かれていないことは書かない。日付の羅列や箇条書きにしない。
- 人の名前は家族の呼び方（{family}）だけを使う。それ以外の名前は書かない。
- `sources` には、まとめに使った記録の id を**全部**書く（使わなかった id は書かない）。
- `people` には、まとめに出てくる家族の呼び方を書く（居なければ空）。

形式：{{"text": "…", "sources": ["id", …], "people": ["呼び方", …]}}

[記録]
{records}
"""


@dataclass(frozen=True)
class CoreResult:
    identical_groups: int = 0
    identical_folded: int = 0
    total_bits: float = 0.0
    excess_bits: float = 0.0
    bundles: int = 0  # 固めようとした束
    written: int = 0
    folded: int = 0
    skipped: int = 0
    records: tuple[Written, ...] = ()


def check(
    text: str,
    sources: list[str],
    people: list[str],
    *,
    bundle_ids: set[str],
    family: tuple[str, ...],
) -> "str | None":
    """通れば None、落ちれば理由。"""
    if not text.strip():
        return "まとめが空"
    if len(text) > SUMMARY_MAX_CHARS:
        return f"{SUMMARY_MAX_CHARS} 字を超えた（{len(text)} 字）"
    if not sources:
        return "出典が無い"
    bad = [s for s in sources if s not in bundle_ids]
    if bad:
        return f"出典が束の外（{bad[0]}）"
    for name in people:
        if name and name not in family:
            return f"家族に無い名前「{name}」"
    return None


def _parse(raw: str) -> "tuple[str, list[str], list[str]] | None":
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    sources = [str(s) for s in (data.get("sources") or []) if s]
    people = [str(p).strip() for p in (data.get("people") or []) if str(p).strip()]
    return str(data.get("text", "")).strip(), sources, people


def _usability(r: dict, *, now: datetime, hl: float, w_t: float, w_g: float) -> float:
    return im.usability(
        age_days=im._age_days(r.get("timestamp"), now),
        g0=float(r.get("groundedness_g0", 1.0) or 1.0),
        n=int(r.get("groundedness_n", 0) or 0),
        hl=hl,
        w_t=w_t,
        w_g=w_g,
    )


def _fold_identical(agent, records: list[dict], tau_same: float) -> tuple[list[dict], int, int]:
    """同一の群を 1 件に。残った記録・群の数・畳んだ件数を返す。"""
    if len(records) < 2:
        return records, 0, 0
    V = np.array([r["vector"] for r in records], dtype=np.float32)
    groups = bundling.duplicate_groups(V, tau_same)
    gone: set[int] = set()
    folded = 0
    for g in groups:
        rep = max(
            g,
            key=lambda i: records[i].get("timestamp") or datetime.min.replace(tzinfo=timezone.utc),
        )
        n_max = max(int(records[i].get("groundedness_n", 0) or 0) for i in g)
        rep_id = str(records[rep]["obs_id"])
        for i in g:
            if i == rep:
                continue
            if agent._oif.supersede(str(records[i]["obs_id"]), rep_id, kind=KIND_FOLD):
                folded += 1
                gone.add(i)
        agent._oif.set_groundedness(rep_id, n_max)
    kept = [r for i, r in enumerate(records) if i not in gone]
    return kept, len(groups), folded


def _timestamp_of(rows: list[dict]) -> datetime:
    ts: list[datetime] = [r["timestamp"] for r in rows if isinstance(r.get("timestamp"), datetime)]
    return max(ts) if ts else datetime.now(timezone.utc)


async def _write_summary(agent, text: str, rows: list[dict]) -> str:
    """`まとめ` を書く。時刻は出典の最新、居合わせた人は出典の面の人（自分は除く）。"""
    from .rest_fold import _pad_for

    pad, arousal = await _pad_for(agent, text)
    mi = MI(
        id="", content=text[:SUMMARY_MAX_CHARS], timestamp=_timestamp_of(rows), direction="まとめ"
    )
    if pad is not None:
        mi.pad = pad
    kw = dict(agent._observation_perspective())
    self_id = str(kw.get("writer_id") or "")
    people: list[str] = []
    for r in rows:
        for pid in r.get("person_ids") or []:
            if pid and pid != self_id and pid not in people:
                people.append(str(pid))
    kw["participants"] = people
    return str(await agent._oif.write(mi, now=True, arousal=arousal, **kw) or "")


async def fold_core(agent, *, now: "datetime | None" = None) -> CoreResult:
    now = now or datetime.now(timezone.utc)
    mem = agent.config.memory
    hl, w_t, w_g = float(mem.recall_half_life_days), float(mem.recall_w_t), float(mem.recall_w_g)
    target = float(mem.info_target_bits)
    tau_same = float(getattr(mem, "core_same_cos", 0.98))
    tau = float(getattr(mem, "core_bundle_cos", 0.5))
    min_size = int(getattr(mem, "core_bundle_min", 3))
    per_night = int(getattr(mem, "core_bundles_per_night", 6))

    records = [r for r in agent._oif.core_records() if r.get("vector")]
    records, groups, same_folded = _fold_identical(agent, records, tau_same)
    measure.record("層1同一", 群=groups, 畳んだ=same_folded, 残り=len(records))

    total = im.total(records, now=now, hl=hl, w_t=w_t, w_g=w_g)
    excess = max(0.0, total - target)
    if excess <= 0.0 or len(records) < min_size:
        measure.record("層1固め", 超過=int(excess), 束=0, 固めた=0, 畳んだ=0, 見送り=0)
        return CoreResult(groups, same_folded, total, excess)

    V = np.array([r["vector"] for r in records], dtype=np.float32)
    res = bundling.bundle(V, tau=tau, k_max=K_MAX, min_size=min_size)
    u = [_usability(r, now=now, hl=hl, w_t=w_t, w_g=w_g) for r in records]
    bits = [im.bits_of_chars(int(r.get("chars", 0) or 0)) for r in records]
    # 平均 u の低い束から。超過ぶんに達するまで、1 晩の上限まで。
    order = sorted(
        range(len(res.bundles)),
        key=lambda b: sum(u[i] for i in res.bundles[b]) / len(res.bundles[b]),
    )
    chosen: list[list[int]] = []
    covered = 0.0
    for b in order:
        if covered >= excess or len(chosen) >= per_night:
            break
        chosen.append(res.bundles[b])
        covered += sum(bits[i] * u[i] for i in res.bundles[b])
    family = family_names_of(agent)
    written = folded = skipped = 0
    outs: list[Written] = []
    for members in chosen:
        rows = [records[i] for i in members]
        ids = {str(r["obs_id"]) for r in rows}
        lines = "\n".join(
            f"{r['obs_id']} {(r.get('timestamp') or now):%Y-%m-%d} [{r.get('direction', '')}] "
            f"{str(r.get('content', ''))[:400]}"
            for r in rows
        )
        prompt = _PROMPT.format(
            max_chars=SUMMARY_MAX_CHARS, family="・".join(family) or "なし", records=lines
        )
        try:
            raw = str(await agent.backend.complete(prompt, max_tokens=600) or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("rest 核の固め：依頼に失敗（%s）", e)
            skipped += 1
            continue
        parsed = _parse(raw)
        if parsed is None:
            skipped += 1
            continue
        text, sources, people = parsed
        reason = check(text, sources, people, bundle_ids=ids, family=family)
        if reason:
            logger.info("rest 核の固め：見送り（%s）", reason)
            skipped += 1
            continue
        used = [r for r in rows if str(r["obs_id"]) in set(sources)]
        new_id = await _write_summary(agent, text, used)
        if not new_id:
            skipped += 1
            continue
        written += 1
        outs.append(Written(new_id, "core_summary", text))
        agent._oif.set_groundedness(new_id, max(int(r.get("groundedness_n", 0) or 0) for r in used))
        for r in used:
            if agent._oif.supersede(str(r["obs_id"]), new_id, kind=KIND_FOLD):
                folded += 1
    measure.record(
        "層1固め", 超過=int(excess), 束=len(chosen), 固めた=written, 畳んだ=folded, 見送り=skipped
    )
    logger.info(
        "rest 核の固め：超過 %d bit・束 %d／%d・固めた %d・畳んだ %d・見送り %d",
        excess,
        len(chosen),
        len(res.bundles),
        written,
        folded,
        skipped,
    )
    return CoreResult(
        groups, same_folded, total, excess, len(chosen), written, folded, skipped, tuple(outs)
    )
