"""事実の置き換えを判定する——REST 内省・層 1 の⑤の続き（記-ah 段 2・2026-11-25）。

その晩に抜いた事実の「答えを隠す向き」の鍵の文（話題）ごとに、意味の近い古い記録の鍵を候補にし（コード）、組ごとに
両方の本文と日付を軽量LLM にまとめて渡して、置き換えたかといつからかを判定させる（本人の決定）。置き換えていれば、
種類「置き換え」の関係を書く（古い記録は隠さない・`store/relations.RelationStore.record_replacement`）。いつからが
分からなければ、新しい記録の時刻を使う。考え方は `設計方針_事実の移り変わり`。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

from ..config import MemoryConfig
from ..core import fact_replace as fr
from ..core import state_json
from ..store import clock
from ..store.clock import local_tz

logger = logging.getLogger(__name__)

MAX_TOKENS = 2000  # 10 組の判定が収まる量
#: どこまで判定し終えたか（事実を抜いた時刻）の控え（`agent_state`）。束が失敗した晩は進めず、次の晩に聞き直す。
STATE_KEY = "fact_replace_until"


@dataclass(frozen=True)
class ReplaceResult:
    """調べた組の数、置き換えと判定して書いた数、失敗した束の数。`skipped` は何もしなかった理由。"""

    pairs: int
    replaced: int
    failed_batches: int
    skipped: str = ""


def _since(verdict: "fr.Verdict", pair: "fr.Pair") -> datetime:
    """置き換えた事実がいつから成り立ったか。返りに日付が無いか、古い記録と新しい記録のあいだに無ければ新しい記録の
    時刻。日付は家の暦の 0 時。"""
    if verdict.since:
        try:
            day = datetime.strptime(verdict.since, "%Y-%m-%d").replace(tzinfo=local_tz())
        except ValueError:
            return pair.new_at
        if pair.old_at <= day <= pair.new_at:
            return day
    return pair.new_at


def _start(since: datetime) -> datetime:
    """判定にかける事実を抜いた時刻の下限。控えがあれば控えから（前の晩に失敗した束を聞き直す）。"""
    raw = state_json.read(STATE_KEY)
    if isinstance(raw, str):
        try:
            return min(since, datetime.fromisoformat(raw))
        except ValueError:
            pass
    return since


async def judge(agent, *, since: datetime, batch: "int | None" = None) -> ReplaceResult:
    """`since` より後に抜いた事実の置き換えを判定して書く。軽量LLM が無ければ何もしない（主LLM で毎晩聞かない）。

    本文が同じ組（同じ話題・同じ古い本文・同じ新しい本文）は 1 回だけ聞き、答えをその本文を持つ記録すべてに書く。
    1 晩に聞く組は `fact_replace_per_night` まで。束が失敗しなければ控えを進め、失敗した晩は次の晩に聞き直す。
    """
    backend = agent._utility_backend
    if backend is agent.backend:
        return ReplaceResult(0, 0, 0, skipped="軽量LLM が無い")
    cfg = MemoryConfig()
    size = batch or cfg.fact_replace_batch
    began = clock.now_utc()
    pairs = await asyncio.to_thread(
        agent._oif.replace_candidates,
        _start(since),
        per_key=cfg.fact_replace_per_key,
        min_cos=cfg.fact_replace_min_cos,
    )
    groups: "dict[tuple[str, str, str], list[fr.Pair]]" = {}
    for p in pairs:
        groups.setdefault((p.topic, p.old_text, p.new_text), []).append(p)
    asks = [members for members in groups.values()][: cfg.fact_replace_per_night]
    replaced = failed = 0
    for start in range(0, len(asks), size):
        chunk = asks[start : start + size]
        prompt = fr.build_prompt([members[0] for members in chunk])
        logger.debug("置き換えの判定の頼み（%d 組・%d 字）：%s", len(chunk), len(prompt), prompt)
        try:
            raw = await asyncio.wait_for(
                backend.complete(prompt, max_tokens=MAX_TOKENS), cfg.fact_extract_timeout_sec
            )
            verdicts = fr.parse(raw, len(chunk))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.warning(
                "置き換えの判定を呼べなかった（%d 組・次の晩に聞き直す）", len(chunk), exc_info=True
            )
            verdicts = None
        if verdicts is None:
            failed += 1
            continue
        for members, v in zip(chunk, verdicts):
            if v is None or not v.replaced:
                continue
            for pair in members:
                rid = await asyncio.to_thread(
                    agent._oif.record_replacement,
                    pair.old_obs,
                    pair.old_key,
                    pair.new_obs,
                    pair.new_key,
                    valid_at=_since(v, pair),
                )
                if rid is not None:
                    replaced += 1
    if not failed and len(asks) == len(groups):
        state_json.write(STATE_KEY, began.isoformat())  # 聞き残しが無いときだけ進める
    logger.info(
        "rest 事実の置き換え：候補 %d 組（聞いた %d 種）・置き換え %d・失敗 %d 束",
        len(pairs),
        len(asks),
        replaced,
        failed,
    )
    return ReplaceResult(len(pairs), replaced, failed)
