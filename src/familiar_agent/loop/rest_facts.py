"""事実を抜く——REST 内省・層 1 の⑤（記-ad・2026-11-20）。

記録の本文から軽量LLM で事実（主語・関係・答え）を抜き、問いに近い形の「鍵の文」にして持つ。想起は鍵の文と
問いの近さでも記録を集め（鍵の軸）、鍵の近さを Ruri の頭に入れる（`ObservationMemory.recall`）。
言い換えた問いでも、短く書き方の揺れた記録を思い出せるようにするため（論文 AtlasKV の KG2KV の考え方）。

抜くのは、まだ印の無い本文（同じ本文の記録は 1 回にまとめる）を新しい順に、1 晩 `fact_extract_per_night` 種まで。
返りが読めなかった束と、Ruri が点にできなかった束は印を付けないので、次の晩に抜き直す。
今ある記録をまとめて埋めるのは `scripts/backfill_facts.py`（同じ `extract_facts` を回す）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ..config import MemoryConfig
from ..core import fact_keys as fk

logger = logging.getLogger(__name__)

MAX_TOKENS = 6000  # 8 本文 × 最大 8 事実が収まる量（測った返りは 1 束 約 700 トークン）
STOP_AFTER = 3  # 続けてこの数の束が失敗したら、その回は打ち切る（残高切れ・障害で失敗を重ねない）


@dataclass(frozen=True)
class FactsResult:
    """抜き待ちの本文の種類、そのうち書けた種類、事実の数、書いた鍵の文の行の数、失敗した束の数。

    `skipped` は、何もしなかった理由（軽量LLM が無い、Ruri が使えない）。したときは空。
    """

    sources: int
    done: int
    facts: int
    keys: int
    failed_batches: int
    skipped: str = ""


async def extract_facts(
    store, backend, *, limit: int, batch: int, workers: int = 1, timeout: float = 120.0
) -> FactsResult:
    """抜き待ちの本文を `batch` 種ずつ軽量LLM に聞き、鍵の文を書く。

    `store` は `fact_ready`・`fact_pending`・`write_fact_keys` を持つもの（REST では OIF、埋める道具では記憶）。
    `workers` 個の束を同時に聞き、書くのは束の順に 1 つずつ。呼び出し（`timeout` 秒の締切つき）・返り・書き込みの
    どれが落ちても、その束を失敗と数えて先へ進む。払った返りは束の途中で捨てない。DB と Ruri は別スレッドで回す。
    """
    sources = await asyncio.to_thread(store.fact_pending, limit)
    done = facts = keys = failed = streak = 0
    model = str(getattr(backend, "model", ""))
    # Ruri が使えなければ、聞いても書けない（LLM の料金だけかかる）。
    if sources and not await asyncio.to_thread(store.fact_ready):
        logger.warning("事実を抜かない（Ruri が使えない・%d 種は次の回）", len(sources))
        return FactsResult(len(sources), 0, 0, 0, 0, skipped="Ruri が使えない")
    chunks = [sources[i : i + batch] for i in range(0, len(sources), batch)]

    async def ask(chunk):
        prompt = fk.build_prompt([s.content for s in chunk])
        logger.debug("事実を抜く頼み（%d 本文・%d 字）：%s", len(chunk), len(prompt), prompt)
        raw = await asyncio.wait_for(backend.complete(prompt, max_tokens=MAX_TOKENS), timeout)
        return fk.parse(raw, len(chunk))

    for start in range(0, len(chunks), workers):
        if streak >= STOP_AFTER:
            logger.warning("事実を抜くのを打ち切った（%d 束続けて失敗・残りは次の回）", streak)
            break
        group = chunks[start : start + workers]
        got = await asyncio.gather(*(ask(c) for c in group), return_exceptions=True)
        for chunk, parsed in zip(group, got):
            if isinstance(parsed, asyncio.CancelledError):
                raise parsed
            if isinstance(parsed, BaseException):
                logger.warning(
                    "事実を抜く呼び出しが落ちた（%d 本文・次の回に抜き直す）",
                    len(chunk),
                    exc_info=parsed,
                )
            elif parsed is None:
                logger.warning(
                    "事実を抜く返りが読めなかった（%d 本文・次の回に抜き直す）", len(chunk)
                )
            else:
                n = None
                try:
                    n = await asyncio.to_thread(store.write_fact_keys, chunk, parsed, model=model)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "鍵の文を書けなかった（%d 本文・次の回に抜き直す）", len(chunk)
                    )
                if n is not None:
                    streak = 0
                    done += n[0]
                    facts += sum(len(f) for f in parsed if f is not None)
                    keys += n[1]
                    continue
            failed += 1
            streak += 1
    result = FactsResult(len(sources), done, facts, keys, failed)
    logger.info(
        "事実を抜いた：抜き待ち %d 種のうち %d 種・事実 %d 個・鍵の文 %d 行・失敗 %d 束（%s）",
        result.sources,
        result.done,
        result.facts,
        result.keys,
        result.failed_batches,
        model,
    )
    return result


async def extract_facts_tonight(agent) -> FactsResult:
    """REST 内省から呼ぶ口。軽量LLM と、1 晩の上限と、並べる束の数と締切（`MemoryConfig`）を使う。

    軽量LLM の設定が無いと `_utility_backend` は主LLM になる。高いモデルで毎晩抜かないよう、そのときは何もしない。
    """
    if agent._utility_backend is agent.backend:
        logger.info("事実を抜かない（軽量LLM が無く、主LLM しか無い）")
        return FactsResult(0, 0, 0, 0, 0, skipped="軽量LLM が無い")
    cfg = MemoryConfig()
    return await extract_facts(
        agent._oif,
        agent._utility_backend,
        limit=cfg.fact_extract_per_night,
        batch=cfg.fact_extract_batch,
        workers=cfg.fact_extract_workers,
        timeout=cfg.fact_extract_timeout_sec,
    )
