#!/usr/bin/env python
"""新しい目盛りで PAD を測り直す（案A・一度きり）。

057 が全行を未測定へ戻したあと、この道具が `emotion_a >= A_GATE` の行を測り直す。
測るのは P・Pn・Dom の3つで、**A は触らない**（機械値で、評価器へ渡していない）。

マイグレーションにしていないのは、マイグレーションが開発とテストのたびに走るためである。
軽量LLM を呼ぶ処理をそこへ置くと、API 鍵と課金と数分の待ちがテストに入り込む。

既定は読むだけで、書き込むには `--apply` を明示する。

    uv run python scripts/remeasure_emotion_pad.py            # 下見（書かない）
    uv run python scripts/remeasure_emotion_pad.py --apply    # 書き戻す
    uv run python scripts/remeasure_emotion_pad.py --limit 50 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, "src")

# `.env` を自前で読む（この道具は familiar の起動経路を通らない）。
for _line in pathlib.Path(".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import psycopg2  # noqa: E402
import psycopg2.extras  # noqa: E402

from familiar_agent.backends import create_backend, create_utility_backend  # noqa: E402
from familiar_agent.config import AgentConfig  # noqa: E402
from familiar_agent.emotion_pad import label_from_pad, pad_to_search_vector  # noqa: E402
from familiar_agent.loop.evaluator import A_GATE, _evaluate_emotion_pad  # noqa: E402
from familiar_agent.mood_register import REST_PAD, MoodPAD  # noqa: E402

#: 測り直しの基準 mood。当時の気分は残っていないので、**全行に平静を渡す**。
#: 実測では、基準を揃えても場面のあいだの順序は保たれた（根拠台帳 §25.2）。
_BASELINE = REST_PAD


def _rows(conn, limit: int | None) -> list[dict]:
    sql = ("SELECT id, content, emotion_a FROM observations "
           "WHERE emotion_p IS NULL AND emotion_a >= %s ORDER BY timestamp")
    args: list = [A_GATE]
    if limit:
        sql += " LIMIT %s"
        args.append(limit)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return [dict(r) for r in cur.fetchall()]


async def _measure(backend, row: dict, sem: asyncio.Semaphore) -> tuple[str, MoodPAD | None]:
    async with sem:
        pad, _a = await _evaluate_emotion_pad(
            backend, row["content"] or "", _BASELINE, float(row["emotion_a"]),
        )
    return row["id"], pad


def _write(conn, oid: str, pad: MoodPAD) -> None:
    vec = "[" + ",".join(
        f"{v:.6f}" for v in pad_to_search_vector((pad.p, pad.pn, pad.a, pad.dom))
    ) + "]"
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE observations SET emotion_p=%s, emotion_pn=%s, emotion_dom=%s, "
            "emotion_vec=%s, emotion=%s WHERE id=%s",
            (pad.p, pad.pn, pad.dom, vec, label_from_pad(pad), oid),
        )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="書き戻す（既定は下見のみ）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    conn = psycopg2.connect(os.environ["DATABASE_URL"],
                            cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = True
    rows = _rows(conn, args.limit)
    print(f"対象 {len(rows)} 行（emotion_a >= {A_GATE} かつ PAD 未測定）")
    if not rows:
        return 0

    cfg = AgentConfig()
    backend = create_utility_backend(cfg) or create_backend(cfg)
    print(f"軽量LLM: {type(backend).__name__} / {getattr(backend, 'model', '?')}")
    print(f"基準 mood: {_BASELINE.p:.2f} {_BASELINE.pn:.2f} {_BASELINE.dom:.2f}")
    print("書き戻す" if args.apply else "**下見のみ。書き戻さない**")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()
    done = miss = 0
    labels: collections.Counter[str] = collections.Counter()
    ps: list[float] = []
    for chunk_start in range(0, len(rows), 200):
        chunk = rows[chunk_start:chunk_start + 200]
        for oid, pad in await asyncio.gather(*(_measure(backend, r, sem) for r in chunk)):
            done += 1
            if pad is None:
                miss += 1
                continue
            labels[label_from_pad(pad)] += 1
            ps.append(pad.p)
            if args.apply:
                _write(conn, oid, pad)
        print(f"  {done}/{len(rows)}  形を外した {miss}  経過 {time.monotonic()-t0:.0f}s",
              flush=True)

    print(f"\n測れた {done - miss} / 外した {miss} / {time.monotonic()-t0:.0f}秒")
    if ps:
        print(f"P の中央値 {statistics.median(ps):.2f}")
    total = sum(labels.values()) or 1
    for label, n in labels.most_common():
        print(f"  {label:10s} {n:5d}  {100*n/total:5.1f}%")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
