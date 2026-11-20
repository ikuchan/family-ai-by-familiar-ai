#!/usr/bin/env python3
"""今ある記録から事実を抜き、鍵の文を埋める（記-ad）。中身は `familiar_agent.loop.rest_facts.extract_facts`。

パジュを止めている間に、本人が回す（本番 DB に書き、軽量LLM を呼ぶ）。接続は `get_db()` を通るので、未適用の
マイグレーション（080）があればここで当たる。印の無い本文だけを抜くので、途中で止まっても、何度回してもよい。
3,392 種で測ったときは 約 0.3 ドル・4 つ並べて 5 分だった。

    uv run python scripts/backfill_facts.py
    uv run python scripts/backfill_facts.py --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import familiar_agent.config as config  # noqa: E402  .env を読む（DATABASE_URL・鍵）。アプリと同じ読み方にする
from familiar_agent.backends.factory import create_utility_backend  # noqa: E402
from familiar_agent.config import MemoryConfig  # noqa: E402
from familiar_agent.loop.rest_facts import extract_facts  # noqa: E402
from familiar_agent.store.ruri_backfill import app_is_running  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100_000, help="抜く本文の種類の上限")
    ap.add_argument("--workers", type=int, default=None, help="同時に聞く束の数（既定は設定値）")
    args = ap.parse_args()
    if app_is_running():
        print(
            "パジュが動いています。止めてから回してください（Ruri の読み込みが本体と重なるため）。"
        )
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backend = create_utility_backend(config.AgentConfig())
    if backend is None:
        print("軽量LLM の設定がありません（utility_platform と鍵）。")
        return 1
    os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")
    from familiar_agent.tools.memory import ObservationMemory

    memory = ObservationMemory()
    cfg = MemoryConfig()
    r = asyncio.run(
        extract_facts(
            memory,
            backend,
            limit=args.limit,
            batch=cfg.fact_extract_batch,
            workers=args.workers or cfg.fact_extract_workers,
            timeout=cfg.fact_extract_timeout_sec,
        )
    )
    print(
        f"抜き待ち {r.sources} 種のうち {r.done} 種・事実 {r.facts} 個・鍵の文 {r.keys} 行・"
        f"失敗 {r.failed_batches} 束{('・' + r.skipped) if r.skipped else ''}"
        "（失敗した束は、もう一度回せば抜き直す）"
    )
    return 0 if r.failed_batches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
