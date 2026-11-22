#!/usr/bin/env python3
"""今ある発話に、話題にされた人の about の面を埋める（記-ae 段 3）。中身は `familiar_agent.loop.rest_about.add_about`。

about の面を立てていた段②のコードは 2026-08 に失われ、9 月からの発話には about の面が無い。REST は毎晩 2 日ぶんしか
見ないので、それより古い分をここで埋める。パジュを止めている間に回す（本番 DB に書き、Jev を呼ぶ。開発するエージェントも
承認なしで回してよい）。書くのはアプリの口（`OIF.add_about`）だけ。about の面がもう立っている人には聞き直さないので、
途中で止まっても、何度回してもよい。調べた印は持たないので、Jev が「いいえ」と答えた発話は、回すたびに新しい順に聞き直す
（問いの上限 `--limit` で区切っても、古い分へ進まない。期間は `--since`・`--until` で区切る）。問いは 1 問ずつ順に聞く。

    uv run python scripts/backfill_about.py --since 2026-09-01 --until 2026-09-03   # 少し回して数える
    uv run python scripts/backfill_about.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import familiar_agent.config as config  # noqa: E402,F401  .env を読む（DATABASE_URL・鍵）。アプリと同じ読み方にする
from familiar_agent.backends.jev import JevClient  # noqa: E402
from familiar_agent.config import MemoryConfig  # noqa: E402
from familiar_agent.io.oif import OIF  # noqa: E402
from familiar_agent.loop.rest_about import add_about  # noqa: E402
from familiar_agent.store import clock  # noqa: E402
from familiar_agent.store.ruri_backfill import app_is_running  # noqa: E402

#: 既定の始まり。8/11〜8/20 に失われたコードが立てた面は、人ごとに聞き直さない（`done`）。
_SINCE = "2026-08-01"


def _at(text: str) -> datetime:
    """時刻の文字。時差が無ければ家の暦の時差を付ける（書いてあれば、そのまま）。"""
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=clock.local_tz())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=100_000, help="Jev に聞く問いの上限")
    ap.add_argument(
        "--since",
        default=_SINCE,
        help="この時刻より後の発話を見る（YYYY-MM-DD なら家の暦のその日の 0 時）",
    )
    ap.add_argument("--until", default=None, help="この時刻より前の発話を見る（無ければ今まで）")
    args = ap.parse_args()
    if app_is_running():
        print(
            "パジュが動いています。止めてから回してください（同じ記録に REST と重なって書かないため）。"
        )
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = MemoryConfig()
    jev = JevClient.from_env(timeout=cfg.about_backfill_timeout_sec)
    if not jev.available:
        print("Jev の鍵がありません（JEV_API_KEY）。")
        return 1
    os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")
    from familiar_agent.tools.memory import ObservationMemory

    r = asyncio.run(
        add_about(
            OIF(ObservationMemory()),
            jev,
            limit=args.limit,
            min_p=cfg.about_min_p,
            since=_at(args.since),
            until=_at(args.until) if args.until else None,
        )
    )
    print(
        f"発話 {r.sources} 件・問い {r.asked}・足した面 {r.added}・答えなし {r.unanswered}・"
        f"書けなかった {r.unwritten}{('・' + r.skipped) if r.skipped else ''}"
        "（答えなしは、もう一度回せば聞き直す）"
    )
    return 1 if r.skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
