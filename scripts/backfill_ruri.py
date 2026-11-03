#!/usr/bin/env python3
"""今ある記録の Ruri の数値を埋める（記-y）。中身は `familiar_agent.store.ruri_backfill`。

パジュを止めている間に、本人が回す（本番 DB に書く）。接続は `get_db()` を通るので、未適用の
マイグレーション（079）があればここで当たる。表に行が無い記録だけを埋めるので、何度回してもよい。

    uv run python scripts/backfill_ruri.py
    uv run python scripts/backfill_ruri.py --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import familiar_agent.config  # noqa: E402,F401  .env を読む（DATABASE_URL）。アプリと同じ読み方にする
from familiar_agent.db import get_db  # noqa: E402
from familiar_agent.store.ruri_backfill import app_is_running, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if app_is_running():
        print(
            "パジュが動いています。止めてから回してください（Ruri の読み込みが本体と重なるため）。"
        )
        return 2

    return run(get_db(), batch_size=args.batch_size, out=lambda line: print(line, flush=True))


if __name__ == "__main__":
    raise SystemExit(main())
