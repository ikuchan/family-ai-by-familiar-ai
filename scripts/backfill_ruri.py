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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from familiar_agent.db import get_db  # noqa: E402
from familiar_agent.store.embedding import _RuriModel  # noqa: E402
from familiar_agent.store.ruri_backfill import app_is_running, fill_missing  # noqa: E402


def _counts(conn) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS c FROM observations")
        obs = cur.fetchone()
        cur.execute("SELECT count(*) AS c FROM obs_ruri_embeddings")
        ruri = cur.fetchone()
    conn.rollback()
    pick = lambda r: int(r["c"] if isinstance(r, dict) else r[0])  # noqa: E731
    return pick(obs), pick(ruri)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if app_is_running():
        print(
            "パジュが動いています。止めてから回してください（Ruri の読み込みが本体と重なるため）。"
        )
        return 2

    db = get_db()
    started = time.monotonic()
    with db.lock:
        conn = db.conn()
        obs, ruri = _counts(conn)
        print(f"はじめ：記録 {obs} 件・Ruri の行 {ruri} 件", flush=True)

        def report(done: int, total: int) -> None:
            if done == total or done % (args.batch_size * 50) == 0:
                print(f"  {done}/{total} 件（{time.monotonic() - started:.0f} 秒）", flush=True)

        written, failed = fill_missing(
            conn, _RuriModel(), batch_size=args.batch_size, report=report
        )
        obs, ruri = _counts(conn)
    print(f"書いた {written} 件・飛ばした {failed} 件・{time.monotonic() - started:.0f} 秒")
    print(f"おわり：記録 {obs} 件・Ruri の行 {ruri} 件")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
