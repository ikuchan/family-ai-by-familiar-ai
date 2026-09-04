#!/usr/bin/env python
"""未適用のマイグレーションを見せ、明示されたときだけ適用する。

`db.apply_or_hold_migrations` は DB へ繋いだ時点で走るので、`migration/` へ置いた
ファイルが意図せず本番へ入る事故があった（そのために `HOLD_MIGRATIONS` を入れた）。
この道具は逆向きで、**既定では何も適用せず**、何が未適用かだけを見せる。

    uv run python scripts/apply_migrations.py            # 未適用の一覧（書かない）
    uv run python scripts/apply_migrations.py --apply    # 適用する
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, "src")

# `.env` を自前で読む（この道具は familiar の起動経路を通らない）。
for _line in pathlib.Path(".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

import psycopg2  # noqa: E402

from familiar_agent.db_migrations import (  # noqa: E402
    apply_migrations,
    default_migration_dir,
    pending_migration_ids,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="適用する（既定は一覧のみ）")
    args = ap.parse_args()

    url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(url)
    conn.autocommit = True
    mig_dir = default_migration_dir()

    pending = pending_migration_ids(conn, mig_dir)
    if not pending:
        print("未適用はありません。")
        return 0

    print(f"未適用 {len(pending)} 件:")
    for mid in pending:
        print(f"  {mid}")

    if not args.apply:
        print("\n**適用していません。**適用するには --apply を付けてください。")
        return 0

    n = apply_migrations(conn, mig_dir)
    print(f"\n適用しました（{n} 件）。")
    print("残り:", pending_migration_ids(conn, mig_dir) or "なし")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
