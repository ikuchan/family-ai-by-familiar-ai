#!/usr/bin/env python
"""人物表を `FAMILY.md` に合わせて統合する（知-af 段 4）。中身は `familiar_agent.store.person_merge`。

記憶の付いた古い行を残して名前と呼び方を `FAMILY.md` に合わせ、記憶 0 件の重なった行とテンプレートから作られた
ゴミの行を消す。記憶の付け替えはしない。**既定では何も書かない**（計画を見せて巻き戻す）。

    uv run python scripts/merge_persons.py            # 計画を見せる（書かない）
    uv run python scripts/merge_persons.py --apply    # 当てる（1 つのトランザクション）

前に：アプリを止める → `scripts/backup_db.sh` → `scripts/apply_migrations.py`（未適用があれば止まる）。
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

from familiar_agent.core.parsing import parse_family_md  # noqa: E402
from familiar_agent.db_migrations import default_migration_dir, pending_migration_ids  # noqa: E402
from familiar_agent.store import person_merge as pm  # noqa: E402


def _family_md() -> str:
    """アプリと同じ順（リポジトリ直下 → `~/.familiar_ai/`）で `FAMILY.md` を読む。"""
    for path in (pathlib.Path("FAMILY.md"), pathlib.Path.home() / ".familiar_ai" / "FAMILY.md"):
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="当てる（既定は計画を見せるだけ）")
    args = ap.parse_args()

    members = parse_family_md(_family_md())
    if not members:
        print("FAMILY.md に家族が居ません。止めます。")
        return 1

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        pending = pending_migration_ids(conn, default_migration_dir())
        conn.rollback()
        if pending:
            print(
                f"未適用のマイグレーションが {len(pending)} 件あります。先に scripts/apply_migrations.py を。"
            )
            return 1

        rows = pm.persons(conn)
        refs = pm.references(conn)
        print("参照している表：" + "、".join(pm.referencing_tables(conn)))
        print(f"\n人物表 {len(rows)} 行:")
        for r in rows:
            print(
                f"  {r['id']}  {r['name']}  （{r['display_name'] or ''}）  参照 {refs.get(str(r['id']), 0)} 件"
            )
        print("\nFAMILY.md:")
        for m in members:
            print(f"  {m['name']}  （{m['display_name']}）")

        plan = pm.make_plan(rows, refs, members)
        print("\n計画:")
        for d in plan.deletes:
            print(f"  消す      {d.id}  {d.name}  — {d.why}")
        for rn in plan.renames:
            print(f"  書き換える {rn.id}  {rn.old_name} → {rn.new_name}（{rn.new_display}）")
        for n in plan.notes:
            print(f"  残す      {n}")
        if not (plan.deletes or plan.renames):
            print("  変えるものはありません。")
        if plan.stops:
            print("\n**止まります（何も書きません）:**")
            for s in plan.stops:
                print(f"  {s}")
            return 1

        pm.apply_plan(conn, plan)
        if not args.apply:
            conn.rollback()
            print("\n**書いていません。**当てるには --apply を付けてください。")
            return 0
        conn.commit()
        print(f"\n当てました（消した {len(plan.deletes)} 行・書き換えた {len(plan.renames)} 行）。")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
