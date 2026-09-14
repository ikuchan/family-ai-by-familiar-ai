"""読み手の無いストアを落とす（環-d・2026-09-14）。

- `self_narrative_log`：セッション日記。REST 内省の層 1（日次の畳み込み）が代替する
  （2026-09-05「移管しない」決定）。書き手 `SelfNarrative` はこの日に撤去。
- `semantic_facts`／`behavior_policies`／`memory_links`／`memory_revisions`：
  `legacy/semantic_layer.py` だけが書き、読む側（`recall_semantic_facts`・`recall_behavior_policies`・
  `get_linked_memories`・`recall_revisions`）の呼び手が 0 件だった。層ごと撤去。

**改名して残す作法（060）は取らない。** 本番のバックアップを確かめたうえでの決定（2026-09-14）。
FK は落とす表が子（`persons`・`observations` を参照）なので、親には触れない。
"""

from __future__ import annotations

TABLES = (
    "self_narrative_log",
    "semantic_facts",
    "behavior_policies",
    "memory_links",
    "memory_revisions",
)


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for t in TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {t}")
