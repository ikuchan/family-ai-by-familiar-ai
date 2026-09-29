"""`memory_salience`・`episode_memories`・`episodes` 表を落とす（環-ab D・2026-09-29）。

`memory_salience` は W（作業記憶）を DB へ溜めていた旧方式の表で、W はいま O から毎反復に組み直す（[D-記憶単一化]）。
`episodes`・`episode_memories` はエピソードの表で、明示のつながりは関係（`relations`）と拡散想起へ移った。読み書き
していた口（`get_working_memory`・`refresh_working_memory`・`create_episode`・`append_to_episode`）は呼び手が無く、
環-ab D で先に外した。`memory_salience` と `episode_memories` が `episodes` を指すので、指す側から落とす。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS memory_salience")
        cur.execute("DROP TABLE IF EXISTS episode_memories")
        cur.execute("DROP TABLE IF EXISTS episodes")
