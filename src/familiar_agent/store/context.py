"""層が共有する文脈。

各層（観測・situated・キュー・撤去予定の意味層）は、宿主の名前空間を覗かずに、
ここから接続とロックと person と埋め込み器を受け取る。共有するものをこの4つに
絞ってあるのは、層の間で何が共通なのかを一目で読めるようにするためである。
増やすときは意図的に増やす。

接続の作り方（Database シングルトンか、テストの sqlite3 か）と、マイグレーションの
適用は、ここに閉じる。層はその違いを知らない。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any

from ..db_migrations import apply_migrations, default_migration_dir
from .db_compat import _RealDictConnWrapper, _SQLiteConnWrapper


def viewpoint_of(person_id: str) -> str:
    """その person で想起するとき、どの面を通って引くか（047）。

    **`default` は視点ではない。** 書き込み側で「話者がまだ分からない」を表す置き場で
    あって、人ではない。関係の面（`situated_memories`）は実在の人と `__self__` にしか
    立たないので、`default` のまま引くと母集合が空になる。

    話者が解決できなかった記録はパジュ自身がしたことなので（048「内なる記録は
    エージェントのもの」）、視点も `__self__` へ寄せる。実物（2026-08-21）でも
    `default` の面は1件も無く、`last_recalled_at` が入るのは `__self__` の行だけだった。

    **書き込み側の `person_id` は変えない。** 誰が書いたか（`writer_id`）と、誰の視点で
    引くかは別の問いである。規則はここ1箇所に置く。
    """
    from ..person_memory_manager import AGENT_SELF_ID, DEFAULT_PERSON_ID

    return AGENT_SELF_ID if person_id == DEFAULT_PERSON_ID else person_id


@dataclass
class StoreContext:
    """層が共有する道具。"""

    db: Any                  # Database シングルトン、またはテストの sqlite3 接続
    lock: threading.Lock
    person_id: str
    embedder: Any

    @property
    def viewpoint(self) -> str:
        """想起がどの面を通って引くか（`viewpoint_of` の規則）。"""
        return viewpoint_of(self.person_id)

    def conn(self) -> Any:
        """現在の接続を返す。未適用のマイグレーションがあればここで当てる。"""
        if callable(getattr(self.db, "conn", None)):
            conn = self.db.conn()
            apply_migrations(conn, default_migration_dir())
            return _RealDictConnWrapper(conn)
        # テストで使う素の sqlite3 接続
        return _SQLiteConnWrapper(self.db)

    def for_person(self, person_id: str) -> "StoreContext":
        """person だけ差し替えた文脈。接続とロックと埋め込み器は共有する。"""
        return replace(self, person_id=person_id)
