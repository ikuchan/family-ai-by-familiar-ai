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


@dataclass
class StoreContext:
    """層が共有する道具。"""

    db: Any                  # Database シングルトン、またはテストの sqlite3 接続
    lock: threading.Lock
    person_id: str
    embedder: Any

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
