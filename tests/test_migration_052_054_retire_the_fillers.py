"""052〜054：つなぎの発話を記憶から外し、宙に浮いた参照を掃除する。

**「つなぎ」は間をつなぐ一言である**（`つなぎに言った：ちょっと待ってね。` など）。
2026-08-21 のダンプの `observations_removed_fillers` には 337 行が入っており、すべて
`direction='発話' AND kind='observation'` で本文が `つなぎに言った：` で始まる。

```
つなぎに言った：ちょっと待ってね。               12
つなぎに言った：はい。                           10
つなぎに言った：えっと、                          5
```

**記録には理由が書かれていたが、その理由は別の仕組みが満たしていた。** 「残さないと、
次の反復の W に『もう一言伝えた』事実が入らず、調停は同じことをまた言う」——だが
`_said_fillers` が**プロンプトへ直接載る**ので（「すでに相手へ伝えた一言」）、O に残さなくても
次の反復には伝わる。二重に持っていた。

**`superseded_by` には外部キーが無い。** `parent_id` は `ON DELETE SET NULL` で自動的に
外れるが、`superseded_by` は行を消すと宙に浮く。053 はそれを掃除する（051 が消した
`self_model` 1068 行を指していた参照）。**053 の中身は推測を含む**——退避表が無いので、
残った状態からの逆算である（`復旧記録` v0.21）。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import psycopg2
import psycopg2.extras


_DB_URL = os.environ["DATABASE_URL"]
_FOLD = "2026-08-15-052_fold_the_filler_utterances.py"
_DANGLING = "2026-08-15-053_drop_the_dangling_rows.py"
_RETIRE = "2026-08-16-054_retire_the_fillers.py"


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = True
    return c


def _run(name: str) -> None:
    path = Path(__file__).parent.parent / "migration" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    conn = psycopg2.connect(_DB_URL)
    try:
        mod.upgrade(conn)
        conn.commit()
    finally:
        conn.close()


# ── 052：つなぎを求めの親へ畳む ────────────────────────────────────────────


# ── 053：消えた行を指す参照を外す ──────────────────────────────────────────


# ── 054：つなぎを退避し、以後は記録しない ──────────────────────────────────


def test_the_loop_records_the_filler_again_but_hides_it() -> None:
    """つなぎを O へ書く経路は戻した（段 4）。ただし想起には出さない。

    054 が外したのは、想起の候補を食うからだった。役割が「想起に出さない」を担う形に
    なったので（`設計方針_MI間の関係` 段 2）、**項として持ちながら想起から外せる**。
    退避した 337 行は戻さない。
    """
    import inspect

    from familiar_agent.loop import event_loop
    from familiar_agent.store.relations import HIDDEN_ROLES

    assert "つなぎに言った" in inspect.getsource(event_loop), "つなぎを O へ書いていない"
    assert "つなぎ" in HIDDEN_ROLES, "つなぎが想起に出てしまう"


def test_the_said_fillers_list_stays() -> None:
    """**プロンプトへ載せる側は残す**（反証側）。

    「もう一言伝えた」を次の反復へ伝えるのは `_said_fillers` の役目で、O への記録は
    二重に持っていたぶんである。こちらまで外すと、同じ言い回しを最初から言い直す。
    """
    import inspect

    from familiar_agent.loop import event_loop

    src = inspect.getsource(event_loop)
    assert "_said_fillers" in src
    assert "すでに相手へ伝えた一言" in src
