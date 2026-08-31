"""039（読み手の居ない2列の撤去）が効いていることを確かめる。

`observations.importance` は P-1 で「日次減衰は使わない・時間減衰は t 軸へ一元化」と
決めたときに役目を失った。読むのは MI の組み立て1箇所だけで、その MI は返り値に使われて
いなかった。書き手 `decay_importance` は本番からの呼び出しが0件だった。

`observations.scope` は書くだけで誰も読んでいない（SELECT・columns 指定・WHERE のいずれ
にも現れない）。道具 `remember` の `scope` 引数は**別物**で、「誰のぶんを書くか」を分岐
させる制御である。分岐が決めた相手は `writer_id`／`subject_id`／`participants_json` に
残るので、列を落としても失われない。
"""

from __future__ import annotations

import os

import psycopg2

_DB_URL = os.environ["DATABASE_URL"]


def _columns(table: str) -> set[str]:
    conn = psycopg2.connect(_DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND table_schema = 'public'",
                (table,),
            )
            return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


def test_dead_columns_are_gone() -> None:
    cols = _columns("observations")
    assert "importance" not in cols
    assert "scope" not in cols


def test_groundedness_g0_remains_as_the_successor() -> None:
    """`importance` の値は 021 で `groundedness_g0` へ移してある。落とすのは旧名だけ。"""
    assert "groundedness_g0" in _columns("observations")


def test_daily_decay_is_gone() -> None:
    """日次減衰の器も落ちていること（P-1 で役目を失い、本番呼び出しは0件だった）。"""
    from familiar_agent.tools.memory import ObservationMemory

    for name in ("decay_importance", "decay_importance_async"):
        assert not hasattr(ObservationMemory, name), name
