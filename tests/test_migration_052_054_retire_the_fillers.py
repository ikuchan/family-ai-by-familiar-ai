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
import uuid
from datetime import datetime, timezone
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


def _plant(content: str, *, direction="発話", kind="observation",
           parent_id=None, superseded_by=None) -> str:
    obs_id = str(uuid.uuid4())
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations "
                "(id, content, timestamp, direction, kind, emotion, parent_id, superseded_by) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (obs_id, content, datetime.now(timezone.utc), direction, kind,
                 "neutral", parent_id, superseded_by),
            )
    finally:
        conn.close()
    return obs_id


def _get(obs_id: str, table: str = "observations") -> dict | None:
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table} WHERE id = %s", (obs_id,))
            row = cur.fetchone()
            return None if row is None else dict(row)
    finally:
        conn.close()


# ── 052：つなぎを求めの親へ畳む ────────────────────────────────────────────

def test_052_folds_the_filler_into_its_request() -> None:
    """つなぎは、その求めへ畳む。単体で想起に出続けるものではない。"""
    parent = _plant("「天気を調べて」と聞かれた", direction="求め")
    filler = _plant("つなぎに言った：ちょっと待ってね。", parent_id=parent)
    other = _plant("普通の発話", parent_id=parent)

    _run(_FOLD)

    assert _get(filler)["superseded_by"] == parent, "つなぎが畳まれていない"
    assert _get(other)["superseded_by"] is None, "つなぎでない発話まで畳んでいる"


# ── 053：消えた行を指す参照を外す ──────────────────────────────────────────

def test_053_clears_references_to_rows_that_are_gone() -> None:
    """`superseded_by` には外部キーが無いので、消えた行を指したまま残る。

    `parent_id` は `ON DELETE SET NULL` で自動的に外れるので、掃除が要るのは
    `superseded_by` の側だけである。
    """
    gone = str(uuid.uuid4())          # 実在しない id
    dangling = _plant("畳先が消えた記録", superseded_by=gone)
    alive_target = _plant("生きている畳先")
    ok = _plant("正しく畳まれた記録", superseded_by=alive_target)

    _run(_DANGLING)

    assert _get(dangling)["superseded_by"] is None, "宙に浮いた参照が残っている"
    assert _get(ok)["superseded_by"] == alive_target, "生きている参照まで外している"


# ── 054：つなぎを退避し、以後は記録しない ──────────────────────────────────

def test_054_moves_the_fillers_to_the_quarantine_table() -> None:
    """削除でなく退避。戻せる形で残す。"""
    filler = _plant("つなぎに言った：はい。")
    other = _plant("つなぎではない発話")

    _run(_RETIRE)

    assert _get(filler) is None, "観測から消えていない"
    assert _get(filler, "observations_removed_fillers") is not None, "退避表に残っていない"
    assert _get(other) is not None, "関係のない発話まで動かしている"


def test_054_is_idempotent() -> None:
    _run(_RETIRE)
    _run(_RETIRE)


def test_the_loop_no_longer_records_the_filler() -> None:
    """つなぎを O へ書く経路そのものを外した。"""
    import inspect

    from familiar_agent.loop import event_loop

    assert "つなぎに言った" not in inspect.getsource(event_loop), (
        "つなぎを O へ書く経路が残っている"
    )


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
