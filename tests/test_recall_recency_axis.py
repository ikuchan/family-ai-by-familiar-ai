"""新しさ軸の一次絞り（[D-想起合成] の多軸 union）。

設計はこう定めている。

> 候補集合＝多軸 union 一次絞り：W に載せる候補は、重みプロファイルで重み>0 の各軸で
> `ORDER BY … LIMIT N` を出し UNION して集め、その和集合に対してアプリが積 score を
> 再計算する

実装は関連軸（`by_vector`）だけで候補を作っていた。そのため話題が近くない限り直近の
記録が候補にすら入らず、t 軸は「候補に入ったあとの並べ替え」にしか効かなかった。
直前の会話を思い出せないのはこれが理由（実機で「それだけ？」に聞き返した）。
"""

from __future__ import annotations

from familiar_agent.store.observations import ObservationStore


def test_by_recency_orders_by_timestamp_and_skips_superseded():
    import inspect

    src = inspect.getsource(ObservationStore.by_recency)
    assert "ORDER BY o.timestamp DESC" in src
    assert "o.superseded_by IS NULL" in src      # 死んだ記録は候補にしない
    assert "s.person_id = %s" in src             # 視点スコープは関連軸と揃える


def test_by_recency_returns_the_same_columns_as_by_vector():
    # 採点器は両軸の行を区別せず扱うので、列が揃っていないと落ちる。
    import inspect
    import re

    def cols(fn):
        return set(re.findall(r"o\.(\w+)", inspect.getsource(fn)))

    assert cols(ObservationStore.by_vector) - {"id"} <= cols(ObservationStore.by_recency)
