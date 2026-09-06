"""MI 間の関係を表す器を置く（`設計方針_MI間の関係` v0.1・段 1）。

**二項の列では、一つのやりとりを一つの関係として指せない。** `superseded_by` は二つの
記録しか結べないので、問いと調べた版と答えの逐語と会話要約からなるまとまりを表そうと
すると、辺を何本も並べるほかに手が無かった。そのため順序を持つ関係が必要になるたびに
`superseded_by` へ寄り、改訂という意味に前後関係と解決と要約への畳み込みが混ざった。

ヘッダ（`relations`）と項（`relation_members`）に分けると、項を何個でも持てる。
`position` が順序を持つので、共起のような順序の無い関係と、やりとりのような順序のある
関係が同じ器に載る。この形は `wr_records` と `wr_record_items`（030）が既に取っている。

`position` が NULL を取れるのは、順序を持たない関係があるためである。共起（ある反復で
一緒に活性した記録の集まり）に前後は無い。

項の識別子は観測の id にする。`wr_record_items.mi_id` は名前に反して観測の id を持って
おり（拡散想起が `o.id::text = i.mi_id` で結合する）、`superseded_by` と同じ粒度である。

段 1 は器だけを置く。既存の経路からは呼ばないので挙動は変わらない。`superseded_by` を
落として想起の絞りを書き換えるのは段 2 である。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                kind       text NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # 主キーを (関係, 観測, 役割) にする。同じ観測が同じ関係の同じ役割で二度入る形は
        # 改訂にも継起にもやりとりにも共起にも無いので、二重書き込みはここで落ちる。
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_members (
                relation_id bigint NOT NULL REFERENCES relations(id) ON DELETE CASCADE,
                obs_id      text   NOT NULL,
                role        text   NOT NULL,
                position    int,
                PRIMARY KEY (relation_id, obs_id, role)
            )
            """
        )
        # 観測から、それが項として入る関係を引く道（段 2 の想起の絞りが通る）。
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_members_obs ON relation_members(obs_id, role)"
        )
        # 種類で絞る道。
        cur.execute("CREATE INDEX IF NOT EXISTS idx_relations_kind ON relations(kind)")
