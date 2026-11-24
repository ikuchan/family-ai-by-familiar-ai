"""鍵の文の表（`obs_fact_keys`）と抜き済みの印（`obs_fact_runs`）の読み書き（記-ad・080）。

抜き待ちは、隠されていない本文のある記録で、想起の視点の面があり、印の無いもの。同じ本文の記録は
1 つの束にまとめ、LLM には 1 回だけ聞く（記録の重複は多く、3 万行に対して本文は 1.8 万種）。
鍵の文の点は呼ぶ側（`ObservationMemory.write_fact_keys`）が Ruri で作って渡す。読み出し（鍵の軸）は
`ObservationStore.by_fact_keys` と `fact_key_cosines`。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

from psycopg2.extras import execute_values

from ..db import vec_to_sql
from .relations import drop_replacements, not_hidden

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactSource:
    """抜き待ちの本文 1 種と、その本文を持つ記録の id。"""

    content: str
    obs_ids: "tuple[str, ...]"


def pending(ctx, limit: int) -> "list[FactSource]":
    """抜き待ちの本文を、新しい記録の順に limit 種まで。

    視点は問わない（面が 1 つでもあればよい）。想起は思い出す人の視点で鍵の軸を引くので、家族の面にしか無い記録にも
    鍵の文が要る。
    """
    live = not_hidden("o")
    with ctx.lock:
        conn = ctx.conn()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT o.content, array_agg(o.id ORDER BY o.id) AS ids, max(o.timestamp) AS ts
                FROM observations o
                WHERE {live}
                  AND COALESCE(o.content, '') <> ''
                  AND o.direction NOT IN ('内省', '聞こえた')  -- 聞こえた話（core/heard）はパジュに言ったことではない
                  AND EXISTS (SELECT 1 FROM situated_memories s WHERE s.obs_id = o.id)
                  AND NOT EXISTS (SELECT 1 FROM obs_fact_runs f WHERE f.obs_id = o.id)
                GROUP BY o.content
                ORDER BY ts DESC
                LIMIT %(n)s
                """,
                {"n": limit},
            )
            rows = cur.fetchall()
        conn.rollback()
    return [FactSource(r["content"], tuple(r["ids"])) for r in rows]


class KeyRow(NamedTuple):
    """1 つの記録に書くもの。`content` は抜いたときの本文（書く前に、いまの本文と同じかを確かめる）。"""

    obs_id: str
    content: str
    n_facts: int
    keys: "list[tuple[str, str, list[float]]]"  # (向き, 鍵の文, 点)


def write(ctx, rows: "list[KeyRow]", model: str) -> "tuple[set[str], int]":
    """記録ごとに鍵の文を書き、印を付ける。(書いた記録の id, 書いた行の数) を返す。

    **本文が抜いたときと変わっていた記録は書かない**（印も付けない）。抜いているあいだに本文へ書き足されると
    （`append_and_reembed`）、古い本文の鍵の文で印を付けてしまい、足した本文が二度と抜かれないため。
    同じ記録の古い鍵の文は消してから書く。1 回の呼び出しを 1 つのトランザクションにする。
    """
    written = 0
    done: "set[str]" = set()
    with ctx.lock:
        conn = ctx.conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    # 行の錠を取る：確かめてから書くまでのあいだに、別の接続が本文を書き換えないように。
                    "SELECT id, content FROM observations WHERE id = ANY(%s) FOR UPDATE",
                    ([r.obs_id for r in rows],),
                )
                now_content = {r["id"]: r["content"] for r in cur.fetchall()}
                for row in rows:
                    if now_content.get(row.obs_id) != row.content:
                        logger.debug("本文が変わったので鍵の文を書かない：%.8s", row.obs_id)
                        continue
                    # 置き換えの関係（記-ah）は鍵の文の行の番号を指すので、書き直す鍵を指す置き換えも消す
                    drop_replacements(cur, row.obs_id)
                    cur.execute("DELETE FROM obs_fact_keys WHERE obs_id = %s", (row.obs_id,))
                    if row.keys:
                        execute_values(
                            cur,
                            "INSERT INTO obs_fact_keys (obs_id, side, sentence, vector) VALUES %s",
                            [
                                (row.obs_id, side, sentence, vec_to_sql(v))
                                for side, sentence, v in row.keys
                            ],
                            template="(%s, %s, %s, %s::vector)",
                        )
                        written += len(row.keys)
                    cur.execute(
                        "INSERT INTO obs_fact_runs (obs_id, model, n_facts) VALUES (%s, %s, %s) "
                        "ON CONFLICT (obs_id) DO UPDATE SET model = EXCLUDED.model, "
                        "n_facts = EXCLUDED.n_facts, extracted_at = now()",
                        (row.obs_id, model, row.n_facts),
                    )
                    done.add(row.obs_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return done, written
