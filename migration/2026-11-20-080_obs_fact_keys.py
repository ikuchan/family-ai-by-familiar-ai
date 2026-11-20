"""鍵の文の表 `obs_fact_keys` と、抜き済みの印 `obs_fact_runs` を足す（記-ad・2026-11-20・本人の承認）。

記録の本文から軽量LLM で事実（主語・関係・答え）を抜き、問いに近い形の「鍵の文」に書き直して持つ
（論文 AtlasKV の KG2KV の考え方）。事実 1 つから 2 行を作る：
- 向き「答え」：答えを隠した文（〈主語〉の〈関係〉）。
- 向き「主語」：主語を隠した文（〈関係〉が「〈答え〉」であるもの）。
点は Ruri（768 次元・平均は引かない）。想起はこの点と問いの近さで記録を集める（鍵の軸）。

抜き済みの印は記録ごとに 1 行。事実が 0 個だった記録にも付け、もう一度 LLM を呼ばない。
どちらの表も記録が消えれば一緒に消える。

点の索引は付けない。行は数万までで、全部を比べても 1 回数十 ms に収まり、索引の近似で正解を
取りこぼす（`計測_設定値根拠台帳` §13 と同じ取りこぼし）ことを避ける。

冪等（`IF NOT EXISTS`）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS obs_fact_keys (
                id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                obs_id    TEXT NOT NULL REFERENCES observations(id) ON DELETE CASCADE,
                side      TEXT NOT NULL CHECK (side IN ('答え', '主語')),
                sentence  TEXT NOT NULL,
                vector    vector(768) NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_fact_keys_obs ON obs_fact_keys (obs_id)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS obs_fact_runs (
                obs_id        TEXT PRIMARY KEY REFERENCES observations(id) ON DELETE CASCADE,
                model         TEXT NOT NULL,
                n_facts       INTEGER NOT NULL,
                extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
