"""PAD の P/Pn/Dom を未測定でありうる形にする（`NOT NULL` を外す）。

**感情軸の母集合は、半分が同じ一点に潰れていた。** 2026-08-21 のダンプでは 6433 行のうち
2941 行（45.7%）が PAD 全部 0.5 で、`emotion_vec` がゼロベクトルだった。内訳はほぼ知覚の
観察（`observation` 2867）である。L2 距離ではこの 2941 行が原点に重なるので、感情軸が
候補を並べ替えられない。

**原因は「埋める」ことだった。** 評価器（軽量LLM）は値踏みゲート（`A_GATE`＝0.25）未満だと
呼ばれず、そのとき P/Pn/Dom を気分の値で埋めていた。気分が平静なら 0.5 が3つ入る。埋めた値は
測ったものではないので、感情軸の母集合に混ぜてはいけない。0.5 が入っていると「測ったのか
埋めたのか」を後から見分けられず、REST 内省が埋め直す余地も消える。

**A（高ぶり）は機械値なので常に入る**（内容の新規性 novelty から作る・`感情ループ全体像`）。
だからここでは `emotion_a` を触らない。未測定になるのは P/Pn/Dom の3つだけである。

**既にある行は NULL へ戻さない。** 埋めた値と測った値を後から見分ける手段が無いためである。
既存行の扱いは REST 内省（記-a）が読み直すときの課題として残す。

この id は 2026-08-21 のダンプに残っていたものと同じである。本番の `schema_migrations` には
既に記録されているので、ここでは流れない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for column in ("emotion_p", "emotion_pn", "emotion_dom"):
            cur.execute(f"ALTER TABLE observations ALTER COLUMN {column} DROP NOT NULL")
