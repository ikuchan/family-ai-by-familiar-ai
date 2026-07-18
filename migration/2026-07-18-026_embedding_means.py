"""Add embedding_means and estimate the global mean vector mu（平均中心化 C1）。

計測台帳 §1（確定）：r／在席者相関 p／声紋照合のコサインを取る前に、全埋め込みから
共通成分（平均ベクトル mu）を引いて L2 正規化する。生コサインは異方性（cone 効果）で
無関係でも mean≈0.88 に圧縮され、関連との窓が 0.016 しかない。mu を引くと無関係が
≈0 へ移り窓は約12倍になる。mu は固定保存・低頻度で再推定する。

このマイグレーションは **器と初回推定だけ**：
- `embedding_means` を作る。**scope 付きの複数行**にしてあり、現在は global の1行だけを
  使うが、将来 person 別中心化やクラスタ別平均が来ても行を足すだけで済む。
- `vector` は BYTEA（float32 の生バイト列・既存 obs_embeddings と同じ形式）。**次元非依存**
  なので埋め込みモデルの大型化で次元が変わってもテーブル定義を変えずに済む。`dim` 列で
  取り違えを防ぐ。
- 既存 `obs_embeddings` から global の mu を一度だけ推定して入れる。観測が0件なら入れない。

**中心化の適用（situated 書き込みと recall クエリで mu を引く・既存 situated の一括再計算）は
C2**。この段では誰も mu を使わないので外部挙動は変わらない。

src を import せず自前完結させる（マイグレーションは過去の一度きりの実行を再現する凍結物）。
"""

import numpy as np


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_means (
                id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                scope        text        NOT NULL,
                scope_key    text        NOT NULL DEFAULT '',
                dim          integer     NOT NULL,
                vector       bytea       NOT NULL,
                sample_count integer     NOT NULL,
                updated_at   timestamptz NOT NULL,
                UNIQUE (scope, scope_key)
            )
        """)

        # 既存コーパスから global の mu を初回推定する。
        cur.execute("SELECT vector FROM obs_embeddings")
        rows = cur.fetchall()
        vectors = []
        for row in rows:
            blob = row[0] if not isinstance(row, dict) else row["vector"]
            if blob is None:
                continue
            vec = np.frombuffer(bytes(blob), dtype=np.float32)
            if vec.size:
                vectors.append(vec)
        if not vectors:
            return  # 観測が無ければ mu を作らない（空 DB で壊れない）

        # 次元が揃わない行（モデル移行の残骸）は最頻の次元に合わせて捨てる。
        dims = [v.size for v in vectors]
        dim = max(set(dims), key=dims.count)
        vectors = [v for v in vectors if v.size == dim]

        mu = np.mean(np.stack(vectors), axis=0).astype(np.float32)
        cur.execute(
            "INSERT INTO embedding_means (scope, scope_key, dim, vector, sample_count, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (scope, scope_key) DO UPDATE SET "
            "  dim = EXCLUDED.dim, vector = EXCLUDED.vector, "
            "  sample_count = EXCLUDED.sample_count, updated_at = EXCLUDED.updated_at",
            ("global", "", dim, mu.tobytes(), len(vectors)),
        )
