"""顔と声の特徴の表（知-ae 段 1・2026-10-02・`設計方針_声で話者を見分ける` v0.1）。

顔（ArcFace）と声（ECAPA-TDNN）の特徴を 1 つの表に持つ（本人の決定ア）。以前は pickle ファイル
（`~/.familiar_ai/face_embeddings.pkl`・`voice_embeddings.pkl`）で、開発ルール「保存は PostgreSQL のみ」に
反していた。どちらのファイルも作られていなかったので、移すデータは無い。

- `kind`：`face`・`voice`。`origin`：`registered`（登録・日をまたいで残る）・`today`（今日の声・`day` の日だけ）。
- 特徴の次元は顔と声で違う（512・192）ので `real[]` に持つ。照らすのは人ごとの重心で、DB では比べない。
- 人を消せば特徴も消える（`ON DELETE CASCADE`）。冪等。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_embeddings (
                id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                person_id  text NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                kind       text NOT NULL CHECK (kind IN ('face', 'voice')),
                origin     text NOT NULL CHECK (origin IN ('registered', 'today')),
                day        date,
                vec        real[] NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_recognition_embeddings_person "
            "ON recognition_embeddings(person_id, kind, origin)"
        )
