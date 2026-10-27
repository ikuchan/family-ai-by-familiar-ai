"""種類（`kind`）にしか無い情報を向き（`direction`）へ移す（記-u 段 1・2026-10-27・本人の決定）。

記録の種類の列をなくし、「何の記録か」を向きだけで持つ（記-u・案ウ。`MI データモデル` 付録A の「kind 廃止（案A）」を
実装する）。列を落とすのは 078 で、ここはその前に、向きから分からない情報を向きへ移す。

- **向き `unknown` の古い行**（2026-10-27 の本番で 180 件）：種類から向きを埋める。utterance（157 件）→ 発話、
  scene（19 件）と witnessed（3 件）→ 観察（witnessed に合う向きは無い）、conversation（1 件）→ 会話。対応の無い
  種類は `unknown` のまま。
- **最終版**（同 411 件・10/14 から）：向き `求め` で文が「〔最終版 」で始まる行を、向き「最終版」にする。最終版は
  起点や途中の版と同じ `求め` で、文の頭の印でしか見分けられなかった。窓の最終版も同じ印なので同じ向きになる。

冪等（2 度目は当たる行が無い）。
"""

from __future__ import annotations

_FROM_KIND = (
    ("utterance", "発話"),
    ("scene", "観察"),
    ("witnessed", "観察"),
    ("conversation", "会話"),
)


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        for kind, direction in _FROM_KIND:
            cur.execute(
                "UPDATE observations SET direction = %s WHERE direction = 'unknown' AND kind = %s",
                (direction, kind),
            )
        cur.execute(
            "UPDATE observations SET direction = '最終版' WHERE direction = '求め' AND content LIKE %s",
            ("〔最終版 %",),
        )
