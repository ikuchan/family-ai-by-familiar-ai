"""`relationship_state` 表を落とす（環-d・2026-09-15）。

書き手 `RelationshipTracker`（`relationship.py`）は毎ターン `record_conversation()` で書いていたが、
`trust`／`intimacy` を読むのは呼び手の無い `_select_addressee` だけだった。設計の「関係の内容を O の
MI へ移管」（案A・2026-07-29）はやめて撤去する（2026-09-15 決定・バックアップあり）。関係は O の
関係の面（`situated_memories`）と `人物` のまとめ（層 1 ①）が担う。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS relationship_state")
