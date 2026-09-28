"""`pending_speech` 表を落とす（環-ab の A・2026-09-28）。

保留した発話を人が来たら配る仕組みの表だった。配る口と宛先の条件は出-as 段 9b（2026-09-26）で外し、止められた発話は
独り言として O に残すようになった。残っていた書く口（主LLM に出していない記憶の道具 `note_to_share`）と記憶の箱
（`PendingSpeechStore`）・設定（`PendingSpeechConfig`）は環-ab の A-1・A-2 で外したので、読む者も書く者も居ない。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS pending_speech")
