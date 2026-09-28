"""`scene_entities`・`scene_events` 表を落とす（環-ab の R-5・2026-09-28）。

場面の実体を追う `SceneTracker`（`scene.py`）の表だった。`agent.py` が作るだけで誰も読まず、書く入口
（`update()`）の呼び手も無かった。コードは環-ab の R-5a で外したので、読む者も書く者も居ない。
写真の読み取り（`scene.read_photo`）は表を使わず、見えたものを O の記録に残す（出-au 段 5-7a）。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS scene_events")
        cur.execute("DROP TABLE IF EXISTS scene_entities")
