"""旧 15 欲求の状態 `agent_state.desires` を消す（環-d・2026-09-15）。

書き手 `DesireSystem`（`desires.py`）はこの日に撤去した。2026-07-26 から更新が無く、読む側も
無かった。欲求は 5 軸（`drive_register`・`core/drive_dynamics`・`agent_state.drive5`）が担う。
"""

from __future__ import annotations


def upgrade(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM agent_state WHERE state_key = 'desires'")
