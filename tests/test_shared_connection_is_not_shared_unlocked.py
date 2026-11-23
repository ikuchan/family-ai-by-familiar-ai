"""共有の DB 接続を、鍵を取らずに確定しない（2026-11-23）。

11/23 09:40、在席の見回りが見えの普通（`pose_norms`）を鍵の外で確定した。そのとき別のスレッドが鍵を持って記録を
書いていて、セーブポイントが消え（`savepoint "ruri_write" does not exist`）、記録の書き込みが落ちた。落ちた取引は
戻されず、続く保存が連鎖して落ち、親を失った続きの記録 11 件が書けなかった。
"""

from __future__ import annotations

import threading
import time
import uuid
from unittest.mock import patch

from familiar_agent.db import get_db
from familiar_agent.store.pose_norms import PoseNormStore
from tests._db import autocommit_conn


def test_a_pose_norm_save_waits_while_the_shared_lock_is_held():
    conn = get_db().conn()
    store = PoseNormStore(conn)
    done = threading.Event()

    def save():
        store.save("正面", [0.1] * 384, observations=1)
        done.set()

    with get_db().lock:
        t = threading.Thread(target=save)
        t.start()
        time.sleep(0.3)
        assert not done.is_set()  # 鍵を持っているあいだは確定しない
    t.join(timeout=5)
    assert done.is_set()


def test_a_failed_save_leaves_the_connection_usable():
    """外部キー違反で 1 件落ちても、接続は戻され、次の保存は書ける。"""
    from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    bad, ok = mem.save_with_id(
        "親の無い記録 " + uuid.uuid4().hex, direction="発話", parent_id=str(uuid.uuid4())
    )
    assert bad is None and ok is False
    good, ok = mem.save_with_id("ふつうの記録 " + uuid.uuid4().hex, direction="発話")
    assert good is not None and ok is True
    with autocommit_conn() as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM observations WHERE id = %s", (good,))
        assert cur.fetchone() is not None
