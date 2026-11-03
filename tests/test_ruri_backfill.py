"""記-y A7：今ある記録の Ruri の数値を埋める道具（`store/ruri_backfill.fill_missing`）。

表に行が無い記録だけを埋めるので、何度回してもよい。隠された記録も埋める（本人の決定イ）。
数値にできなかった束は飛ばして数え、0 は書かない。
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from familiar_agent.db import get_db
from familiar_agent.store.embedding import _RuriModel
from familiar_agent.store.ruri_backfill import fill_missing
from familiar_agent.tools.memory import ObservationMemory, _EmbeddingModel

_FIXED = [1.0] + [0.0] * 1023


def _count(ids: list[str]) -> int:
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS c FROM obs_ruri_embeddings WHERE obs_id = ANY(%s)", (ids,)
            )
            row = cur.fetchone()
        conn.rollback()
    return int(row["c"] if isinstance(row, dict) else row[0])


def _save_without_ruri(n: int) -> list[str]:
    with patch.object(_EmbeddingModel, "pre_warm"):
        mem = ObservationMemory()
    ids = []
    with (
        patch.object(_EmbeddingModel, "encode_document", return_value=[_FIXED]),
        patch.object(_RuriModel, "encode_document", return_value=None),
    ):
        for _ in range(n):
            obs_id, _ = mem.save_with_id("backfill " + uuid.uuid4().hex, materialize_now=True)
            ids.append(obs_id)
    return ids


def _fill(**kw):
    db = get_db()
    with db.lock:
        return fill_missing(db.conn(), _RuriModel(), batch_size=2, **kw)


def test_fills_only_missing_rows_and_is_repeatable() -> None:
    ids = _save_without_ruri(3)
    assert _count(ids) == 0
    written, failed = _fill()
    assert (written, failed) == (3, 0)
    assert _count(ids) == 3
    assert _fill() == (0, 0)


def test_failed_batches_are_skipped_not_zero_filled() -> None:
    ids = _save_without_ruri(3)
    with patch.object(_RuriModel, "encode_document", return_value=None):
        written, failed = _fill()
    assert (written, failed) == (0, 3)
    assert _count(ids) == 0


def test_progress_is_reported() -> None:
    _save_without_ruri(3)
    seen: list[tuple[int, int]] = []
    _fill(report=lambda done, total: seen.append((done, total)))
    assert seen and seen[-1] == (3, 3)


def test_app_running_check_reads_pgrep() -> None:
    """パジュが動いたまま回すと、Ruri の読み込みが本体と重なってメモリが足りなくなりうる。止まっているかを見る。"""
    import subprocess

    from familiar_agent.store.ruri_backfill import app_is_running

    found = subprocess.CompletedProcess(args=[], returncode=0, stdout="123 python familiar --gui\n")
    none = subprocess.CompletedProcess(args=[], returncode=1, stdout="")
    with patch("subprocess.run", return_value=found):
        assert app_is_running() is True
    with patch("subprocess.run", return_value=none):
        assert app_is_running() is False


def test_one_bad_record_does_not_block_its_batch() -> None:
    """束で失敗したら 1 件ずつやり直す。いつも数値にできない 1 件のせいで、同じ束の残りが埋まらないままにしない。"""
    ids = _save_without_ruri(3)
    db = get_db()
    with db.lock:
        conn = db.conn()
        with conn.cursor() as cur:
            cur.execute("SELECT id, content FROM observations WHERE id = ANY(%s)", (ids,))
            rows = list(cur.fetchall())
        conn.rollback()
    first = rows[0]
    bad_id, bad_text = (first["id"], first["content"]) if isinstance(first, dict) else first
    real = _RuriModel.encode_document

    def one_bad(self, texts):
        # その本文を含む呼び出しはいつも失敗する（束でも 1 件でも）。
        return None if bad_text in texts else real(self, texts)

    with patch.object(_RuriModel, "encode_document", one_bad):
        written, failed = _fill()
    assert (written, failed) == (2, 1)
    assert _count([bad_id]) == 0
    assert _count([i for i in ids if i != bad_id]) == 2


def test_stops_early_when_the_model_cannot_load() -> None:
    """モデルが読み込めないときは、1 件ずつやり直さずに打ち切る（警告を件数ぶん出さない）。"""
    ids = _save_without_ruri(3)
    with patch.object(_RuriModel, "_load", side_effect=RuntimeError("no weights")):
        written, failed = _fill()
    assert (written, failed) == (0, 3)
    assert _count(ids) == 0


def test_run_does_not_deadlock_on_settings() -> None:
    """設定値の控えが空でも固まらない（2026-11-03 本番で固まった）。

    Ruri の入れ物は作るときに設定値を読み、設定値の読みは DB の錠を取る。錠を持ったまま入れ物を作ると、
    自分で自分を待つ。テストでは控えが先に入っていて通ってしまったので、控えを空にして確かめる。
    """
    import threading

    from familiar_agent import config_overrides
    from familiar_agent.store.ruri_backfill import run

    ids = _save_without_ruri(2)
    config_overrides._cache = None
    out: list[str] = []
    result: dict = {}
    t = threading.Thread(
        target=lambda: result.update(code=run(get_db(), out=out.append)), daemon=True
    )
    t.start()
    t.join(timeout=20)
    assert not t.is_alive(), "固まった：" + " / ".join(out)
    assert result["code"] == 0
    assert _count(ids) == 2
    assert out[0].startswith("Ruri を読み込んだ") and out[1].startswith("はじめ")
    assert out[-1].startswith("おわり")
