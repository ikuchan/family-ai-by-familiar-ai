"""今ある記録の Ruri の数値を埋める（記-y・2026-11-03）。

079 で表（`obs_ruri_embeddings`）を作っただけでは、今ある記録には Ruri の数値が無い。ここで本文を
Ruri で数値にして埋める。**表に行が無い記録だけ**を埋めるので、途中で止まっても、何度回してもよい。
隠された記録も埋める（本人の決定イ：もし戻っても抜けないように）。

数値にできなかった束は飛ばして数える。0 は書かない（0 は「近さ 0」と区別できない値として残る）。
飛ばした記録は、もう一度回せば埋まる。本番で回すのはパジュを止めている間で、回すのは本人
（`scripts/backfill_ruri.py`）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..db import vec_to_sql

logger = logging.getLogger(__name__)


def app_is_running() -> bool:
    """パジュ（`familiar --gui`）が動いているか。動いたまま埋めると、Ruri の読み込みが本体と重なり、
    主記憶 14 GB の機体ではメモリが足りなくなりうる（大きな処理を並行で走らせて落ちた前例がある）。"""
    import subprocess

    out = subprocess.run(
        ["pgrep", "-af", "familiar --gui"], capture_output=True, text=True, check=False
    )
    return out.returncode == 0 and bool(out.stdout.strip())


def fill_missing(
    conn: Any,
    ruri: Any,
    *,
    batch_size: int = 16,
    report: "Callable[[int, int], None] | None" = None,
) -> tuple[int, int]:
    """行の無い記録を Ruri で数値にして書く。(書いた件数, 飛ばした件数) を返す。

    本文の短い順に束ねる（長さの近い文を一度に数値にすると、詰め物が少なく速い）。束ごとに確定する。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT o.id, o.content FROM observations o "
            "WHERE COALESCE(o.content, '') <> '' "
            "  AND NOT EXISTS (SELECT 1 FROM obs_ruri_embeddings r WHERE r.obs_id = o.id) "
            "ORDER BY length(o.content), o.id"
        )
        rows = [(r["id"], r["content"]) if isinstance(r, dict) else (r[0], r[1]) for r in cur]
    total = len(rows)
    written = failed = 0
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        vecs = ruri.encode_document([content for _, content in batch])
        if vecs:
            done = list(zip(batch, vecs))
        elif getattr(ruri, "failed", False):
            # モデルが読み込めないなら、1 件ずつやり直しても同じ。残りを数えて打ち切る（原因はモデルの読み込みの 1 つ）。
            failed += total - start
            logger.error("Ruri のモデルが読み込めないので打ち切った（残り %d 件）", total - start)
            if report is not None:
                report(written + failed, total)
            break
        else:
            # 束で失敗したら 1 件ずつやり直す。数にならない値が 1 件でも出ると束ごと None になるので、
            # そのままだと同じ束の残りが何度回しても埋まらない（束の並びは毎回同じ）。
            done = []
            for item in batch:
                one = ruri.encode_document([item[1]])
                if one:
                    done.append((item, one[0]))
                else:
                    failed += 1
                    logger.warning("Ruri で数値にできなかったので飛ばした：%.8s", item[0])
        if done:
            with conn.cursor() as cur:
                for (obs_id, _), vec in done:
                    cur.execute(
                        "INSERT INTO obs_ruri_embeddings (obs_id, vector) VALUES (%s, %s::vector) "
                        "ON CONFLICT (obs_id) DO NOTHING",
                        (obs_id, vec_to_sql(vec)),
                    )
            conn.commit()
            written += len(done)
        if report is not None:
            report(written + failed, total)
    return written, failed


def _counts(conn: Any) -> tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) AS c FROM observations")
        obs = cur.fetchone()
        cur.execute("SELECT count(*) AS c FROM obs_ruri_embeddings")
        ruri = cur.fetchone()
    conn.rollback()

    def pick(r: Any) -> int:
        return int(r["c"] if isinstance(r, dict) else r[0])

    return pick(obs), pick(ruri)


def run(db: Any, *, batch_size: int = 16, out: "Callable[[str], None]" = print) -> int:
    """道具の本体（`scripts/backfill_ruri.py` から呼ぶ）。終了コードを返す（0＝全部埋めた）。"""
    import time

    from .embedding import _RuriModel

    started = time.monotonic()
    # **Ruri は錠の外で作り、読み込んでおく。** 作るときに設定値を読み、設定値の読みは DB の錠を取る。
    # 錠を持ったまま作ると自分で自分を待って固まる（2026-11-03 本番で固まった）。
    ruri = _RuriModel()
    if ruri.ensure() is None:
        out("Ruri のモデルを読み込めなかったので、何もしないで終わる")
        return 1
    out(f"Ruri を読み込んだ（{time.monotonic() - started:.0f} 秒）")
    with db.lock:
        conn = db.conn()
        obs, ruri_rows = _counts(conn)
        out(f"はじめ：記録 {obs} 件・Ruri の行 {ruri_rows} 件")

        def report(done: int, total: int) -> None:
            if done == total or done % (batch_size * 50) == 0:
                out(f"  {done}/{total} 件（{time.monotonic() - started:.0f} 秒）")

        written, failed = fill_missing(conn, ruri, batch_size=batch_size, report=report)
        obs, ruri_rows = _counts(conn)
    out(f"書いた {written} 件・飛ばした {failed} 件・{time.monotonic() - started:.0f} 秒")
    out(f"おわり：記録 {obs} 件・Ruri の行 {ruri_rows} 件")
    return 0 if failed == 0 else 1
