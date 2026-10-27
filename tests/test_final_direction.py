"""最終版を向き「最終版」で持ち、古い行の向きを種類から埋める（記-u 段 1・2026-10-27・本人の決定）。

記録の種類（`kind`）の列をなくし、「何の記録か」を向きだけで持つ（記-u・案ウ）。その前に、種類にしか無い情報を
向きへ移す。向き `unknown` の古い行 180 件は種類から向きを埋め（utterance → 発話・scene と witnessed → 観察・
conversation → 会話）、向き `求め` で「〔最終版」で始まる 411 件は向き「最終版」にする。窓の最終版も同じ向き。
W の見出しは「{名}のやりとりのまとめ」（機器や情動から始まった求めにも、窓の会話の全部にも合う・本人）。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

from familiar_agent.core import final_version as fv

_DB_URL = os.environ["DATABASE_URL"]
_PATH = Path(__file__).resolve().parents[1] / "migration" / "2026-10-27-077_directions_from_kind.py"


def _insert(cur, content: str, direction: str, kind: str) -> str:
    oid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
        "VALUES (%s, %s, %s, %s, %s, 'neutral')",
        (oid, content, datetime.now(timezone.utc), direction, kind),
    )
    return oid


def _upgrade(conn) -> None:
    from familiar_agent.db_migrations import _load_module

    _load_module(_PATH).upgrade(conn)


def test_the_migration_moves_kind_into_direction():
    """1 つのトランザクションの中で植えて当て、見て、巻き戻す。

    077 の UPDATE は植えた行に限らないので、確定させるとほかの試験の行（並行に走る継起の候補の試験の、印で始まる
    `求め` の行など）まで書き換える。巻き戻せば、ほかの試験からは何も起きなかったことになる。
    """
    conn = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with conn.cursor() as cur:
            planted = {
                _insert(cur, "077 発話", "unknown", "utterance"): "発話",
                _insert(cur, "077 場面", "unknown", "scene"): "観察",
                _insert(cur, "077 目撃", "unknown", "witnessed"): "観察",
                _insert(cur, "077 会話", "unknown", "conversation"): "会話",
                _insert(
                    cur, f"{fv.MARK}077〕「a」と聞かれ、「b」と答えた", "求め", "observation"
                ): "最終版",
                # **反証の側**：印の無い求め・文の途中に印がある求め・向きのある行・対応の無い種類は動かさない。
                _insert(cur, "「a」と聞かれ、1番：…", "求め", "observation"): "求め",
                _insert(cur, f"版の中で引いた {fv.MARK}077〕…", "求め", "observation"): "求め",
                _insert(cur, f"{fv.MARK}077〕発話に印", "発話", "observation"): "発話",
                _insert(cur, "077 不明のまま", "unknown", "observation"): "unknown",
            }
        _upgrade(conn)
        _upgrade(conn)  # 冪等：2 度目は当たる行が無い
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, direction FROM observations WHERE id = ANY(%s)", (list(planted),)
            )
            got = {r["id"]: r["direction"] for r in cur.fetchall()}
        assert got == planted
    finally:
        conn.rollback()
        conn.close()


def test_the_final_is_headed_as_an_exchange_summary():
    from familiar_agent.tools.memory import subject_line

    assert subject_line("最終版", "パジュ") == "パジュのやりとりのまとめ"
    # **反証の側**：求め（起点と途中の版）の見出しは今のまま。
    assert subject_line("求め", "パジュ") == "パジュが調べていたこと"
