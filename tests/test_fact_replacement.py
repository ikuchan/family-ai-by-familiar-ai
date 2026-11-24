"""記-ah 段 1：事実の置き換えを、今の関係の表に持つ（`設計方針_事実の移り変わり` §3・2026-11-25 本人の決定）。

記録が語る事実が入れ替わったとき（「サッカーに通っている」→「先月から水泳に変わった」）、古い記録は隠さず、種類
「置き換え」の関係で結ぶ。項は置き換えられた記録（役割「前」）と置き換えた記録（役割「後」）で、どの事実かは項の番号の欄
（`position`）に事実の鍵の文の行の番号を入れる。いつからは、関係の表に足した時刻の列 `valid_at`（082）に置く。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import psycopg2.extras
import pytest

from familiar_agent.db import get_db
from familiar_agent.person_memory_manager import DEFAULT_PERSON_ID
from familiar_agent.store.context import StoreContext
from familiar_agent.store.relations import KIND_REPLACE, RelationStore, hidden


@pytest.fixture
def store() -> RelationStore:
    db = get_db()
    return RelationStore(
        StoreContext(db=db, lock=db.lock, person_id=DEFAULT_PERSON_ID, embedder=None)
    )


def _conn(store):
    return store._ctx.conn()


def test_the_relations_table_has_a_valid_time_column(store):
    with store._ctx.lock:
        with _conn(store).cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT data_type, is_nullable FROM information_schema.columns "
                "WHERE table_name = 'relations' AND column_name = 'valid_at'"
            )
            row = cur.fetchone()
    assert row is not None, "082 が当たっていない"
    assert row["data_type"] == "timestamp with time zone" and row["is_nullable"] == "YES"


def _record_with_key(store, body: str, sentence: str) -> "tuple[str, int]":
    """記録 1 件と、その事実の鍵の文 1 本を書く。（記録の id, 鍵の行の番号）。"""
    obs = str(uuid.uuid4())
    vec = "[" + ",".join(["0"] * 767 + ["1"]) + "]"
    with store._ctx.lock:
        conn = _conn(store)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, emotion) "
                "VALUES (%s, %s, now(), '会話', 'neutral')",
                (obs, body),
            )
            cur.execute(
                "INSERT INTO obs_fact_keys (obs_id, side, sentence, vector) "
                "VALUES (%s, '答え', %s, %s::vector) RETURNING id",
                (obs, sentence, vec),
            )
            r = cur.fetchone()
            key = int(r["id"] if isinstance(r, dict) else r[0])
        conn.commit()
    return obs, key


def test_a_replacement_keeps_the_old_record_visible_and_is_read_with_its_time(store):
    old, old_key = _record_with_key(
        store, "ママ：さぶろうはサッカーに通ってる", "さぶろうの習い事はサッカー"
    )
    new, new_key = _record_with_key(store, "ママ：先月から水泳に変わった", "さぶろうの習い事は水泳")
    since = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rid = store.record_replacement(old, old_key, new, new_key, valid_at=since)
    assert rid is not None
    got = store.replacements_of([old])
    assert [(r.old_key, r.new_key, r.old_sentence, r.new_sentence) for r in got[old]] == [
        (old_key, new_key, "さぶろうの習い事はサッカー", "さぶろうの習い事は水泳")
    ]
    assert got[old][0].valid_at == since
    # 古い記録は隠さない（隠すのは版の管理だけ）
    with store._ctx.lock:
        with _conn(store).cursor() as cur:
            cur.execute(f"SELECT {hidden('o')} AS h FROM observations o WHERE o.id = %s", (old,))
            r = cur.fetchone()
            assert (r["h"] if isinstance(r, dict) else r[0]) is False


def test_records_without_replacements_are_absent(store):
    """**反証の側**：置き換えられていない記録は、返りの辞書に出ない。"""
    lone, _ = _record_with_key(store, "パパ：明日は晴れ", "明日の天気は晴れ")
    assert store.replacements_of([lone]) == {}
    assert KIND_REPLACE == "置き換え"
    assert store.replacements_of([]) == {}


def test_rewriting_a_records_keys_drops_its_replacements(store):
    """鍵の文を書き直すと番号が変わるので、その記録の置き換えも消す（次の判定で付け直す）。"""
    from familiar_agent.store.relations import drop_replacements

    old, old_key = _record_with_key(store, "ママ：サッカーに通ってる", "習い事はサッカー")
    new, new_key = _record_with_key(store, "ママ：水泳に変わった", "習い事は水泳")
    store.record_replacement(
        old, old_key, new, new_key, valid_at=datetime(2026, 10, 1, tzinfo=timezone.utc)
    )
    with store._ctx.lock:
        conn = _conn(store)
        with conn.cursor() as cur:
            drop_replacements(cur, new)
        conn.commit()
    assert store.replacements_of([old]) == {}


def test_the_same_pair_is_written_once(store):
    old, old_key = _record_with_key(store, "ママ：サッカー", "習い事はサッカー")
    new, new_key = _record_with_key(store, "ママ：水泳", "習い事は水泳")
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    first = store.record_replacement(old, old_key, new, new_key, valid_at=at)
    again = store.record_replacement(old, old_key, new, new_key, valid_at=at)
    assert first == again and len(store.replacements_of([old])[old]) == 1


def test_a_key_of_another_record_is_not_written(store):
    """**反証の側**：鍵の文がその記録のものでなければ書かない（判定が番号を取り違えたとき）。"""
    old, old_key = _record_with_key(store, "ママ：サッカー", "習い事はサッカー")
    new, new_key = _record_with_key(store, "ママ：水泳", "習い事は水泳")
    other, other_key = _record_with_key(store, "パパ：晴れ", "天気は晴れ")
    at = datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert store.record_replacement(old, other_key, new, new_key, valid_at=at) is None
    assert store.record_replacement("", old_key, new, new_key, valid_at=at) is None
    assert store.replacements_of([old]) == {}
