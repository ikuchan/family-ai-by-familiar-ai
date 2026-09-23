"""宛先の条件——それを言うのに誰が要るか（出-ap・2026-09-23）。

実機 15:49、話者が `unconfirmed` のまま「こんにちは。おかえりなさい。**どなたでしょうか？**
あ、それから…**フーコック旅行は送迎だけ頼むことにして、連休のキャンプは全部キャンセルに**
なったのですね」と、家の予定を全部話した。**同じ発話の中で相手が誰かを聞いている。**

在席の注記（「名前で呼ばない」）は誰かを**呼ぶ**ことだけを止めており、**何を話してよいか**は
誰も見ていなかった。話したいことは、それを言うのに**誰が要るか**で 4 段に分かれる（本人）。

| 段 | 条件 | 例 |
|---|---|---|
| 0 | 誰もいなくても | 独り言 |
| 1 | 誰かいたら | タイマーが鳴っている |
| 2 | 家族がいたら | 家の記録・予定・メモ |
| 3 | 特定の誰かがいたら | その人への申し送り |

今回は**きっかけの札で一律に決める**（本人の決定）——`[メモ]` は段 2、ほかは段 1。中身ごとに
主LLM へ決めさせる案は、測る必要があるので見送った。
"""

from __future__ import annotations

import pytest

from familiar_agent.core.audience import ANYONE, FAMILY, level_of, meets

_KNOWN = [{"person_id": "p1", "name": "パパ", "confidence": 1.0, "is_speaker": True}]
_UNKNOWN = [{"person_id": None, "name": "不明", "confidence": 0.5, "is_speaker": False}]


# ── きっかけの札から段を決める ────────────────────────────────────────────


def test_a_memo_needs_family():
    assert level_of("[メモ] パジュへのメモが変わった。新しく書かれたこと：…") == FAMILY


def test_a_timer_only_needs_someone():
    assert level_of("[タイマー] タイマー：「パスタ」の時間") == ANYONE


def test_arriving_only_needs_someone():
    assert level_of("[入室] たいき が来た") == ANYONE


def test_no_label_means_anyone():
    """札が無いものは既定の段（誰かいたら）。**厳しい側へ倒さない**——届かなくなる。"""
    assert level_of("") == ANYONE
    assert level_of("ただの文") == ANYONE


# ── いまの在席が段を満たすか ──────────────────────────────────────────────


def test_anyone_is_met_by_an_unknown_person():
    assert meets(ANYONE, _UNKNOWN) is True


def test_anyone_is_not_met_by_an_empty_room():
    assert meets(ANYONE, []) is False


def test_family_needs_someone_we_can_name():
    assert meets(FAMILY, _KNOWN) is True


def test_family_is_not_met_by_an_unknown_person():
    """名前の分からない在席者（出-ae-は の札）では足りない。家族と確かめられていない。"""
    assert meets(FAMILY, _UNKNOWN) is False


def test_family_is_met_when_a_known_person_is_among_unknowns():
    assert meets(FAMILY, _UNKNOWN + _KNOWN) is True


# ── 保留が段を持つ ────────────────────────────────────────────────────────


@pytest.fixture()
def store():
    """マイグレーションを当ててから器を 1 つ（`test_pending_speech.py` と同じ作り）。

    **テストごとに閉じる。** `conftest` は各テストの前に `pending_speech` を TRUNCATE する
    ので、器の接続を開いたまま次のテストへ渡すと、そこで待たされて止まる。
    """
    import os

    import psycopg2

    from familiar_agent.db_migrations import apply_migrations, default_migration_dir
    from familiar_agent.tools.pending_speech_store import PendingSpeechStore

    url = os.environ["DATABASE_URL"]
    c = psycopg2.connect(url)
    apply_migrations(c, default_migration_dir())
    c.commit()
    c.close()
    s = PendingSpeechStore(database_url=url)
    yield s
    if s._conn is not None and not s._conn.closed:
        s._conn.rollback()
        s._conn.close()


def _obs(body: str) -> str:
    import os
    import uuid

    import psycopg2

    obs_id = str(uuid.uuid4())
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO observations (id, content, timestamp, direction, kind, emotion) "
                "VALUES (%s,%s,now(),%s,%s,%s)",
                (obs_id, body, "保留", "observation", "neutral"),
            )
        conn.commit()
    finally:
        conn.close()
    return obs_id


def test_a_held_line_keeps_its_audience(store):
    s = store
    s.add(_obs("メモの話"), None, audience=FAMILY)
    rows = [r for r in s.list_active() if r.get("content") == "メモの話"]
    assert rows and rows[0]["audience"] == FAMILY


def test_the_default_audience_is_anyone(store):
    s = store
    s.add(_obs("タイマーの話"), None)
    rows = [r for r in s.list_active() if r.get("content") == "タイマーの話"]
    assert rows and rows[0]["audience"] == ANYONE


# ── 配るとき、段を満たさないものは箱に残す ───────────────────────────────


class _FakeStore:
    def __init__(self, rows):
        self._rows = list(rows)
        self.deleted: list = []

    def list_active(self):
        return list(self._rows)

    def freshness_score(self, row, now, cfg):
        return 1.0

    def is_expired(self, row, score, cfg):
        return False

    def delete(self, pid):
        self.deleted.append(pid)
        self._rows = [r for r in self._rows if r["id"] != pid]


def _row(pid: str, content: str, audience: int) -> dict:
    import datetime

    return {
        "id": pid,
        "observation_id": f"o-{pid}",
        "content": content,
        "audience": audience,
        "created_at": datetime.datetime(2026, 9, 21, 15, 48, tzinfo=datetime.timezone.utc),
    }


def _loop(store, presence_rows):
    from unittest.mock import MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    agent = MagicMock()
    agent._pending_store = store
    agent._pmm.presence_status = MagicMock(return_value=presence_rows)
    ip._agent = agent
    ip._req = MagicMock()
    ip._req.speech_to_deliver = []
    return ip


def _released(ip) -> str:
    return "\n".join(ip._req.speech_to_deliver)


def test_a_family_line_waits_while_nobody_is_named():
    import asyncio

    store = _FakeStore(
        [_row("a", "タイマーが鳴っていますよ", ANYONE), _row("b", "メモ更新、覚えたよ", FAMILY)]
    )
    ip = _loop(store, _UNKNOWN)
    asyncio.run(ip._release_pending_speech())
    got = _released(ip)
    assert "タイマー" in got, "誰かいれば言えるものは配る"
    assert "メモ" not in got, "家族と確かめられていないので配らない"
    assert store.deleted == ["a"], "配らなかったものは箱に残す"


def test_a_family_line_goes_out_once_someone_is_named():
    import asyncio

    store = _FakeStore([_row("b", "メモ更新、覚えたよ", FAMILY)])
    ip = _loop(store, _KNOWN)
    asyncio.run(ip._release_pending_speech())
    assert "メモ" in _released(ip)
    assert store.deleted == ["b"]


def test_anyone_level_goes_out_even_when_no_face_matched():
    """段 1 は、顔が照合できていなくても配る。

    この口は**在席がゼロから立ち上がった瞬間**にしか呼ばれないので、段 1 はその時点で
    満たされている。`presence_status()` は顔が照合できた人しか載らないため、ここで段 1 まで
    見ると、誰か分からない相手のときにタイマーの知らせまで止まる。
    """
    import asyncio

    store = _FakeStore([_row("a", "タイマーが鳴っていますよ", ANYONE)])
    ip = _loop(store, [])
    asyncio.run(ip._release_pending_speech())
    assert "タイマー" in _released(ip)
    assert store.deleted == ["a"]


def test_a_family_line_still_waits_when_no_face_matched():
    import asyncio

    store = _FakeStore([_row("b", "メモ更新、覚えたよ", FAMILY)])
    ip = _loop(store, [])
    asyncio.run(ip._release_pending_speech())
    assert _released(ip) == ""
    assert store.deleted == []


def test_a_row_without_the_column_is_treated_as_anyone():
    """既存の 317 件は段を持たない。**厳しい側へ倒さない**（届かなくなる）。"""
    import asyncio

    row = _row("a", "むかしの保留", ANYONE)
    row.pop("audience")
    ip = _loop(_FakeStore([row]), _UNKNOWN)
    asyncio.run(ip._release_pending_speech())
    assert "むかしの保留" in _released(ip)


def test_a_memo_is_held_as_family_only():
    """保留へ積むとき、きっかけの札から段を付ける。"""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="o-1")
    agent._observation_perspective = MagicMock(return_value={})
    agent._pending_store.add = MagicMock(return_value="p-1")
    ip._agent = agent
    ip._req = MagicMock()
    ip._req.request_text = "[メモ] パジュへのメモが変わった。新しく書かれたこと：…"
    asyncio.run(ip._hold_speech("メモ更新、覚えたよ", "聞く相手が居ない"))
    assert agent._pending_store.add.call_args.kwargs.get("audience") == FAMILY


def test_a_timer_is_held_for_anyone():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    agent = MagicMock()
    agent._oif.write = AsyncMock(return_value="o-2")
    agent._observation_perspective = MagicMock(return_value={})
    agent._pending_store.add = MagicMock(return_value="p-2")
    ip._agent = agent
    ip._req = MagicMock()
    ip._req.request_text = "[タイマー] タイマー：「パスタ」の時間"
    asyncio.run(ip._hold_speech("タイマーが鳴っていますよ", "聞く相手が居ない"))
    assert agent._pending_store.add.call_args.kwargs.get("audience") == ANYONE
