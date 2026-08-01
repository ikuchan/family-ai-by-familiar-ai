"""発話の記録への追記と、親子で畳む仕組みの撤去（設計方針『求めの版チェーン』V4）。

**発話の記録は鎖の外に置く。** 人が言ったことの記録であって、求めの状態ではない。ただし
検索を始めたことは、その記録からも辿れたほうがよい。**一度だけ**「検索を始めた」を追記する。
何を調べているかは版の側が持つので、発話の記録が持つ必要はない（二重に持つと、どちらが正か
が曖昧になる）。

追記すると content が書き込み後に変わる。埋め込みは書き込み時に作られるので、**作り直さない
と content と situated ベクトルが食い違う**。想起はベクトルで探すので、足した文言は検索に
効かず、W へ出る文面だけが変わってしまう。実測で「500字の埋め込みと挿入」は 26ms であり、
LLM 呼び出しが1ターン 10.5 秒のうち 10.2 秒を占めることに比べれば誤差である。

`close_with_children`（親と全子をまとめて畳む）は撤去する。`superseded_by` を版履歴だけの
意味に戻したので、親子のファンアウトを畳む操作そのものが要らない。
"""

from __future__ import annotations

import os
import pathlib
import uuid

import psycopg2
import pytest

from familiar_agent.tools.memory import ObservationMemory

_NOTE = "検索を始めた"


def _pg():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    conn.autocommit = True
    return conn


def _col(obs_id: str, col: str):
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute(f"SELECT {col} FROM observations WHERE id = %s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def _vector(obs_id: str):
    conn = _pg()
    with conn.cursor() as cur:
        cur.execute("SELECT vector FROM obs_embeddings WHERE obs_id = %s", (obs_id,))
        row = cur.fetchone()
    conn.close()
    return bytes(row[0]) if row else None


@pytest.fixture()
def memory():
    return ObservationMemory()


def test_note_is_appended_once(memory) -> None:
    """「検索を始めた」を追記する。二度目は足さない。"""
    obs_id, _ = memory.save_with_id(f"昨日の天気覚えてる？ {uuid.uuid4()}",
                                    direction="発話", kind="observation")
    memory.note_lookup_started(obs_id)
    once = str(_col(obs_id, "content"))
    memory.note_lookup_started(obs_id)
    twice = str(_col(obs_id, "content"))

    assert _NOTE in once, "追記されていない"
    assert once == twice, "二度目も足している"
    assert once.count(_NOTE) == 1, f"印が重なっている: {once}"


def test_the_embedding_is_recomputed(memory) -> None:
    """追記したら埋め込みを作り直す（content とベクトルを食い違わせない）。"""
    obs_id, _ = memory.save_with_id(f"今日の天気は？ {uuid.uuid4()}",
                                    direction="発話", kind="observation")
    before = _vector(obs_id)
    assert before, "埋め込みが作られていない（前提が崩れている）"

    memory.note_lookup_started(obs_id)

    after = _vector(obs_id)
    assert after, "埋め込みが消えている"
    assert after != before, "content を変えたのに埋め込みが古いまま"


def test_the_original_text_survives(memory) -> None:
    """人が言ったことは消さない（追記であって置き換えではない）。"""
    body = f"たいきのサッカーは？ {uuid.uuid4()}"
    obs_id, _ = memory.save_with_id(body, direction="発話", kind="observation")
    memory.note_lookup_started(obs_id)
    assert body in str(_col(obs_id, "content")), "元の発話が消えている"


def test_a_missing_record_is_tolerated(memory) -> None:
    """記録が見つからなくても落ちない。"""
    memory.note_lookup_started(str(uuid.uuid4()))


def test_close_with_children_is_gone() -> None:
    """親子をまとめて畳む仕組みは撤去した。"""
    from familiar_agent.store.observations import ObservationStore

    assert not hasattr(ObservationMemory, "close_with_children"), (
        "ObservationMemory に close_with_children が残っている"
    )
    assert not hasattr(ObservationStore, "close_with_children"), (
        "ObservationStore に close_with_children が残っている"
    )


def test_no_source_calls_close_with_children() -> None:
    """呼び出しがソースに残っていない。"""
    root = pathlib.Path(__file__).resolve().parents[1]
    stale = []
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "close_with_children" in text:
            stale.append(str(path.relative_to(root)))
    assert not stale, "close_with_children が残っている:\n" + "\n".join(stale)
