"""親子の MI（求めと、その求めのために投げた調査）。

調査は複数並行しうるので、状態は1本の鎖では表せない。親（人の求め・情動）と子（調査）の
**2階層だけ**を持ち、孫は作らない。**親を閉じるとき、生きている子を全部閉じる**（一段だけ・
再帰なし）。子どうしの supersede は従来どおり。

抜けの検出は作らない。W から落ちたものは薄れた＝忘れたのであって、改めて調べるのが自然な
振る舞いである（W は「速く薄れる」・用語一覧）。
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

_DB_URL = os.environ["DATABASE_URL"]


def _conn():
    c = psycopg2.connect(_DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    c.autocommit = False
    return c


_AGENT_SELF = "00000000-0000-0000-0000-000000000000"


def _plant(conn, content: str, parent_id: str | None = None) -> str:
    oid = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO persons (id, name, created_at, updated_at) VALUES (%s, %s, now(), now()) "
            "ON CONFLICT (id) DO NOTHING",
            (_AGENT_SELF, "__self__"),
        )
        cur.execute(
            "INSERT INTO observations (id,content,timestamp,direction,kind,emotion,person_id,"
            " writer_id,subject_id,participants_json,scope,parent_id) "
            "VALUES (%s,%s,%s,'unknown','observation','neutral',%s,%s,%s,'[]','speaker',%s)",
            (oid, content, datetime.now(timezone.utc), _AGENT_SELF, _AGENT_SELF,
             _AGENT_SELF, parent_id),
        )
    conn.commit()
    return oid


def test_parent_id_column_exists() -> None:
    conn = _conn()
    with conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_name='observations' AND column_name='parent_id'")
        row = cur.fetchone()
    conn.close()
    assert row is not None


def test_parent_child_links_survive_without_bulk_closing() -> None:
    """親子の紐づけ（`parent_id`）は残す。まとめて畳む操作だけを撤去した。

    求めは1本の版チェーンとして進むので、親子のファンアウトを畳む必要が無い。ただし
    `parent_id` そのものは、どの求めのために書かれた記録かを辿るのに使う。
    """
    conn = _conn()
    tag = uuid.uuid4().hex[:8]
    parent = _plant(conn, f"{tag} 求め")
    child = _plant(conn, f"{tag} 版1", parent_id=parent)

    with conn.cursor() as cur:
        cur.execute("SELECT parent_id FROM observations WHERE id = %s", (child,))
        got = cur.fetchone()
    conn.close()
    assert got and str(got[0] if not isinstance(got, dict) else got["parent_id"]) == parent

def test_the_answer_is_written_at_speak_time_and_survives_the_close():
    """自分の答えを、背景の永続化を待たずに O へ書き、閉じても生き残らせる。

    背景処理（要約・内省）は2秒かかる。そのあいだに次の反復が起きると「さっき何と
    言ったか」を拾えない（実機で「それだけ？」に聞き返した）。かといって子として書くと、
    求めの決着で `superseded_by` が入り、想起の候補（新しさ軸・関連軸とも
    `superseded_by IS NULL`）から外れて結局見つけられない。
    **閉じる側をこの記録にする**ことで、生き残るのはこの1件だけになる。
    """
    from familiar_agent.backend import ToolCall
    from tests.test_event_loop import _agent, _run, _turn

    a = _agent(stream_returns=[_turn([ToolCall(id="t", name="say", input={"text": "晴れだよ"})])])
    _run(a, utterance="今日の天気は？")

    answers = [c for c in a._memory.save_async_with_id.call_args_list
               if c.kwargs.get("direction") == "発話" and "自分が答えた" in c.args[0]]
    assert len(answers) == 1
    assert "晴れだよ" in answers[0].args[0]

    # **この記録は鎖の外**。何も畳まない（求めの版チェーンは別に進む）。
    assert not a._memory.close_with_children.called, "close_with_children を呼んでいる"
    # 背景の永続化には、この記録を supersede する対象として渡る（要約が恒久記録を担う）。
    _, kwargs = a._run_post_response_pipeline.call_args
    assert kwargs["superseded_ids"], "答えの記録が渡っていない"
