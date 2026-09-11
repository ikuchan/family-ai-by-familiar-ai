"""**W の1行は言葉で書く**——その記録が何で、誰のものかを。

いままでは種別の語をそのまま括弧に入れていた（`(発話)`・`(会話)`）。これは**記号**で、
読み手（主LLM・軽量LLM）が対応表を知っていて初めて意味になる。しかも `発話` は人の言葉にも
パジュの言葉にも付いており、**誰が言ったかを言えていなかった**。

「誰がやったか」を持つのは `situated_memories` の **`actor` の面**である。`direction` でも
`やりとり` の役割でもない。ただし **`actor` の意味は記録ごとに変わる**——`発話` なら言った人、
`会話`（要約）なら**その会話の相手**（書いたのはパジュ）である。だから種別ごとに文面を持つ。

**分からないほうを落とす。** 種別を知らなければ主体だけ言い、主体が分からなければ種別だけ
言う。名前を捏造しない。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from familiar_agent.tools.memory import subject_line as _line_of


def _line(direction: str, name: str | None) -> str:
    return _line_of(direction, name)


def test_an_utterance_names_who_said_it():
    assert _line("発話", "ゆうすけ") == "ゆうすけが言った"
    assert _line("発話", "わたし") == "わたしが言った"


def test_a_conversation_names_the_other_side_not_the_writer():
    """`会話` の `actor` は**その会話の相手**で、書いたのはパジュである。
    `(会話・ゆうすけ)` では「ゆうすけが書いた」とも読めてしまう。"""
    assert _line("会話", "ゆうすけ") == "ゆうすけとの会話"


def test_the_agents_own_records_say_so():
    assert _line("観察", "わたし") == "わたしが見た"
    assert _line("独白", "わたし") == "わたしが考えた（言わなかった）"
    assert _line("保留", "わたし") == "わたしが言えずにいたこと"


def test_an_unknown_direction_says_only_the_subject():
    """表に無い種別（本番 DB に残る `完了`・`意図`・`中断`・`unknown`）は主体だけ言う。"""
    for direction in ("完了", "意図", "中断", "unknown", ""):
        assert _line(direction, "ゆうすけ") == "ゆうすけの記録", direction


def test_a_missing_actor_says_only_the_kind():
    """**名前を捏造しない。** `actor` の面は `materialize_now` で立つので、W に載る時点で
    まだ無いことがある。"""
    assert _line("会話", None) == "会話"
    assert _line("", None) == "記録"


def test_the_workspace_attaches_the_subject_from_the_actor_face():
    """W を組むときに、`actor` の面から主体を引いて添える。"""
    from familiar_agent.loop import workspace
    from familiar_agent.loop.request import Request

    from datetime import datetime

    from familiar_agent.io.oif import MI, Recalled

    def _r(obs_id, content):
        return Recalled(
            mi=MI(
                id=obs_id,
                obs_id=obs_id,
                content=content,
                timestamp=datetime(2026, 9, 11, 15, 0),
                direction="発話",
            ),
            fit=0.5,
            groundedness=1.0,
            confidence=0.8,
        )

    oif = MagicMock()
    oif.actors.return_value = {"m1": "ゆうすけ"}
    text, _ = workspace.compose(oif, [_r("m1", "あ"), _r("m2", "い")], Request())
    assert "ゆうすけが言った: あ" in text
    assert "発話: い" in text  # 面の無い m2 は種別だけ（名前を捏造しない）


def test_the_names_come_from_the_actor_face_only():
    """引くのは `actor` の面だけ。`present`・`addressee`・`about` は「誰がやったか」では
    ないので、主体にしない。"""
    import inspect

    from familiar_agent.store.situated import SituatedVectors

    src = inspect.getsource(SituatedVectors.actors_of)
    assert "relation_key = 'actor'" in src
