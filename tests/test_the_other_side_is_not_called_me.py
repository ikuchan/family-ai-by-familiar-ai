"""**話者が分からなくても、相手の言葉を「わたし」と呼ばない**（2026-09-12 実機で露見）。

話者が同定されていないと、規則 048 で人の言葉の `actor` 面は `__self__` に立つ（解決
できない記録はパジュ自身のもの）。記-f の印字はその面を読んで「わたしが言った」と書き、
主LLM には**「自分が『何が見えますか？』と言った」**と見えていた。

048 は**面（誰の記憶か）**の規則で、**誰が言ったか**とは別である。誰が言ったかは
`やりとり` の役割が持つ——`起点` は相手、`答え`・`つなぎ` はパジュ。いまの求めの分は
`turn_records` に、閉じたやりとりの分は関係にある。

あわせて、`see` の帰りを版に書く文言も直す。「（見たことは観察に記録した）」は主LLM に
「別の場所を見ろ」と読ませ、W の最下位にある `見えたもの` より重く見られて、`see` が5回
続いた。W のどこにあるかを言う。
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request


def _r(obs_id, content, direction="発話"):
    return Recalled(
        mi=MI(
            id=obs_id,
            obs_id=obs_id,
            content=content,
            timestamp=datetime(2026, 9, 12, 12, 41),
            direction=direction,
        ),
        fit=0.5,
        groundedness=1.0,
        confidence=0.8,
    )


def _oif(actors, roles):
    o = MagicMock()
    o.actors.return_value = actors
    o.roles.return_value = roles
    return o


def test_an_unresolved_speakers_words_are_the_other_sides():
    """面は `__self__`（048）でも、役割 `起点` なら相手が言った。"""
    req = Request()
    oif = _oif({"q": "わたし"}, {"q": "起点"})
    text, _ = workspace.compose(oif, [_r("q", "何が見えますか？")], req)
    assert "相手が言った: 何が見えますか？" in text
    assert "わたしが言った" not in text


def test_a_named_speaker_keeps_the_name():
    oif = _oif({"q": "ゆうすけ"}, {"q": "起点"})
    text, _ = workspace.compose(oif, [_r("q", "何が見えますか？")], Request())
    assert "ゆうすけが言った: 何が見えますか？" in text


def test_the_agents_own_answer_is_still_me():
    oif = _oif({"a": "わたし"}, {"a": "答え"})
    text, _ = workspace.compose(oif, [_r("a", "自分が答えた：はい")], Request())
    assert "わたしが言った" in text


def test_the_current_requests_origin_is_known_before_the_exchange_closes():
    """いまの求めの起点は、まだ関係に無い。`turn_records` から分かる。"""
    req = Request()
    req.turn_records = [("q", "起点")]
    oif = _oif({"q": "わたし"}, {})  # 関係にはまだ無い
    text, _ = workspace.compose(oif, [_r("q", "何が見えますか？")], req)
    assert "相手が言った" in text


def test_the_version_points_to_the_workspace_for_what_was_seen():
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    src = inspect.getsource(InformationProcessing._intake)
    assert "（見たことは観察に記録した）" not in src
    assert "わたしが見た" in src
