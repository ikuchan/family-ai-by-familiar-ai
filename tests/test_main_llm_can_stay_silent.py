"""主LLM も、人の言葉に返しようがなければ黙れる（出-db 段 F・2026-10-25・本人の決定ウ・ア）。

主LLM に「返しようがなければ say を呼ばず『（黙る）』とだけ書く」と伝えると、テレビと崩れた声 5 件を全部黙らせ、頼み・
しりとりの一語・はい・ちがうには答えた（claude-sonnet-5-5・実験）。伝えるのは、人の声への最初の返事のときだけ（調べものの
結果が返ったとき・内からの求め・キーボードで打った言葉には伝えない）。その回の指示（`[反復]` の行）に載せ、system 文の固定部分
には載せない（キャッシュを崩さない）。「（黙る）」だけの文は、返事か独り言かを Jev に聞かず、差し戻しもせず、声にせずに閉じる。
"""

from __future__ import annotations

import pytest

from familiar_agent.loop.generator import _iter_ctx
from tests.test_bare_text_is_sent_back_for_say import _act, _bare, _with_jev


def _ctx(**kw):
    return _iter_ctx(chain=1, max_chain=4, thinking_round=1, capped=False, **kw)


def test_the_instruction_is_given_only_when_asked():
    said = _ctx(may_stay_silent=True)
    assert "say を呼ばず「（黙る）」とだけ書く" in said
    assert "（黙る）" not in _ctx()  # **反証の側**


@pytest.mark.parametrize(
    "text",
    [
        "（黙る）",
        " (黙る) ",
        "（黙る）\n",
        "「（黙る）」",
        "（黙る）。",
        "テレビの声なので（黙る）",
    ],
)
def test_the_silence_mark_closes_without_jev_or_send_back(text):
    ip, a = _with_jev("reply")
    assert _act(ip, _bare(text)) == ""
    a._jev.ask.assert_not_awaited()
    ip._dispatch_main_llm.assert_not_called()
    ip._speak.assert_not_awaited()
    ip._emit.assert_not_called()
    assert ip._finish.await_args.args[2] == "沈黙"


def test_other_bare_text_is_still_judged():
    """**反証の側**：印でない文は、いまどおり Jev に返事か独り言かを聞く。"""
    ip, a = _with_jev("reply")
    _act(ip, _bare("（黙る）って言いたいけど、晴れだよ"))
    a._jev.ask.assert_awaited()


# --- いつ伝えるか（`Iteration` が決める）---


def _may(**kw):
    from familiar_agent.loop.iteration import may_stay_silent

    base = dict(kind="発話", round_=1, returned=False, source="voice")
    base.update(kw)
    return may_stay_silent(**base)


def test_told_only_for_the_first_reply_to_a_voice():
    assert _may()
    assert not _may(kind="情動")
    assert not _may(kind="機器")
    assert not _may(round_=2)  # 言い直し・考え直し
    assert not _may(returned=True)  # 調べものの結果が返った
    assert not _may(source="keyboard")


def test_the_mark_inside_say_is_not_spoken():
    from familiar_agent.backends import ToolCall
    from familiar_agent.backends.types import TurnResult

    ip, a = _with_jev("reply")
    said = TurnResult(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id="t1", name="say", input={"text": "（黙る）"})],
    )
    assert _act(ip, said) == ""
    ip._speak.assert_not_awaited()
    assert ip._finish.await_args.args[2] == "沈黙"


def test_a_retry_does_not_stay_silent_on_the_mark():
    """**反証の側**：言い直し・差し戻しの返りには指示を渡していないので、印を見て黙らない（いまどおりの道を通る）。"""
    ip, a = _with_jev("reply")
    _act(ip, _bare("（黙る）"), say_asked=True)
    assert ip._finish.await_args.args[2] == "沈黙"  # 差し戻しの返りの文は、いまどおり沈黙で閉じる
    ip2, a2 = _with_jev("reply")
    _act(ip2, _bare("（黙る）"), retried=True)
    a2._jev.ask.assert_not_awaited()


def test_retries_are_sent_without_the_note():
    from familiar_agent.loop.event_loop import _without_silence_note
    from familiar_agent.loop.generator import SILENCE_NOTE

    system = ("安定", "[反復] 1/4\n" + SILENCE_NOTE + "\n[タイマー] なし")
    got = _without_silence_note(system)
    assert SILENCE_NOTE not in "".join(got) and "[タイマー] なし" in got[1]
    assert _without_silence_note("x") == "x"


def test_the_mark_is_seen_before_rethinking():
    """印は考え直しの判定より先に見る（印を下書きにして考え直さない）。"""
    from unittest.mock import AsyncMock

    from familiar_agent.backends import ToolCall
    from familiar_agent.backends.types import TurnResult

    ip, a = _with_jev("reply")
    ip._rethink_or_speak = AsyncMock()
    said = TurnResult(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id="t1", name="say", input={"text": "（黙る）"})],
    )
    _act(ip, said)
    ip._rethink_or_speak.assert_not_awaited()
