"""調停の文章は、要るときだけ軽量LLM が 1 回で書く（出-au 段 5-7c・2026-09-27・`設計方針_判定の段` v0.4 §2.2.2）。

| Jev の答え | 書くもの |
|---|---|
| light | 返事（`text`） |
| full で深さが low でない | つなぎ（`filler`）。情動が起点なら書かない（自発に断りは要らない・情-e） |
| action | 動作の中身（`query`・`tool_input`）とつなぎ。首を向ける・道具（タイマーなど）・確認への「いい」はつなぎ無し |
| 時期を指している | 日付（`time_ref`・`time_span_days`） |

何も要らなければ呼ばない。口調の決まり（ME と FAMILY に従う・丁寧さを混ぜない・言い直さない）は、いままでの調停の文を持つ。
書けない・時間切れなら None（呼び手が full へ倒す）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import Arbiter, ArbiterInput


def _inp(**kw):
    base = dict(
        utterance="パジュ、3分測って", workspace_ctx="（なし）", family_md="", origin="発話"
    )
    base.update(kw)
    return ArbiterInput(**base)


def _writer(reply='{"text": "はーい"}'):
    w = MagicMock()
    w.complete = AsyncMock(return_value=reply)
    return w


def _write(data, inp=None, writer=None):
    w = writer or _writer()
    got = asyncio.run(Arbiter(jev=None, writer=w, timeout=2.0)._write(inp or _inp(), data))
    return got, w


def _asked(w) -> str:
    return w.complete.await_args.args[0]


def test_light_asks_only_for_the_reply():
    got, w = _write({"branch": "light"})
    assert got == {"text": "はーい"}
    p = _asked(w)
    assert '"text"' in p and '"filler"' not in p and '"tool_input"' not in p


def test_full_at_low_effort_needs_nothing_and_calls_nothing():
    got, w = _write({"branch": "full", "effort": "low"})
    assert got == {}
    w.complete.assert_not_awaited()


def test_full_at_higher_effort_asks_for_a_filler_but_not_for_self_driven_turns():
    _, w = _write({"branch": "full", "effort": "medium"})
    assert '"filler"' in _asked(w)
    got, w2 = _write({"branch": "full", "effort": "medium"}, _inp(origin="情動"))
    assert got == {}
    w2.complete.assert_not_awaited()


def test_a_timer_asks_for_the_tool_input_with_its_description_and_no_filler():
    _, w = _write({"branch": "action", "action": "set_timer"}, _inp(extra_actions=("set_timer",)))
    p = _asked(w)
    assert '"tool_input"' in p and "after_minutes" in p and '"filler"' not in p


def test_a_search_asks_for_the_words_and_a_filler():
    _, w = _write({"branch": "action", "action": "search_deferred"})
    p = _asked(w)
    assert '"query"' in p and '"filler"' in p


def test_a_time_reference_asks_for_the_date():
    _, w = _write({"branch": "full", "effort": "low", "refers_time": True})
    assert '"time_ref"' in _asked(w)


def test_the_tone_rules_and_the_stance_reach_the_writer():
    _, w = _write(
        {"branch": "light"}, _inp(self_understanding="わたしはパジュ", family_md="## パパ")
    )
    assert "丁寧さを混ぜない" in _asked(w)
    assert "わたしはパジュ" in (w.complete.await_args.kwargs.get("system") or "")


def test_an_unreadable_or_failed_write_is_none():
    got, _ = _write({"branch": "light"}, writer=_writer("ごめん"))
    assert got is None
    w = MagicMock()
    w.complete = AsyncMock(side_effect=RuntimeError("down"))
    assert _write({"branch": "light"}, writer=w)[0] is None


def test_the_writer_is_told_to_write_not_to_choose():
    """書く側はもう選ばない。4 つの起点すべてで、選ばせる言い方を渡さない（出-au 段 5-7d）。"""
    cases = [
        _inp(),
        _inp(origin="機器", utterance="[タイマー] 鳴った"),
        _inp(tool_return=True),
        _inp(origin="情動", utterance="[内的な促し:bond] 話したい"),
    ]
    for inp in cases:
        _, w = _write({"branch": "light"}, inp)
        asked = _asked(w)
        assert "次のどれかを選ぶ" not in asked, inp.origin
        assert "何をするかを決める" not in asked, inp.origin
        assert "決まったことに要る言葉を書く" in asked, inp.origin
