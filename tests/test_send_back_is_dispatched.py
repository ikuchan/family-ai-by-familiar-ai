"""整合チェックの差し戻しも投げっぱなしにする（環-h・段は-2）。

`_act_on_decision` の中に、ループ最後の同期 `stream_turn` が残っていた——整合チェックが
違反を捕まえたときだけ、主LLM を `await` で呼び直す箇所である。

そこだけ同期だと、言い直しのあいだ（`effort` 次第で数秒〜10秒）**打ち切りが効かず**、
情動・機器のキューがそのぶん待たされる。`Decision` が `system`・`effort` を運んでいたのも、
この差し戻しのためだけだった。

投げっぱなしにすると、**出す反復が2回起きる**。

| 出す反復 | `retried` | やること |
|---|---|---|
| 1回目 | False | say を拾う → 申告 → 整合チェック → **違反なら投げて閉じない** |
| 2回目 | True | **検査しない**（1回だけ）→ 話して閉じる |

判定（軽量LLM）・規則・「1回だけ」の決まりは変えない。変えるのは送り方だけである。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.backends import ToolCall
from familiar_agent.backends.types import TurnResult
from familiar_agent.loop.event_loop import InformationProcessing

_VIOLATION = "見ていないのに見たと言っている"


def _ip():
    ip = InformationProcessing.__new__(InformationProcessing)
    ip._request_generation = 0
    ip._cue = "手がかり"
    ip._request_id = "req-1"
    ip._lookups = []
    ip._background_tasks = set()
    ip._completion_queue = asyncio.Queue()
    ip._say_filler = AsyncMock()
    ip._start_lookup = MagicMock()
    ip._apply_memory_verdicts = MagicMock()
    ip._coherence_violation = AsyncMock(return_value=None)
    ip._speak = AsyncMock(return_value=("直した", "発話"))
    ip._finish = AsyncMock()
    ip._emit = MagicMock()
    ip._tools = MagicMock(return_value=[])
    ip._write_version = AsyncMock(return_value="ver-1")
    ip._dispatch_main_llm = MagicMock()
    a = MagicMock()
    a.backend.stream_turn = AsyncMock()
    a.backend.make_user_message = MagicMock(side_effect=lambda t: {"role": "user", "content": t})
    a.config.max_tokens = 1000
    ip._agent = a
    return ip, a


def _say(text="そこに本があるね", **extra):
    return TurnResult(
        stop_reason="tool_use", text="", tool_calls=[ToolCall("t", "say", {"text": text, **extra})]
    )


def _run(ip, result, *, retried=False, original_text="", gen=0):
    return asyncio.run(
        ip._act_on_decision(
            result,
            memories=[],
            recent_ctx="",
            utterance="そこに何がある？",
            system=("安定", "可変"),
            effort="high",
            gen=gen,
            capped=False,
            retried=retried,
            original_text=original_text,
        )
    )


# ── 1. 投げて閉じない ──────────────────────────────────────────────────────


def test_a_violation_is_dispatched_and_the_turn_stays_open():
    """違反なら投げて閉じる**前に**返る。話してしまえば、求めに答えが2件書かれる。"""
    ip, _a = _ip()
    ip._coherence_violation = AsyncMock(return_value=_VIOLATION)
    assert _run(ip, _say()) == ""
    ip._speak.assert_not_awaited()
    ip._finish.assert_not_awaited()
    ip._dispatch_main_llm.assert_called_once()


def test_the_send_back_does_not_wait_for_the_main_llm():
    """同期の `stream_turn` は残っていない（ここが 段は-2 の目的）。"""
    ip, a = _ip()
    ip._coherence_violation = AsyncMock(return_value=_VIOLATION)
    _run(ip, _say())
    a.backend.stream_turn.assert_not_awaited()


# ── 2. 何を運ぶか ──────────────────────────────────────────────────────────


def test_the_send_back_carries_the_flag_and_the_original_answer():
    ip, _a = _ip()
    ip._coherence_violation = AsyncMock(return_value=_VIOLATION)
    _run(ip, _say("そこに本があるね"))
    kw = ip._dispatch_main_llm.call_args.kwargs
    assert kw["retried"] is True
    assert kw["original_text"] == "そこに本があるね"


def test_the_send_back_message_names_the_violation():
    ip, _a = _ip()
    ip._coherence_violation = AsyncMock(return_value=_VIOLATION)
    _run(ip, _say())
    sent = ip._dispatch_main_llm.call_args.kwargs["messages"][0]["content"]
    assert "[SELF-CHECK]" in sent
    assert "そこに本があるね" in sent
    assert _VIOLATION in sent


def test_the_send_back_writes_a_version():
    """投げた事実は版に載る（`N番：考えている`）。載らないと求めの状態が飛ぶ。"""
    ip, _a = _ip()
    ip._coherence_violation = AsyncMock(return_value=_VIOLATION)
    _run(ip, _say())
    ip._write_version.assert_awaited_once()


# ── 3. 言い直しは検査しない（1回だけ）──────────────────────────────────────


def test_the_reworded_answer_is_not_checked_again():
    ip, _a = _ip()
    _run(ip, _say("直した"), retried=True, original_text="そこに本があるね")
    ip._coherence_violation.assert_not_awaited()
    ip._speak.assert_awaited_once_with("直した")


# ── 4. 言い直しが say を返さなかったら元の応答で出す ────────────────────────


def test_a_reworded_answer_without_say_falls_back_to_the_original():
    ip, _a = _ip()
    _run(
        ip,
        TurnResult(stop_reason="end_turn", text="うまく言えない", tool_calls=[]),
        retried=True,
        original_text="そこに本があるね",
    )
    ip._speak.assert_awaited_once_with("そこに本があるね")
    ip._finish.assert_awaited_once()


# ── 5. 言い直しに渡す道具は say だけ ───────────────────────────────────────


def test_the_reword_is_given_only_say():
    """言い直しは答え直すためのもので、調べ直すためのものではない。

    上限（`capped`）と同じ結論になるが、**理由が違う**ので1つの名前へまとめない。
    """
    src = inspect.getsource(InformationProcessing._run_main_llm)
    assert "capped or retried" in src


# ── 6. 言い直しも「考えた回数」に数える ───────────────────────────────────


def test_the_send_back_goes_through_the_same_door():
    """普段の呼び出しと**同じ口**（`_dispatch_main_llm`）から投げる。

    その口が `_lookups` へ `action="主LLM"` として積むので、言い直しも**考えた回数に
    数えられる**（2026-09-09 の決定）。別の口を作れば数えないことになる。
    """
    src = inspect.getsource(InformationProcessing._act_on_decision)
    assert "self._dispatch_main_llm(" in src
    assert "stream_turn(" not in src, "同期の呼び出しが残っている"


# ── 7. 言い直しの最中でも打ち切れる（段は-2 の目的）────────────────────────


def test_a_reword_of_an_aborted_request_is_dropped():
    ip, _a = _ip()
    ip._request_generation = 3  # 待っているあいだに話しかけられた
    assert _run(ip, _say("直した"), retried=True, original_text="元の文", gen=1) == ""
    ip._speak.assert_not_awaited()
    ip._finish.assert_not_awaited()
