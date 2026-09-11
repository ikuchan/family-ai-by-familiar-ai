"""**人の言葉は、その人がやったことである**（`actor` の面は話者に立つ）。

`situated_memories` は人ごとで、「誰がやったか」は `relation_key='actor'` の面が持つ
（`store/situated.py`）。想起は `_active_memory()`＝**話者の面**を引くので、人の言葉が
`__self__` の面にしか立たないと、**その人の面には、その人が言ったことが1件も無い**。

本番 DB（2026-07〜08・読み取りで確認）では、人の言葉 484 件の `actor` が
`いくながゆうすけ` に立っており、`__self__` は 37 件（話者が解決できなかったぶん・規則 048）
だった。**イベント駆動ループへ移したときに、この配線が落ちた。**

ループが O を書く6箇所のうち、**話者のものは1つだけ**である——人の言葉（`_begin_request`）。
版・見た・つなぎ・保留・答えは、どれもパジュ自身がやったことなので `__self__` でよい。
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing


def _ip():
    ip = object.__new__(InformationProcessing)
    from familiar_agent.loop.request import Request

    ip._req = Request()
    ip._note_origin = MagicMock()
    a = MagicMock()
    a._observation_perspective.return_value = {"writer_id": "__self__", "participants": []}
    a._conversation_perspective.return_value = {"writer_id": "話者", "participants": []}
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-1", None))
    ip._agent = a
    return ip, a


def test_a_human_utterance_is_written_as_the_speakers():
    ip, a = _ip()
    asyncio.run(ip._begin_request(kind="発話", text="ただいま", utterance="ただいま"))
    assert a._memory.save_async_with_id.await_args.kwargs["writer_id"] == "話者"
    a._observation_perspective.assert_not_called()


def test_an_affect_or_a_device_stays_the_agents():
    """情動と機器はパジュ自身のことなので、面はいままでどおり `__self__` に立つ。"""
    for kind in ("情動", "機器"):
        ip, a = _ip()
        asyncio.run(ip._begin_request(kind=kind, text=f"[{kind}] なにか"))
        assert a._memory.save_async_with_id.await_args.kwargs["writer_id"] == "__self__"
        a._conversation_perspective.assert_not_called()


def test_only_the_utterance_uses_the_speakers_face():
    """**話者の面を使う書き込みは1箇所だけ。** 増えたら、なぜ増えたかを言う。"""
    src = inspect.getsource(InformationProcessing)
    assert src.count("_conversation_perspective()") == 1
