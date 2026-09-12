"""話せなかった理由を、記録に残す。

`_delivery_block_reason` は止めた理由を3通りに返し分けている。

- `黙っているよう頼まれている`（「黙っていて」と言われた）
- `聞く相手が居ない`（誰も居ない）
- `静穏時間である`（夜間に自分から話そうとした）

`_speak` はその理由をログには出していたが、**`_hold_speech` へ渡していなかった**。記録は
固定文で `話したかったが、聞く相手が居なかった：…` と書かれ、**「黙っていてと言われたので
やめた」が「誰も居なかった」として残っていた**。

保留は後で配られる（`_release_pending_speech`）ので、パジュはその文面を読んで話し始める。
理由が違えば、話し出し方も違う。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock


def _ip():
    from familiar_agent.loop.event_loop import InformationProcessing

    ip = InformationProcessing.__new__(InformationProcessing)
    a = MagicMock()
    a._memory.save_async_with_id = AsyncMock(return_value=("obs-1", True))
    from familiar_agent.io.oif import OIF

    # 書き込みは OIF を通る（環-e-い）。**口は本物・内側の記憶だけ偽物**にすれば、
    # `save_async_with_id` への検証がそのまま効く。書き手は口が必須で求める。
    a._observation_perspective = MagicMock(return_value={"writer_id": "__self__"})
    a._conversation_perspective = MagicMock(return_value={"writer_id": "話者"})
    a._oif = OIF(a._memory)
    a._pending_store.add = MagicMock()
    ip._agent = a
    return ip, a


def _written(a) -> str:
    return a._memory.save_async_with_id.await_args.args[0]


def test_being_asked_to_stay_quiet_is_recorded_as_such():
    ip, a = _ip()
    asyncio.run(ip._hold_speech("おはよう", "黙っているよう頼まれている"))
    got = _written(a)
    assert "黙っているよう頼まれていた" in got
    assert "聞く相手が居なかった" not in got


def test_nobody_present_still_says_so():
    ip, a = _ip()
    asyncio.run(ip._hold_speech("おはよう", "聞く相手が居ない"))
    assert "聞く相手が居なかった" in _written(a)


def test_quiet_hours_is_recorded_as_such():
    ip, a = _ip()
    asyncio.run(ip._hold_speech("おはよう", "静穏時間である"))
    assert "静穏時間だった" in _written(a)


def test_an_unknown_reason_is_kept_verbatim():
    """理由が増えたとき、黙って誤った文へ倒さない。"""
    ip, a = _ip()
    asyncio.run(ip._hold_speech("おはよう", "まだ知らない理由"))
    got = _written(a)
    assert "まだ知らない理由" in got
    assert "聞く相手が居なかった" not in got


def test_the_text_itself_is_still_kept():
    ip, a = _ip()
    asyncio.run(ip._hold_speech("おはよう", "聞く相手が居ない"))
    assert "おはよう" in _written(a)
    assert a._memory.save_async_with_id.await_args.kwargs["direction"] == "保留"


def test_speaking_passes_the_reason_it_already_has():
    """`_speak` は理由を持っている（ログへ出している）。渡すだけである。"""
    from familiar_agent.loop.request import Request

    ip, _a = _ip()
    ip._req = Request()  # 起点は既定（人の発話）。独り言は積まない（情-c）
    ip._delivery_block_reason = MagicMock(return_value="黙っているよう頼まれている")
    ip._hold_speech = AsyncMock()
    ip._dif = MagicMock(speak=AsyncMock())
    ip._emit = MagicMock()
    assert asyncio.run(ip._speak("おはよう")) == ("", "保留")
    ip._hold_speech.assert_awaited_once_with("おはよう", "黙っているよう頼まれている")
