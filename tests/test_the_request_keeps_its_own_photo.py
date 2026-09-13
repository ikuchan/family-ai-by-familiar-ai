"""写真の在りかは求めが持つ（2026-09-13 実機で露見）。

W に載っているかで探すと、VLM の差し替え（新しい記録）がその瞬間まだ検索に載っておらず
「写真なし」になり、調停が see を出し直した。この求めで最後に撮った写真の在りかを
`Request.seen_image_path` に持ち、W に依らず主LLM・調停へ渡す。求めが閉じれば消える。
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def test_the_mark_writes_the_path_into_the_request(monkeypatch) -> None:
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)

    async def scenario():
        await ip._write_seen_mark("出入り口を見た。", image_path="/tmp/x.jpg")
        got = ip._req.seen_image_path
        await ip.close()
        return got

    assert asyncio.run(scenario()) == "/tmp/x.jpg"


def test_the_photo_comes_from_the_request_not_the_workspace(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip = InformationProcessing(_agent(stream_returns=[]))
    ip._req.seen_image_path = str(path)
    out = ip._user_content("x", [])  # W は空でも添える
    assert isinstance(out, list)
    assert base64.b64decode(out[1]["source"]["data"]) == b"JPEG"


def test_no_photo_without_a_path() -> None:
    ip = InformationProcessing(_agent(stream_returns=[]))
    assert ip._user_content("x", []) == "x"


def test_the_path_is_cleared_when_the_request_closes() -> None:
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.seen_image_path = "/tmp/x.jpg"
    ip._hold_speech = AsyncMock()  # type: ignore[method-assign]

    async def scenario():
        await ip._finish("", [], "沈黙")
        got = ip._req.seen_image_path
        await ip.close()
        return got

    assert asyncio.run(scenario()) is None
