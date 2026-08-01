"""意味づけに失敗したとき、何が返ってきたかを残す。

実機で `see` が「見たが、意味づけは何も返さなかった」を出し続けた。原因は
`json.loads(raw)` の `Expecting value: line 1 column 1 (char 0)`＝**先頭から JSON でない**
ことだったが、失敗のログが `debug`（本番では切る）で例外の型しか出ておらず、**VLM が何を
返したのかが分からなかった**。

空文字なのか、説明文なのか、JSON もどきなのかで、対応が変わる。まずそれを残す。

記録に載せるのは**返答の先頭だけ**にする。情景の説明は長く、`warning` で全文を出すと
ログが埋まる。会話・記憶の内容を INFO 以上に出さない方針とも揃える。
"""

from __future__ import annotations

import logging

import pytest

from familiar_agent.scene import extract_entities

_LOGGER = "familiar_agent.scene"


class _Backend:
    """指定した文字列を返すだけの器（画像の口は持たない）。"""

    def __init__(self, reply):
        self._reply = reply

    async def complete(self, _prompt, *_a, **_kw):
        return self._reply


@pytest.mark.asyncio
async def test_a_non_json_reply_is_logged_with_its_head(caplog) -> None:
    """JSON でない返答は、先頭を添えて warning に残す。"""
    reply = "This image shows a living room with a sofa and a window." * 10
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await extract_entities("説明", _Backend(reply))

    assert got == [], "解釈できないのに何か返している"
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs, "失敗が warning に残っていない"
    assert "This image shows" in msgs[0], f"返答の先頭が残っていない: {msgs[0]}"


@pytest.mark.asyncio
async def test_an_empty_reply_is_logged_as_empty(caplog) -> None:
    """空の返答は、空だと分かる形で残す（説明文が返る場合と区別する）。"""
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await extract_entities("説明", _Backend(""))

    assert got == []
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs, "失敗が warning に残っていない"
    assert "空" in msgs[0], f"空だと分からない: {msgs[0]}"


@pytest.mark.asyncio
async def test_the_logged_head_is_bounded(caplog) -> None:
    """残すのは先頭だけ（長い説明でログを埋めない）。"""
    reply = "あ" * 5000
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await extract_entities("説明", _Backend(reply))

    msgs = [r.getMessage() for r in caplog.records]
    assert msgs
    assert len(msgs[0]) < 500, f"ログが長すぎる: {len(msgs[0])}字"


@pytest.mark.asyncio
async def test_a_good_reply_logs_nothing(caplog) -> None:
    """解釈できたときは鳴らさない（毎回鳴ると読まれなくなる）。"""
    reply = '{"entities": [{"label": "chair", "category": "object", "confidence": 0.9}]}'
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await extract_entities("説明", _Backend(reply))

    assert len(got) == 1 and got[0]["label"] == "chair"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], "成功で鳴っている"
