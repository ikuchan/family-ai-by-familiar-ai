"""「一度きり」を真偽値でなく、カーソル自身で表す（環-g・段へ）。

以前は `_show_seeded` という真偽値を持っていた。

    if not self._show_seeded:
        self._show_seeded = True
        self._recent_cursor = agent._memory.latest_exchange_origin()

**一度立つと二度と戻らない。** `_recent_cursor` が `None` のまま真偽値が立つと（DB に
やりとりが1件も無い・引くのに失敗した）、次に `_close_exchange` が値を入れるまで直近の
やりとりは載らない。「一度きり」を真偽値で表しているのは、**状態機械が真偽値へ潰れた跡**
である（Harel, *Statecharts*・`モジュール分割設計`）。

**カーソルが `None` かどうかで判定すれば、変数が1つ減る。** そのぶん、DB が空のあいだは
毎ターン引きに行く（挙動の変化）。1件でもあれば一度で埋まり、以後は `_close_exchange` が
更新するので引き直しは起きない。
"""

from __future__ import annotations

import io
import re
import tokenize
from unittest.mock import MagicMock
from pathlib import Path

_LOOP = Path(__file__).parent.parent / "src/familiar_agent/loop/event_loop.py"


def _code_only() -> str:
    out: list[str] = []
    with open(_LOOP, "rb") as f:
        for tok in tokenize.tokenize(io.BytesIO(f.read()).readline):
            if tok.type not in (tokenize.COMMENT, tokenize.STRING):
                out.append(tok.string)
    return " ".join(out)


def _ip(cursor=None, latest="past-1", rows=None):
    from familiar_agent.loop.event_loop import InformationProcessing
    from familiar_agent.loop.request import Request

    ip = InformationProcessing.__new__(InformationProcessing)
    # `__new__` は `__init__` を通らないので、求めの器は自分で置く（に-5-に-1）。
    ip._req = Request()
    ip._recent_cursor = cursor
    ip._req.request_id = "req-1"
    a = MagicMock()
    a._memory.latest_exchange_origin = MagicMock(return_value=latest)
    a._memory.recent_exchanges = MagicMock(return_value=rows or [])
    ip._agent = a
    return ip, a


def test_the_boolean_is_gone():
    assert not re.search(r"\b_show_seeded\b", _code_only())


def test_an_empty_cursor_is_filled_from_the_store():
    ip, a = _ip(cursor=None)
    ip._recent_ctx("follows-1", {})
    a._memory.latest_exchange_origin.assert_called_once()
    assert ip._recent_cursor == "past-1"


def test_a_filled_cursor_is_not_looked_up_again():
    """1件でもあれば一度で埋まり、以後は引き直さない。"""
    ip, a = _ip(cursor="already")
    ip._recent_ctx("follows-1", {})
    a._memory.latest_exchange_origin.assert_not_called()


def test_an_empty_store_is_tried_again_next_turn():
    """**挙動の変化。** 以前は一度きりで、空だと二度と引かなかった。"""
    ip, a = _ip(cursor=None, latest=None)
    ip._recent_ctx("follows-1", {})
    ip._recent_ctx("follows-1", {})
    assert a._memory.latest_exchange_origin.call_count == 2
