"""この求めの記録（起点・生きている版・見た印）は W に**全文**で載せる（2026-09-13 実機で露見）。

W の 1 行は 120 字で切っていた。求めの版は「調べた結果が届いた：…」を持つので、検索結果の
中身（『9月14日(月) 30℃/22℃ 40%』）が 120 字の外に落ち、調停は「詳細」を検索し直し、
主LLM は「数字は読み取れなかった」と答えた。`compose` の docstring は「1 件の途中では切らない」
と言っていたが、`_lines` が切っていた。過去の記憶は今までどおり 120 字。
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop import workspace
from familiar_agent.loop.request import Request

_RESULT = (
    "「明日の天気は？」と聞かれ、1番：search_deferred「2026年9月14日 東京 天気」の結果が届いた："
    "Title: 東京の今日・明日の天気｜2週間先までの1時間予報 - Toshin.com\n"
    "Description: 降水確率は · 50％のため... · 40% 明日 · 2026年 · 9月14日(月) 30℃/22℃ · 40%"
)
_OLD = "去年の夏、" + "海へ行った話。" * 30


def _recalled(obs_id, content, direction):
    mi = MI(
        id=obs_id, content=content, timestamp=datetime.now(), direction=direction, obs_id=obs_id
    )
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


def _oif():
    oif = MagicMock()
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    return oif


def test_the_live_version_keeps_its_full_content() -> None:
    req = Request()
    req.request_id = "起点1"
    req.live_version_id = "版1"
    req.turn_records = [("起点1", "起点"), ("版1", "版")]
    text, _ = workspace.compose(
        _oif(), [_recalled("版1", _RESULT, "求め"), _recalled("m9", _OLD, "会話")], req
    )
    assert "30℃/22℃" in text, "検索結果の中身が 120 字の外に落ちている"
    assert len(_OLD) > 120 and _OLD[:120] in text and _OLD[:140] not in text, "過去の記憶は 120 字"


def test_a_seen_mark_of_this_request_is_also_full() -> None:
    req = Request()
    req.request_id = "起点1"
    req.turn_records = [("起点1", "起点"), ("見た1", "見た")]
    long_mark = "出入り口を見た。見えたもの：" + "、".join(f"物{i}" for i in range(60))
    text, _ = workspace.compose(_oif(), [_recalled("見た1", long_mark, "観察")], req)
    assert "物59" in text
