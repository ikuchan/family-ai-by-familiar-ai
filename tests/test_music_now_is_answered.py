"""「いま何の曲かけてる？」に、いま鳴っている曲を答える（出-bf・2026-10-10・実機 11:07:17・本人の決定ア）。

曲名は O に残っていたのに、意味「音楽に関する依頼」の 2 回目に「いま鳴っている曲を伝える」が無く、聞き返して「確認いたし
ますので少々お待ちください」で終わった。2 回目に `music_now` を足し、選ばれたら機械が決まった形で答える（曲名を取り違えず、
軽量LLM を待たない）。合図は効果音 B。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.backends.jev import JevAnswer
from familiar_agent.core import reaction_cue as rc
from familiar_agent.core import utterance_meaning as um
from familiar_agent.core.music_now import text_for
from tests._arbiter_fakes import decide


@pytest.mark.parametrize(
    ("status", "text"),
    [
        (
            {"playing": True, "title": "Plazma", "artist": "米津玄師"},
            "いまは『Plazma』、米津玄師だよ",
        ),
        ({"playing": True, "title": "Plazma", "artist": ""}, "いまは『Plazma』だよ"),
        ({}, "いまは何も鳴ってないよ"),
        ({"playing": False, "title": "Plazma", "artist": "米津玄師"}, "いまは何も鳴ってないよ"),
        (None, "いま何が鳴ってるか、分からなかった"),
    ],
)
def test_the_answer_is_made_from_the_status(status, text):
    assert text_for(status) == text


def test_music_now_is_an_action_of_the_music_meaning():
    assert "music_now" in um.ACTIONS_BY_MEANING["music"]
    assert rc.cue_for("music_now") == rc.ACK


def test_the_arbiter_answers_without_the_writer():
    jev = MagicMock()
    jev.available = True
    jev.ask = AsyncMock(
        return_value=JevAnswer(
            ok=True,
            answers={
                "meaning": {"choice": "music", "confidence": 0.9},
                "action_music": {"choice": "music_now", "confidence": 0.9},
            },
        )
    )
    writer = MagicMock()
    writer.complete = AsyncMock(return_value='{"text": "書いた"}')
    told: list[str] = []

    async def status():
        return {"playing": True, "title": "Plazma", "artist": "Kenshi Yonezu"}

    d = asyncio.run(
        decide(
            jev=jev,
            writer=writer,
            utterance="パジュー、今何の曲かけてる?",
            origin="発話",
            extra_actions=("play_music", "stop_music"),
            music_status=status,
            on_decided=told.append,
        )
    )
    assert (d.branch, d.text) == ("light", "いまは『Plazma』、Kenshi Yonezuだよ")
    writer.complete.assert_not_awaited()
    assert told == ["music_now"]


def test_the_loop_reads_the_status_once_for_both_uses():
    """主LLM の `[音楽]` の枠と調停の答えは、同じ読み方（`_music_status`）を使う。"""
    import inspect

    from familiar_agent.loop.event_loop import InformationProcessing

    assert "_music_status" in inspect.getsource(InformationProcessing._music_now)
