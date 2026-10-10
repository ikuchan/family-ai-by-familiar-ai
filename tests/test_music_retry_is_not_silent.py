"""音楽のかけ直しが、黙って「かけた」扱いにならない（出-be・2026-10-10・実機 11:05:16・本人の決定ア）。

「プレイリスト待機をかけて」で別の曲が鳴り（「別の曲が鳴っている」）、調停がかけ直しを選んだ。音楽の道具には見出しが無く
（`_query_label` が空文字）、同じ求めの 2 回目の音楽の操作は名前に関係なく「すでに調べた語なので投げない」で止まった。
投げなかったことを知らせる完了には失敗の印が無く、前の失敗の印を「失敗していない」で上書きし、完了の表が「音楽をかけた
→ 黙る」と読んで、違う曲が鳴ったまま黙った。

直し方：音楽の道具に見出しを付ける（名前が違えば別の操作として投げる）。投げなかった完了は前の失敗の印を引き継ぐ。
「この求めですでに調べた」で返ってきた完了は、道具に関係なく軽く伝える（同じ名前のかけ直しが上限まで続かないように）。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core.completion_kind import kind_of
from familiar_agent.loop.event_loop import _query_label
from familiar_agent.loop.request import Lookup

# ── 見出し ───────────────────────────────────────────────────────────────────


def test_play_music_is_labelled_by_name():
    assert _query_label("play_music", {"name": "待機"}) == "音楽をかける「待機」"
    assert _query_label("play_music", {"name": "たいき"}) != _query_label(
        "play_music", {"name": "待機"}
    )
    assert _query_label("play_music", {}) == "音楽をかける（止まっていた続き）"


@pytest.mark.parametrize(
    ("action", "tool_input"),
    [
        ("stop_music", {}),
        ("next_track", {}),
        ("music_volume", {"how": "大きく"}),
        ("music_suggestion_reply", {"reply": "いらない"}),
    ],
)
def test_every_music_tool_has_a_label(action, tool_input):
    assert _query_label(action, tool_input)


# ── 投げるかどうか・失敗の印 ─────────────────────────────────────────────────────


def _ip_with_failed_play():
    from familiar_agent.loop.event_loop import InformationProcessing
    from tests.test_event_loop import _agent

    ip = InformationProcessing(_agent(stream_returns=[]))
    first = Lookup(
        index=1,
        action="play_music",
        query="音楽をかける「待機」",
        generation=ip._request_generation,
    )
    first.result = "「待機」をかけようとしたが、別の曲が鳴っている"
    first.failed = True
    ip._req.lookups.append(first)
    return ip


def test_another_name_is_played():
    ip = _ip_with_failed_play()

    async def go():
        ip._run_lookup = lambda *a, **k: asyncio.sleep(0)  # type: ignore[method-assign]
        ip._dispatch_lookup("play_music", {"name": "たいき"}, "音楽をかける「たいき」", None)
        n = len(ip._req.lookups)
        await ip.close()
        return n

    assert asyncio.run(go()) == 2


def test_the_same_name_is_not_played_again_and_stays_a_failure():
    ip = _ip_with_failed_play()

    async def go():
        ip._dispatch_lookup("play_music", {"name": "待機"}, "音楽をかける「待機」", None)
        t = ip._triggers.get_nowait()
        await ip.close()
        return t

    t = asyncio.run(go())
    assert len(ip._req.lookups) == 1
    assert t.kind == "完了" and "すでに調べた" in t.result
    assert t.failed is True  # 前の失敗を引き継ぐ


# ── 完了の表 ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("failed", [True, False])
def test_an_already_tried_result_is_told_lightly(failed):
    result = "「音楽をかける「待機」」はこの求めですでに調べた。前の結果：別の曲が鳴っている"
    got = kind_of("play_music", failed=failed, result=result, origin="発話")
    assert got is not None and got[1] == ("tell_light",)
