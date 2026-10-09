"""道具（タイマー・アラーム）の帰りの反復（出-x・2026-09-18・`イベント駆動ループ` v0.84）。

実機 3 回（14:50・15:28・15:42）で、`set_timer` が「確かめて」を返した帰りに調停が掛け直し、空回りした。
実験（`根拠台帳` §35・3 種の実機 W × 8 回）：いまの形は 22/24 が `action`、
「想起なし＋返りを受ける先導文＋返った道具を候補から外す＋失敗の記録があっても道具で取り返さない」で 24/24 が light。

- 帰りの反復では**想起を回さない**（W＝返り＋直近）。検索・see の帰りは従来どおり。
- 先導文は実験と同じ文（`arbiter._LEAD_TOOL_RETURN`）。
- 返ってきた道具そのものは候補から外す（掛け直しは「いい」の側で機械が行う）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop import arbiter, workspace
from tests._arbiter_fakes import decide, jev_says, prompt_of, writer_says
from familiar_agent.loop.event_loop import InformationProcessing, Lookup
from familiar_agent.loop.request import Request

from tests.test_event_loop import _agent


def _req_after_timer_return():
    req = Request()
    req.lookups.append(
        Lookup(
            index=1,
            action="set_timer",
            query="タイマーを掛ける「x」",
            generation=0,
            result="まだ掛けていない。本人に一度聞く：「3 分のタイマーね、いい？」",
        )
    )
    req.just_returned.append(1)
    return req


def test_a_timer_return_skips_recall_and_keeps_recent():
    oif = MagicMock()
    oif.recall = AsyncMock(return_value=[])
    oif.latest_origins = MagicMock(return_value=[])
    oif.exchanges = MagicMock(return_value=[])
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    req = _req_after_timer_return()
    ws = asyncio.run(
        workspace.recall(oif, "手がかり", viewpoint="p", weights=(1, 1, 1, 1, 1), req=req)
    )
    oif.recall.assert_not_awaited()  # 想起なし
    text = ws.render(2)
    assert text.startswith("[いま道具から返った]") and "[過去の記憶" not in text


def test_a_search_return_still_recalls():
    oif = MagicMock()
    oif.recall = AsyncMock(return_value=[])
    oif.latest_origins = MagicMock(return_value=[])
    oif.exchanges = MagicMock(return_value=[])
    oif.actors = MagicMock(return_value={})
    oif.roles = MagicMock(return_value={})
    req = Request()
    req.lookups.append(
        Lookup(index=1, action="search_deferred", query="金木犀", generation=0, result="…")
    )
    req.just_returned.append(1)
    asyncio.run(workspace.recall(oif, "手がかり", viewpoint="p", weights=(1, 1, 1, 1, 1), req=req))
    oif.recall.assert_awaited_once()


def test_the_lead_is_the_experiments_sentence_and_the_returned_tool_is_not_offered():
    assert "取り返そうとして道具を選ばない" in arbiter._LEAD_TOOL_RETURN
    assert "掛け直しはあなたの仕事ではない" in arbiter._LEAD_TOOL_RETURN
    a = _agent(stream_returns=[])
    a._timer_tool = MagicMock()
    a._timer_tool.get_tool_definitions = MagicMock(
        return_value=[
            {"name": n}
            for n in ("set_timer", "cancel_timer", "pause_timer", "resume_timer", "start_stopwatch")
        ]
    )
    a._alarm_tool = None
    a._mcp = None
    ip = InformationProcessing(a)
    ip._req = _req_after_timer_return()
    ip._gated = lambda defs: defs
    ip._ACTIONS = {
        **{
            k: (lambda ip, k=k: [])
            for k in (
                "house_rules",
                "family_schedule",
                "notion_search",
                "journal",
                "vault",
                "set_alarm",
                "cancel_alarm",
                "confirm",
                "decline",
                "stop_stopwatch",
                "play_music",  # 音楽は器が無い（知-ak 段 6 で調停の候補に載った）
                "stop_music",
                "next_track",
                "music_volume",
                "music_suggestion_reply",  # すすめた曲への返事（出-ay 段 4-4f で調停の候補を通る）
            )
        },
        **{
            k: (lambda ip, k=k: [{"name": k}])
            for k in ("set_timer", "cancel_timer", "pause_timer", "resume_timer", "start_stopwatch")
        },
    }
    returned = workspace.returned_actions(ip._req)
    assert returned == {"set_timer"} and returned & workspace.RETURN_WITHOUT_RECALL
    acts = set(ip._extra_actions(exclude=returned))
    assert "set_timer" not in acts and "cancel_timer" in acts
    assert "set_timer" in ip._extra_actions()  # 外すのは返った反復だけ


def test_the_arbiter_uses_the_tool_return_lead_when_asked():
    """道具の帰りの先導文は、Jev に送る文と軽量LLM の文章の口の両方に載る。"""
    from familiar_agent.loop.arbiter import Arbiter, ArbiterInput

    def written(**kw) -> str:
        writer = writer_says({"text": "3 分ね、いい？"})
        d = asyncio.run(decide(jev=jev_says("light"), writer=writer, utterance="x", **kw))
        assert d.branch == "light"
        return prompt_of(writer)

    assert "取り返そうとして道具を選ばない" in written(tool_return=True)
    assert "取り返そうとして道具を選ばない" not in written()
    state = Arbiter(jev=None, writer=None)._state(
        ArbiterInput(utterance="x", workspace_ctx="", tool_return=True)
    )
    assert "取り返そうとして道具を選ばない" in state


def test_the_needs_tools_guard_does_not_fire_on_a_tool_return():
    """出-x-ろ（実機 18:31）：守り（「測って」は light で答えない）は道具が返った反復では掛けない。"""

    def light(**kw):
        writer = writer_says({"text": "11 分のタイマーね、いい？"})
        return asyncio.run(
            decide(jev=jev_says("light"), writer=writer, utterance="パジュ、1１分測って", **kw)
        )

    ws = "[いま道具から返った]\n- set_timer「タイマーを掛ける「x」」→ まだ掛けていない。本人に一度聞く：「…」"
    assert light(workspace_ctx=ws, tool_return=True).branch == "light"
    assert light().branch == "full"  # 初回の反復の守りは残る
