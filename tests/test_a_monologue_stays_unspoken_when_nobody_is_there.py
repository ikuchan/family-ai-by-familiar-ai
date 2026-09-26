"""独り言は相手が居なければ発話せず、積まずに捨て、言わなかった思いとして残す（情-c）。

配信ゲートが閉じているとき、発話は保留に積まない（積む口そのものを出-as 段 9b で外した）。本応答は
役割 `独白` で O に残る（`_finish` の既存経路）。不在の独り言には写真も添えない（費用の 4 割がここだった）。
ここでは「保留」の役割の記録が 1 件も書かれないことを数える。
"""

from __future__ import annotations

import time
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, trigger_kind: str, blocked: str):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = trigger_kind
    ip._req.fired_axis = "bond"  # 情動なら話しかける軸（出-as 段 8：話すのは bond・esteem だけ）
    ip._delivery_block_reason = lambda: blocked  # type: ignore[method-assign]
    a._dif = MagicMock()
    a._dif.speak = AsyncMock()
    ip._dif = a._dif
    ip._wake_window().open(time.monotonic())  # 入口を通った会話として窓を開けておく（出-as 段 4）
    return ip, a


def test_a_blocked_monologue_is_not_held_and_ends_as_monologue() -> None:
    async def scenario():
        ip, a = _ip(trigger_kind="情動", blocked="聞く相手が居ない")
        got = await ip._speak("部屋が静かだね。")
        await ip.close()
        return (
            got,
            sum(
                c.kwargs.get("direction") == "保留"
                for c in ip._agent._memory.save_async_with_id.call_args_list
            ),
            a._dif.speak.await_count,
        )

    got, held, spoke = asyncio.run(scenario())
    assert got == ("部屋が静かだね。", "独白"), "本文は残し、結末は独白"
    assert held == 0, "pending_speech へ積まない"
    assert spoke == 0


def test_a_blocked_monologue_is_recorded_as_a_thought_not_said() -> None:
    """話そうとして止められた本文は「〇〇に言いたかったこと」（役割 独白）で書く（出-as 段 7・2026-09-26）。

    以前は「考えたが言わなかった」だった。声にしなかっただけの独り言（結末「沈黙」）はいまもそう書く。
    """

    async def scenario():
        ip, a = _ip(trigger_kind="情動", blocked="聞く相手が居ない")
        text, outcome = await ip._speak("部屋が静かだね。")
        await ip._finish(text, [], outcome)
        await ip.close()
        return [
            (c.args[0], c.kwargs.get("direction"))
            for c in a._memory.save_async_with_id.call_args_list
        ]

    written = asyncio.run(scenario())
    assert any(
        str(w).endswith("に言いたかったこと：部屋が静かだね。") and d == "独白" for w, d in written
    )
    assert not any(d == "保留" for _, d in written)


def test_a_blocked_utterance_is_a_monologue_not_held() -> None:
    """止められた発話はすべて独り言（出-as §2.6・2026-09-26）。以前は人への返事だけ保留に積んでいた。"""

    async def scenario():
        ip, a = _ip(trigger_kind="発話", blocked="聞く相手が居ない")
        got = await ip._speak("おはよう")
        await ip.close()
        return got, sum(
            c.kwargs.get("direction") == "保留"
            for c in ip._agent._memory.save_async_with_id.call_args_list
        )

    got, held = asyncio.run(scenario())
    assert got == ("おはよう", "独白") and held == 0


def test_any_block_reason_silences_a_monologue() -> None:
    for reason in ("黙っているよう頼まれている", "静穏時間である"):

        async def scenario(reason=reason):
            ip, a = _ip(trigger_kind="情動", blocked=reason)
            got = await ip._speak("ひとりごと")
            await ip.close()
            return got, sum(
                c.kwargs.get("direction") == "保留"
                for c in ip._agent._memory.save_async_with_id.call_args_list
            )

        got, held = asyncio.run(scenario())
        assert got[1] == "独白" and held == 0, reason


def test_an_absent_monologue_gets_no_photo(tmp_path, caplog) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="情動", blocked="聞く相手が居ない")
    ip._req.seen_image_path = str(path)
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        out = ip._user_content("x", [])
    assert isinstance(out, str) and out.startswith("[いま湧いたこと]")  # 文字だけ（情-e の枠つき）
    assert any("写真を添えない" in r.getMessage() for r in caplog.records)


def test_a_monologue_with_someone_present_keeps_the_photo(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="情動", blocked="")
    ip._req.seen_image_path = str(path)
    assert isinstance(ip._user_content("x", []), list)


def test_a_reply_to_a_person_keeps_the_photo_even_if_blocked(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="発話", blocked="聞く相手が居ない")
    ip._req.seen_image_path = str(path)
    assert isinstance(ip._user_content("x", []), list)
