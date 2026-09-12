"""独り言は相手が居なければ発話せず、積まずに捨て、言わなかった思いとして残す（情-c）。

欲求で起きた求め（起点 `情動`）は、配信ゲートが閉じているとき `pending_speech` へ積まない
（その場に居なければ無かったことになる）。ただし本応答は役割 `独白` で O に残る（`_finish` の
既存経路）。不在の独り言には写真も添えない（費用の 4 割がここだった）。人の発話・機器が
起点の求めは今までどおり保留する。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.io.oif import MI, Recalled
from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


def _ip(*, trigger_kind: str, blocked: str):
    a = _agent(stream_returns=[])
    ip = InformationProcessing(a)
    ip._req.trigger_kind = trigger_kind
    ip._delivery_block_reason = lambda: blocked  # type: ignore[method-assign]
    ip._hold_speech = AsyncMock()  # type: ignore[method-assign]
    a._dif = MagicMock()
    a._dif.speak = AsyncMock()
    ip._dif = a._dif
    return ip, a


def test_a_blocked_monologue_is_not_held_and_ends_as_monologue() -> None:
    async def scenario():
        ip, a = _ip(trigger_kind="情動", blocked="聞く相手が居ない")
        got = await ip._speak("部屋が静かだね。")
        await ip.close()
        return got, ip._hold_speech.await_count, a._dif.speak.await_count

    got, held, spoke = asyncio.run(scenario())
    assert got == ("部屋が静かだね。", "独白"), "本文は残し、結末は独白"
    assert held == 0, "pending_speech へ積まない"
    assert spoke == 0


def test_a_blocked_monologue_is_recorded_as_a_thought_not_said() -> None:
    """`_finish` は結末が発話でない本文を「考えたが言わなかった」（役割 独白）で書く。"""

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
    assert ("考えたが言わなかった：部屋が静かだね。", "独白") in written
    assert not any(d == "保留" for _, d in written)


def test_a_blocked_reply_to_a_person_is_still_held() -> None:
    async def scenario():
        ip, a = _ip(trigger_kind="発話", blocked="聞く相手が居ない")
        got = await ip._speak("おはよう")
        await ip.close()
        return got, ip._hold_speech.await_count

    got, held = asyncio.run(scenario())
    assert got == ("", "保留") and held == 1


def test_any_block_reason_silences_a_monologue() -> None:
    for reason in ("黙っているよう頼まれている", "静穏時間である"):

        async def scenario(reason=reason):
            ip, a = _ip(trigger_kind="情動", blocked=reason)
            got = await ip._speak("ひとりごと")
            await ip.close()
            return got, ip._hold_speech.await_count

        got, held = asyncio.run(scenario())
        assert got[1] == "独白" and held == 0, reason


def _recalled(obs_id, image_path):
    mi = MI(
        id=obs_id,
        content="見た",
        timestamp=datetime.now(),
        direction="観察",
        obs_id=obs_id,
        image_path=image_path,
    )
    return Recalled(mi=mi, fit=0.5, groundedness=0.5, confidence=0.6)


def test_an_absent_monologue_gets_no_photo(tmp_path, caplog) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="情動", blocked="聞く相手が居ない")
    ip._req.turn_records = [("起点", "起点"), ("見た1", "見た")]
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        out = ip._user_content("x", [_recalled("見た1", str(path))])
    assert out == "x"
    assert any("写真を添えない" in r.getMessage() for r in caplog.records)


def test_a_monologue_with_someone_present_keeps_the_photo(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="情動", blocked="")
    ip._req.turn_records = [("起点", "起点"), ("見た1", "見た")]
    assert isinstance(ip._user_content("x", [_recalled("見た1", str(path))]), list)


def test_a_reply_to_a_person_keeps_the_photo_even_if_blocked(tmp_path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"JPEG")
    ip, _ = _ip(trigger_kind="発話", blocked="聞く相手が居ない")
    ip._req.turn_records = [("起点", "起点"), ("見た1", "見た")]
    assert isinstance(ip._user_content("x", [_recalled("見た1", str(path))]), list)
