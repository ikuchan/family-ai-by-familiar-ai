"""端-b 段 1-ろ-4：端末の言葉を受け、返事を送り返す待ち受けの口（`io/device_link`・課題9 §6.6）。

- 端末は `/device?name=端末の名前` に WebSocket でつなぐ。名前の無い端末はつながせない（空は家の印）。
- つないだら声の設定（合成のモデルと声の種類）を送る。鍵は送らない。
- 端末から `{"type": "say", "text": …, "speaker": …}` が届いたら、端末の言葉として入口へ流す。読む輪は入口を待たない。
- 返事（say・cue・closed・refused）は、その端末へ、受け手に渡った順に送る。
- 読めない字・知らない種類・長すぎる字は捨て、つながりは切らない。同じ名前の 2 本目がつながれば、古いほうを閉じる。
- 待ち受けの番地を取れなくても、起動は止めない。

動いているパジュの口には何も送らない。試験はポート 0 で自分の口を立てる。
"""

from __future__ import annotations

import asyncio
import json

from unittest.mock import MagicMock

import aiohttp
import pytest

from familiar_agent.io.device_link import DeviceLink

VOICE = {"model": "m-1", "voice": "v-1"}


class _Pushes:
    def __init__(self, block: bool = False) -> None:
        self.calls: list = []
        self.block = block

    async def __call__(self, device: str, text: str, speaker: str) -> None:
        self.calls.append((device, text, speaker))
        if self.block:
            await asyncio.Event().wait()  # 最初の反復が終わらない入口


def _run(body, push=None):
    async def go():
        link = DeviceLink(push or _Pushes(), VOICE, host="127.0.0.1", port=0, max_chars=20)
        assert await link.start()
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    body(link, session, f"http://127.0.0.1:{link.port}"), 5.0
                )
        finally:
            await link.close()

    return asyncio.run(go())


async def _recv(ws) -> dict:
    return json.loads((await asyncio.wait_for(ws.receive(), 2.0)).data)


async def _settle(cond) -> None:
    for _ in range(100):
        if cond():
            return
        await asyncio.sleep(0.01)


def test_a_device_without_a_name_cannot_connect():
    async def body(link, session, url):
        for q in ("", "?name=", "?name=%20%20"):
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(f"{url}/device{q}")

    _run(body)


def test_the_device_gets_the_voice_settings_and_no_key_on_connect():
    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        hello = await _recv(ws)
        assert hello["type"] == "hello"
        assert sorted(hello["voice"]) == ["model", "voice"]
        assert hello["voice"]["model"] == "m-1"

    _run(body)


def test_a_device_word_goes_to_the_entrance_with_its_speaker():
    push = _Pushes()

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        await ws.send_json({"type": "say", "text": " 明日の天気は？ ", "speaker": "ママ"})
        await _settle(lambda: push.calls)

    _run(body, push)
    assert push.calls == [("tab", "明日の天気は？", "ママ")]


def test_the_read_loop_does_not_wait_for_the_entrance():
    """入口（`push_utterance`）は最初の反復まで返らない。待つと、次の言葉も切断も読めない。"""
    push = _Pushes(block=True)

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        await ws.send_json({"type": "say", "text": "一つ目"})
        await ws.send_json({"type": "say", "text": "二つ目"})
        await _settle(lambda: len(push.calls) == 2)

    _run(body, push)
    assert [c[1] for c in push.calls] == ["一つ目", "二つ目"]


def test_replies_reach_the_device_in_the_order_they_were_handed():
    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        link.send("tab", "cue", {"kind": "a"})
        link.send("tab", "say", {"text": "晴れ", "display": "晴れ"})
        link.send("tab", "closed", {})
        link.send("other", "say", {"text": "x"})  # つながっていない端末へは捨てる
        return [await _recv(ws) for _ in range(3)]

    got = _run(body)
    assert [m["type"] for m in got] == ["cue", "say", "closed"]
    assert got[1]["text"] == "晴れ"


def test_unreadable_or_unknown_or_too_long_words_are_dropped_and_the_link_stays():
    push = _Pushes()

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        await ws.send_str("{壊れた")
        await ws.send_json(["say"])
        await ws.send_json({"type": "dance", "text": "x"})
        await ws.send_json({"type": "say", "text": "あ" * 21})
        await ws.send_json({"type": "say", "text": "   "})
        await ws.send_json({"type": "say", "text": 3})
        await ws.send_json({"type": "say", "text": "残る", "speaker": 5})
        await _settle(lambda: push.calls)
        assert not ws.closed

    _run(body, push)
    assert push.calls == [("tab", "残る", "")]


def test_a_dropped_say_is_answered_refused_so_the_device_does_not_wait():
    """捨てた言葉（空・長すぎる・字でない）には refused を返す（端-c c1 の見直し）。返さないと、端末は返事を待ち続ける。
    言葉でない字（読めない・知らない種類）には返さない。"""

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        await ws.send_str("{壊れた")
        await ws.send_json({"type": "dance"})
        await ws.send_json({"type": "say", "text": "あ" * 21})
        await ws.send_json({"type": "say", "text": "   "})
        await ws.send_json({"type": "say", "text": 3})
        return [await _recv(ws) for _ in range(3)]

    assert _run(body) == [{"type": "refused"}] * 3


def test_a_second_link_with_the_same_name_replaces_the_first():
    async def body(link, session, url):
        old = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(old)
        new = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(new)
        msg = await asyncio.wait_for(old.receive(), 2.0)
        assert msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED)
        link.send("tab", "closed", {})
        assert (await _recv(new))["type"] == "closed"

    _run(body)


def test_a_taken_port_does_not_stop_the_start():
    async def go():
        first = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=0)
        assert await first.start()
        second = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=first.port)
        try:
            assert await second.start() is False
        finally:
            await second.close()
            await first.close()

    asyncio.run(go())


def test_closing_twice_is_fine():
    async def go():
        link = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=0)
        await link.start()
        await link.close()
        await link.close()

    asyncio.run(go())


def _agent(link_config):
    from unittest.mock import AsyncMock, MagicMock

    from familiar_agent.agent import EmbodiedAgent

    a = EmbodiedAgent.__new__(EmbodiedAgent)
    a.config = MagicMock()
    a.config.tts.gemini_model = "m-1"
    a.config.tts.gemini_voice = "v-1"
    a.config.tts.gemini_api_key = "秘密"
    a._device_link_config = link_config
    a._info_processing = MagicMock(push_utterance=AsyncMock(return_value=""))
    a._ensure_event_loop = MagicMock()
    a._camera = None
    a._occupancy_sensor = None
    a._motion_events = None
    return a


def test_the_agent_opens_the_link_at_start_and_registers_it_as_the_device_listener():
    from familiar_agent.config import DeviceLinkConfig

    async def go():
        a = _agent(DeviceLinkConfig(port=0))
        await a.start_autonomy()
        await a.start_autonomy()  # 二重には立たない
        link = a._device_link
        try:
            assert link.port
            a._info_processing.set_device_listener.assert_called_with(link.send)
            assert link._voice == {"model": "m-1", "voice": "v-1"}
        finally:
            await a._close_device_link()
        assert a._device_link is None

    asyncio.run(go())


def test_the_agent_tells_devices_the_names_and_the_family_call_names():
    """hello に、パジュの呼び名と、話者のボタンに並べる家族の呼び名（FAMILY.md・1 人 1 つ）を添える（端-c c2）。"""
    from familiar_agent.config import DeviceLinkConfig

    async def go():
        a = _agent(DeviceLinkConfig(port=0))
        a.config.agent_names = ["パジュ"]
        a._family_md = "## ママ\n- **名前**：ママ\n- **呼び方**：ママ\n## パパ\n- **名前**：パパ\n- **呼び方**：パパ\n"
        await a.start_autonomy()
        link = a._device_link
        try:
            first = link._names(), link._speakers()
            a._family_md += (
                "## むすめ\n- **名前**：むすめ\n- **呼び方**：むすめ\n"  # /reload で読み直した
            )
            return first, link._speakers()
        finally:
            await a._close_device_link()

    first, reloaded = asyncio.run(go())
    assert first == (["パジュ"], ["ママ", "パパ"])
    assert reloaded == ["ママ", "パパ", "むすめ"]  # つなぐたびに読む（読み直した家族も並ぶ）


def test_no_link_when_it_is_turned_off():
    async def go():
        a = _agent(None)
        await a.start_autonomy()
        assert getattr(a, "_device_link", None) is None

    asyncio.run(go())


def test_a_device_word_goes_straight_to_the_entrance_without_reading_home_commands():
    """端末の字が `/speaker …` や `[名前] …` で始まっても、家の話者の命令として読まない（b3）。"""
    a = _agent(None)
    a.set_speaker = MagicMock()

    asyncio.run(a.push_device_utterance("tab", "/speaker パパ", "ママ"))
    asyncio.run(a.push_device_utterance("tab", "[パパ] おはよう", ""))
    calls = a._info_processing.push_utterance.await_args_list
    assert [c.args[0] for c in calls] == ["/speaker パパ", "[パパ] おはよう"]
    assert calls[0].kwargs == {"source": "device", "device": "tab", "speaker": "ママ"}
    a.set_speaker.assert_not_called()


def test_a_browser_page_cannot_connect():
    """ブラウザのページは Origin を付けてつなぐ。家族の端末のアプリは付けないので、Origin のあるつなぎは断る（仮）。"""

    async def body(link, session, url):
        with pytest.raises(aiohttp.WSServerHandshakeError):
            await session.ws_connect(f"{url}/device?name=tab", origin="http://evil.example")

    _run(body)


def test_a_device_cannot_pile_up_words_without_limit():
    """入口を待つ言葉は、1 台につき 3 つまで（仮）。それより多い言葉は断ったと返す。"""
    push = _Pushes(block=True)

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        for i in range(4):
            await ws.send_json({"type": "say", "text": f"言葉{i}"})
        assert (await _recv(ws))["type"] == "refused"

    _run(body, push)
    assert len(push.calls) == 3


def test_closing_leaves_no_send_pending():
    async def go():
        link = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=0)
        await link.start()
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"http://127.0.0.1:{link.port}/device?name=tab")
            await _recv(ws)
            for _ in range(5):
                link.send("tab", "cue", {"kind": "a"})
            tasks = set(link._links["tab"].tasks)
            await link.close()
        assert all(t.done() for t in tasks)

    asyncio.run(go())


def test_reconnecting_does_not_reset_the_word_limit():
    push = _Pushes(block=True)

    async def body(link, session, url):
        ws = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(ws)
        for i in range(3):
            await ws.send_json({"type": "say", "text": f"言葉{i}"})
        await _settle(lambda: len(push.calls) == 3)
        again = await session.ws_connect(f"{url}/device?name=tab")
        await _recv(again)
        await again.send_json({"type": "say", "text": "もう一つ"})
        assert (await _recv(again))["type"] == "refused"

    _run(body, push)
    assert len(push.calls) == 3


def test_the_last_replies_are_sent_before_closing():
    async def go():
        link = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=0)
        await link.start()
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(f"http://127.0.0.1:{link.port}/device?name=tab")
            await _recv(ws)
            link.send("tab", "say", {"text": "またね", "display": "またね"})
            link.send("tab", "closed", {})
            await link.close()
            kinds = []
            while True:
                msg = await asyncio.wait_for(ws.receive(), 2.0)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                kinds.append(json.loads(msg.data)["type"])
        assert kinds == ["say", "closed"]

    asyncio.run(go())


def test_a_failing_link_does_not_stop_the_start():
    from unittest.mock import AsyncMock, patch

    from familiar_agent.config import DeviceLinkConfig

    async def go():
        a = _agent(DeviceLinkConfig(port=0))
        a._occupancy_sensor = MagicMock(start=AsyncMock())
        with patch(
            "familiar_agent.io.device_link.DeviceLink.start", AsyncMock(side_effect=RuntimeError)
        ):
            await a.start_autonomy()
        a._occupancy_sensor.start.assert_awaited()
        assert getattr(a, "_device_link", None) is None

    asyncio.run(go())


# ── 合言葉（端-c・家の Wi-Fi から端末をつなぐ・課題9 §8.10） ──
KEY = "k" * 16


def _run_keyed(body, key=KEY):
    async def go():
        link = DeviceLink(_Pushes(), VOICE, host="127.0.0.1", port=0, key=key)
        assert await link.start()
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    body(link, session, f"http://127.0.0.1:{link.port}"), 5.0
                )
        finally:
            await link.close()

    return asyncio.run(go())


@pytest.mark.parametrize("key", ["", "short-key", "あいことば" * 4])
def test_without_a_long_enough_key_the_link_stays_inside_the_machine(key):
    """合言葉が無いか 16 字に満たないか ASCII でなければ（端末は頭書きに日本語を書けない）、外で待ち受ける番地を渡されても、機体の中（127.0.0.1）だけで待ち受ける。"""
    link = DeviceLink(_Pushes(), VOICE, host="0.0.0.0", port=0, key=key)
    assert link.host == "127.0.0.1"


def test_with_a_key_the_link_listens_where_it_is_told():
    """**反証の側**：合言葉があれば、渡された番地（家の Wi-Fi から入れる 0.0.0.0）で待ち受ける。"""
    assert DeviceLink(_Pushes(), VOICE, host="0.0.0.0", port=0, key=KEY).host == "0.0.0.0"


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer " + "x" * 16}, {"Authorization": KEY}]
)
def test_a_device_without_the_key_cannot_connect(headers):
    async def body(link, session, url):
        with pytest.raises(aiohttp.WSServerHandshakeError) as e:
            await session.ws_connect(f"{url}/device?name=tab", headers=headers)
        assert e.value.status == 401

    _run_keyed(body)


def test_a_device_with_the_key_connects():
    async def body(link, session, url):
        ws = await session.ws_connect(
            f"{url}/device?name=tab", headers={"Authorization": f"Bearer {KEY}"}
        )
        assert (await _recv(ws))["type"] == "hello"
        await ws.close()

    _run_keyed(body)


def test_the_key_comes_from_the_env_and_the_link_opens_to_the_home_network(monkeypatch):
    from familiar_agent.config import DeviceLinkConfig

    monkeypatch.setenv("DEVICE_LINK_KEY", KEY)
    cfg = DeviceLinkConfig()
    got = cfg.key  # 落ちたときに鍵を丸ごと出さない
    assert got == KEY and cfg.host == "0.0.0.0"
