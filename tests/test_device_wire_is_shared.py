"""端-c 段 c1：機体とアプリが、同じ口の形の見本（`tests/data/device_wire.json`）を読む（課題9 §6.6・§8）。

アプリの試験（`android/core` の `WireFixtureTest`）も同じ見本を読む。どちらかが口の形を変えれば、もう一方の試験が落ちる。

- 機体 → 端末：端末の口が送る字が、見本の字と 1 字も違わない。hello には、名前を呼ぶ合図に使う呼び名（`names`）と、
  話者のボタンに並べる家族の呼び名（`speakers`）を添える（端-c c2・アプリに家族の名前を書かない）。
- 端末 → 機体：アプリが作る字（見本）を、端末の口が言葉と話者として受ける。
- 名前の合図：アプリの聞き分け（見本の `heard`）が、機体の門の文頭の名前の決まりと同じ。
- 会話が続くあいだ：アプリが名前なしの声を送るあいだ（見本の `window_sec`）が、機体の窓の長さと同じ（端-c c3）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiohttp

from familiar_agent.core.silence_rules import names_me
from familiar_agent.core.wake_strictness import is_name_only, is_name_repeat
from familiar_agent.io.device_link import DeviceLink

WIRE = json.loads((Path(__file__).parent / "data" / "device_wire.json").read_text("utf-8"))
TAB = "tablet-1"


def _run(body, push=None):
    async def noop(device, text, speaker):
        return None

    async def go():
        link = DeviceLink(
            push or noop,
            {"model": "m", "voice": "v"},
            host="127.0.0.1",
            port=0,
            names=lambda: WIRE["names"],
            speakers=lambda: ["ママ", "パパ"],
        )
        assert await link.start()
        try:
            async with aiohttp.ClientSession() as session:
                return await asyncio.wait_for(
                    body(link, session, f"http://127.0.0.1:{link.port}"), 5.0
                )
        finally:
            await link.close()

    return asyncio.run(go())


async def _raw(ws) -> str:
    return (await asyncio.wait_for(ws.receive(), 2.0)).data


def test_what_the_machine_sends_is_the_shared_sample_letter_for_letter():
    fm = WIRE["from_machine"]

    async def body(link, session, url):
        tab = await session.ws_connect(f"{url}/device?name={TAB}")
        assert await _raw(tab) == fm[0]  # hello・話せる
        link.window("")  # 家の窓
        assert await _raw(tab) == fm[6]
        phone = await session.ws_connect(f"{url}/device?name=phone-2")
        assert await _raw(phone) == fm[1]  # hello・話せない
        say = json.loads(fm[2])
        link.send(TAB, "say", {"text": say["text"], "display": say["display"]})
        assert await _raw(tab) == fm[2]
        link.send(TAB, "cue", {"kind": "recalling"})
        assert await _raw(tab) == fm[3]
        link.send(TAB, "closed", {})
        assert await _raw(tab) == fm[4]
        link.send(TAB, "refused", {})
        assert await _raw(tab) == fm[5]
        link.window(None)
        assert await _raw(tab) == fm[7]

    _run(body)


def test_what_the_app_sends_is_taken_as_words_and_speaker():
    got: list = []
    done = asyncio.Event()

    async def push(device, text, speaker):
        got.append((device, text, speaker))
        if len(got) == len(WIRE["to_machine"]):
            done.set()

    async def body(link, session, url):
        tab = await session.ws_connect(f"{url}/device?name={TAB}")
        await _raw(tab)
        for sample in WIRE["to_machine"]:
            await tab.send_str(sample["wire"])
        await asyncio.wait_for(done.wait(), 2.0)

    _run(body, push)
    assert got == [(TAB, s["text"], s["speaker"]) for s in WIRE["to_machine"]]
    # アプリが打たれた字（typed）から落とす空白は、機体の端末の口が落とす空白と同じ
    assert all(s["typed"].strip() == s["text"] for s in WIRE["to_machine"])


def _machine_heard(text: str, names: list) -> str:
    if is_name_repeat(text, names) or not names_me(text, names, fuzzy=False):
        return "none"
    return "name_only" if is_name_only(text, names) else "named"


def test_the_app_hears_its_name_by_the_same_rule_as_the_machine_gate():
    names = WIRE["names"]
    for case in WIRE["wake"]:
        assert _machine_heard(case["text"], names) == case["heard"], case["text"]


def test_the_app_keeps_the_conversation_as_long_as_the_machine_window():
    """端-c c3：機体は端末に自分の窓が開いているかを送らない。アプリは同じ長さ（見本の `window_sec`）を自分で数える。"""
    from familiar_agent.core.wake_window import WAKE_WINDOW_SEC

    assert WIRE["window_sec"] == WAKE_WINDOW_SEC
