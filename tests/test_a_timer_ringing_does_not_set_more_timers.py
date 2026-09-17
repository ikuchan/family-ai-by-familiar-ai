"""タイマーが鳴った知らせで、タイマーを掛け直さない（2026-09-16 実機 17:00）。

「10秒のタイマーをかけてその間黙ってて」で掛けた id=6 が鳴り、その知らせ（起点＝機器）の
求めで調停が `set_timer` を 3 回（ラベル違い・id=7〜9）掛け直し、それが同時に鳴って
「かしこまりました」が 3 回続いた。起点が機器でも調停は返事型のプロンプトのまま、W の
直近にある人の言葉を**いまの頼み**として読んでいた（出-q の機器版）。

- 起点が機器の求めには「これは知らせ。直近のやりとりは済んだこと」を渡す（`_LEAD_DEVICE`）。
- タイマーが鳴った求めでは、調停の候補から `set_timer`・`start_stopwatch` を外す（機械の守り。
  止める `cancel_timer` は残す）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.arbiter import arbitrate
from familiar_agent.loop.event_loop import InformationProcessing


def _backend(reply: str):
    b = AsyncMock()
    b.complete = AsyncMock(return_value=reply)
    return b


def test_a_device_origin_is_framed_as_a_notice_not_a_request():
    b = _backend('{"branch":"light","text":"タイマーの時間だよ"}')
    asyncio.run(
        arbitrate(
            b,
            utterance="[タイマー] タイマー：「パパとの約束」の時間（パパに頼まれたもの）",
            workspace_ctx="[直近のやりとり]\n- 17:00 パパ：10秒のタイマーをかけてその間黙ってて",
            origin="機器",
        )
    )
    prompt = b.complete.call_args.args[0]
    assert "[届いた知らせ]" in prompt
    assert "済んだこと" in prompt and "改めて応じない" in prompt

    b = _backend('{"branch":"light","text":"はい"}')
    asyncio.run(arbitrate(b, utterance="10秒のタイマーをかけて", workspace_ctx=""))
    assert "[届いた知らせ]" not in b.complete.call_args.args[0]


def _ip_with_timer():
    a = MagicMock()
    tool = MagicMock()
    tool.get_tool_definitions = MagicMock(
        return_value=[{"name": n} for n in ("set_timer", "start_stopwatch", "cancel_timer")]
    )
    a._timer_tool = tool
    a._mcp = None
    ip = InformationProcessing(a)
    ip._gated = lambda defs: defs
    ip._ACTIONS = {
        **{
            k: (lambda ip, k=k: [])
            for k in ("house_rules", "family_schedule", "notion_search", "journal", "vault")
        },
        "set_timer": lambda ip: [{"name": "set_timer"}],
        "start_stopwatch": lambda ip: [{"name": "start_stopwatch"}],
        "cancel_timer": lambda ip: [{"name": "cancel_timer"}],
        "pause_timer": lambda ip: [{"name": "pause_timer"}],
        "resume_timer": lambda ip: [{"name": "resume_timer"}],
        "set_alarm": lambda ip: [{"name": "set_alarm"}],
        "cancel_alarm": lambda ip: [{"name": "cancel_alarm"}],
    }
    return ip


def test_a_ringing_timer_cannot_set_another_one():
    ip = _ip_with_timer()
    ip._req.trigger_kind = "機器"
    ip._req.request_text = "[タイマー] タイマー：「パパとの約束」の時間（パパに頼まれたもの）"
    # 鳴っているときに掛け直しは外す。止める・一時停止・再開は残る（2026-09-18・段 4）。
    assert set(ip._extra_actions()) == {
        "cancel_timer",
        "pause_timer",
        "resume_timer",
        "set_alarm",
        "cancel_alarm",
    }


def test_a_person_can_still_set_a_timer():
    ip = _ip_with_timer()
    ip._req.trigger_kind = "発話"
    ip._req.request_text = "10秒のタイマーをかけて"
    assert {"set_timer", "start_stopwatch", "cancel_timer"} <= set(ip._extra_actions())


def test_an_arrival_notice_can_still_set_a_timer():
    """機器でもタイマーの知らせ以外（入室）は外さない。"""
    ip = _ip_with_timer()
    ip._req.trigger_kind = "機器"
    ip._req.request_text = "[入室] パパ が来た"
    assert "set_timer" in ip._extra_actions()
