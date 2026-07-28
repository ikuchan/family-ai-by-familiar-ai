"""目と首をイベント駆動ループへ繋ぐ（#14）。

`see`（見る）と `look`（首を振る）は `tools/camera.py` にあるが、動作の表 `_ACTIONS` に
無く、旧 `run()` からしか呼べない。既定を反転した時点（#11 段階5）で新しい系統から目と首が
落ちていた。

**`recall` と同じ扱いにする。** `recall` も同期で結果が返る動作で、`_run_lookup` の中で
実行して完了キューへ積む。同じ経路に乗せれば 1反復1出力が保たれ、見ると決めた反復は
つなぎだけ出し、見た結果が届いた次の反復で話す。

**撮っただけでは何も伝わらない。** `see` が返すテキストは「撮って保存した」と言うだけで、
何が写っているかは base64 の画像のほうにある。完了キューはテキストしか運ばないので、
`知覚在席` §3-2 が定める意味づけ（I 側・必要時・VLM）を通し、見えたものを言葉にして添える。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from tests.test_event_loop import _agent

from familiar_agent.loop.event_loop import _FULL_ACTIONS, _LOOKUP_ACTIONS, InformationProcessing


def _camera(call_return=("You see the current view (saved to /tmp/a.jpg).", "BASE64")):
    cam = MagicMock()
    cam.get_tool_definitions = MagicMock(return_value=[
        {"name": "see", "description": "see", "input_schema": {"type": "object", "properties": {}}},
        {"name": "look", "description": "look", "input_schema": {"type": "object",
                                                                 "properties": {}}},
    ])
    cam.call = AsyncMock(return_value=call_return)
    return cam


def _with_camera(cam=None):
    a = _agent(stream_returns=[])
    a._camera = cam if cam is not None else _camera()
    return a


def _drain(ip):
    """完了キューに積まれたものを取り出す。"""
    out = []
    while not ip._completion_queue.empty():
        out.append(ip._completion_queue.get_nowait())
    return out


# --- 動作の表 -------------------------------------------------------------


def test_camera_actions_are_offered_when_a_camera_exists():
    ip = InformationProcessing(_with_camera())
    names = {d["name"] for d in ip._tools(actions=_FULL_ACTIONS)}
    assert {"see", "look"} <= names


def test_camera_actions_are_absent_without_a_camera():
    a = _agent(stream_returns=[])
    a._camera = None
    ip = InformationProcessing(a)
    names = {d["name"] for d in ip._tools(actions=_FULL_ACTIONS)}
    assert "see" not in names and "look" not in names


def test_seeing_is_a_lookup():
    # 結果は次の反復へ回る。その場で喋らせると 1反復1出力 が崩れる。
    assert "see" in _LOOKUP_ACTIONS and "look" in _LOOKUP_ACTIONS


# --- 実行と完了 -----------------------------------------------------------


def test_seeing_reports_what_was_recognised():
    ip = InformationProcessing(_with_camera())
    with patch("familiar_agent.loop.event_loop.extract_entities",
               AsyncMock(return_value=[{"label": "cat", "category": "animal", "confidence": 0.9},
                                       {"label": "mug", "category": "object", "confidence": 0.7}])):
        asyncio.run(ip._run_lookup("see", {}, "目の前を見る", None))
    (_q, result, _i, kind) = _drain(ip)[0]
    assert kind == "完了"
    assert "cat" in result and "mug" in result


def test_seeing_still_reports_when_the_vlm_finds_nothing():
    # 意味づけが空でも、見たという事実は残す（何も積まないと求めが閉じない）。
    ip = InformationProcessing(_with_camera())
    with patch("familiar_agent.loop.event_loop.extract_entities", AsyncMock(return_value=[])):
        asyncio.run(ip._run_lookup("see", {}, "目の前を見る", None))
    (_q, result, _i, kind) = _drain(ip)[0]
    assert result.strip() != "" and kind == "完了"


def test_looking_reports_the_move_result():
    cam = _camera(call_return=("Moved left 30 degrees.", None))
    ip = InformationProcessing(_with_camera(cam))
    asyncio.run(ip._run_lookup("look", {"direction": "left"}, "首を左へ向ける", None))
    (_q, result, _i, _k) = _drain(ip)[0]
    assert "Moved left" in result


def test_looking_does_not_call_the_vlm():
    # 首を振っただけで画像は無い。VLM を呼ぶ理由がない。
    cam = _camera(call_return=("Moved left 30 degrees.", None))
    ip = InformationProcessing(_with_camera(cam))
    vlm = AsyncMock(return_value=[])
    with patch("familiar_agent.loop.event_loop.extract_entities", vlm):
        asyncio.run(ip._run_lookup("look", {"direction": "left"}, "首を左へ向ける", None))
    vlm.assert_not_called()


def test_a_broken_camera_degrades_like_recall():
    # カメラは落ちる前提の機器で、落ちたらループごと止まるのは避ける。
    cam = _camera()
    cam.call = AsyncMock(side_effect=RuntimeError("camera offline"))
    ip = InformationProcessing(_with_camera(cam))
    asyncio.run(ip._run_lookup("see", {}, "目の前を見る", None))
    (_q, result, _i, kind) = _drain(ip)[0]
    assert "camera offline" in result and kind == "完了"


def test_the_vlm_failing_does_not_lose_the_observation():
    ip = InformationProcessing(_with_camera())
    with patch("familiar_agent.loop.event_loop.extract_entities",
               AsyncMock(side_effect=RuntimeError("vlm down"))):
        asyncio.run(ip._run_lookup("see", {}, "目の前を見る", None))
    assert len(_drain(ip)) == 1


# --- 求めの見出し ---------------------------------------------------------


def test_camera_actions_get_a_heading_of_their_own():
    """`see` の入力は空、`look` は `direction` しか持たない。

    見出しを `query`／`url` から取ると両方とも空文字になり、飛行中の一覧・完了の照合・
    W の「この求めのために調べたもの」がすべて衝突する。
    """
    from familiar_agent.loop.event_loop import _query_label

    assert _query_label("see", {}) != ""
    assert _query_label("look", {"pose": "窓側"}) != _query_label("look", {"pose": "襖側"})


def test_the_heading_of_a_normal_lookup_is_unchanged():
    from familiar_agent.loop.event_loop import _query_label

    assert _query_label("recall", {"query": "夏の話"}) == "夏の話"
    assert _query_label("fetch_deferred", {"url": "http://x"}) == "http://x"
