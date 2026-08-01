"""見た印の中身を、見たことだけにする（実機で2つの汚れが出た）。

実機（2026-08-01）で見た印が書かれるようになったが、中身に2種類のごみが混ざった。

**カメラの定型文とファイルパス。**

```
出入り口を見た。You see the current view (saved to /home/…/capture_20260801_123536.jpg) 見えたもの：child、adult
```

`see` が返す `"You see the current view (saved to …)."` は、撮ったことを LLM へ伝える
ための文であって、見た内容ではない。想起はこの文でベクトルを作るので、**毎回同じ英語の
定型句とパスがノイズになる**。印に要るのは「どの定点を見て、何が見えたか」だけである。

**見ていないのに立つ印。**

```
「目の前を見る」はこの求めですでに調べた。結果は W にある。
```

同じ語を二度調べないための内部メッセージが、観察として書かれていた。`_intake` が
`action == "see"` で分岐するため、弾かれた完了も見た印になっていた。カメラは回って
いない。

どちらも「どこで観察を書くか」に由来する。**実際にカメラを回した `_run_camera` の中で
書く。** 弾かれた完了はそこを通らないので、二つめは構造で起きなくなる。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent

_CAMERA_REPLY = "You see the current view (saved to /home/u/.familiar_ai/x.jpg)."


class _Camera:
    """撮ったふりをする目（実機もモデルも呼ばない）。"""

    async def call(self, action, _tool_input):
        if action == "see":
            return _CAMERA_REPLY, "BASE64"
        return "窓側のほうを向いた。", None

    async def position(self):
        return None


def _ip():
    a = _agent(stream_returns=[])
    a._camera = _Camera()
    ip = InformationProcessing(a)
    ip._origin_text = "周りを見て"
    return a, ip


def _observations(agent) -> list[str]:
    return [c.args[0] for c in agent._memory.save_async_with_id.call_args_list
            if c.kwargs.get("direction") == "観察"]


def _patch_scene(ip, labels):
    """意味づけの結果を差し替える（VLM は呼ばない）。"""
    import familiar_agent.loop.event_loop as mod

    async def _fake(*_a, **_kw):
        return [{"label": x} for x in labels]

    mod.extract_entities = _fake


def test_the_camera_boilerplate_is_not_recorded(monkeypatch) -> None:
    """カメラの定型文とファイルパスを印に残さない。"""
    async def scenario():
        a, ip = _ip()
        _patch_scene(ip, ["child", "adult"])
        monkeypatch.setattr(ip, "_current_pose_name", lambda: _async("出入り口"))
        await ip._run_camera("see", {})
        await ip.close()
        return a

    a = asyncio.run(scenario())
    got = _observations(a)
    assert got, "印が書かれていない"
    assert "You see" not in got[0], f"定型文が残っている: {got[0]}"
    assert ".jpg" not in got[0], f"ファイルパスが残っている: {got[0]}"


def test_the_mark_keeps_the_pose_and_the_labels(monkeypatch) -> None:
    """印には定点名と見えたものが入る。"""
    async def scenario():
        a, ip = _ip()
        _patch_scene(ip, ["child", "desk"])
        monkeypatch.setattr(ip, "_current_pose_name", lambda: _async("窓側"))
        await ip._run_camera("see", {})
        await ip.close()
        return a

    a = asyncio.run(scenario())
    got = _observations(a)[0]
    assert "窓側を見た" in got, f"定点名が無い: {got}"
    assert "child" in got and "desk" in got, f"見えたものが無い: {got}"


def test_a_blocked_lookup_leaves_no_mark() -> None:
    """弾かれた調査は印を残さない（カメラを回していない）。"""
    async def scenario():
        a, ip = _ip()
        ip._lookup_action_by_query["目の前を見る"] = "see"
        ip._completion_queue.put_nowait(
            ("目の前を見る", "「目の前を見る」はこの求めですでに調べた。結果は W にある。",
             None, "完了", 1))
        await ip._intake()
        await ip.close()
        return a

    a = asyncio.run(scenario())
    assert not _observations(a), f"見ていないのに印が立っている: {_observations(a)}"


def test_look_alone_leaves_no_mark(monkeypatch) -> None:
    """首を振っただけでは印を残さない（観察していない）。"""
    async def scenario():
        a, ip = _ip()
        await ip._run_camera("look", {"pose": "窓側"})
        await ip.close()
        return a

    a = asyncio.run(scenario())
    assert not _observations(a), "look だけで印が立っている"


def test_a_failed_capture_leaves_no_mark(monkeypatch) -> None:
    """撮れなかったときは印を残さない（見ていない）。"""
    async def scenario():
        a, ip = _ip()

        async def _no_frame(_action, _tool_input):
            return ("Camera capture failed.", None)

        a._camera.call = _no_frame
        await ip._run_camera("see", {})
        await ip.close()
        return a

    a = asyncio.run(scenario())
    assert not _observations(a), "撮れていないのに印が立っている"


async def _async(value):
    return value
