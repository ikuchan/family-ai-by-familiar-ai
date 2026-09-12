"""見た印は画像の在りかを持ち、VLM を待たずに書かれる（`イベント駆動ループ` v0.43）。

即席の印（YOLO のラベル）を先に書いて完了を積み、VLM は背景で投げて返ったら差し替える。
"""

from __future__ import annotations

import asyncio

from familiar_agent.loop.event_loop import InformationProcessing

from tests.test_event_loop import _agent


class _Camera:
    last_capture_path = "/tmp/captures/capture_1.jpg"

    async def call(self, action, _tool_input):
        if action == "see":
            return "You see the current view (saved to /tmp/captures/capture_1.jpg).", "BASE64"
        return "窓側のほうを向いた。", None

    async def position(self):
        return None


class _Detector:
    async def labels(self, _path):
        return ["person", "chair"]


def _ip(*, detector=_Detector()):
    a = _agent(stream_returns=[])
    a._camera = _Camera()
    a._person_detector = detector
    ip = InformationProcessing(a)
    ip._req.request_text = "周りを見て"
    return a, ip


def _marks(agent):
    return [
        (c.args[0], c.kwargs.get("image_path"))
        for c in agent._memory.save_async_with_id.call_args_list
        if c.kwargs.get("direction") == "観察"
    ]


async def _never(*_a, **_kw):
    await asyncio.sleep(3600)


def _patch_vlm(monkeypatch, fn):
    import familiar_agent.loop.event_loop as mod

    monkeypatch.setattr(mod, "extract_entities", fn)


async def _pose(_self=None):
    return "出入り口"


def test_the_mark_carries_the_image_path_and_the_quick_labels(monkeypatch) -> None:
    async def scenario():
        a, ip = _ip()
        _patch_vlm(monkeypatch, _never)
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        out = await ip._run_camera("see", {})
        marks = _marks(a)
        await ip.close()
        return out, marks

    out, marks = asyncio.run(scenario())
    assert marks, "VLM を待たずに印が書かれていない"
    content, image_path = marks[0]
    assert content == "出入り口を見た。見えたもの：person、chair"
    assert image_path == "/tmp/captures/capture_1.jpg"
    assert "person" in out


def test_the_vlm_labels_replace_the_quick_ones(monkeypatch) -> None:
    async def scenario():
        a, ip = _ip()

        async def _vlm(*_a, **_kw):
            return [{"label": "person"}, {"label": "drawer"}, {"label": "poster"}]

        _patch_vlm(monkeypatch, _vlm)
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        await ip._run_camera("see", {})
        await asyncio.gather(*ip._background_tasks)
        marks = _marks(a)
        records = list(ip._req.turn_records)
        superseded = a._memory.mark_superseded.call_args_list
        await ip.close()
        return marks, records, superseded

    marks, records, superseded = asyncio.run(scenario())
    assert [m[0] for m in marks] == [
        "出入り口を見た。見えたもの：person、chair",
        "出入り口を見た。見えたもの：person、drawer、poster",
    ]
    assert marks[1][1] == "/tmp/captures/capture_1.jpg", "差し替え後も画像の在りかを持つ"
    old, new = superseded[0].args[0], superseded[0].args[1]
    assert (old, "見た") not in records and (new, "見た") in records


def test_an_empty_vlm_answer_keeps_the_quick_mark(monkeypatch) -> None:
    async def scenario():
        a, ip = _ip()

        async def _vlm(*_a, **_kw):
            return []

        _patch_vlm(monkeypatch, _vlm)
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        await ip._run_camera("see", {})
        await asyncio.gather(*ip._background_tasks)
        marks = _marks(a)
        await ip.close()
        return marks, a._memory.mark_superseded.call_args_list

    marks, superseded = asyncio.run(scenario())
    assert len(marks) == 1 and not superseded


def test_a_new_request_does_not_get_the_old_seen_record(monkeypatch) -> None:
    """求めが別世代になっていたら、O は差し替えるが並びは触らない。"""

    async def scenario():
        a, ip = _ip()
        gate = asyncio.Event()

        async def _vlm(*_a, **_kw):
            await gate.wait()
            return [{"label": "poster"}]

        _patch_vlm(monkeypatch, _vlm)
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        await ip._run_camera("see", {})
        ip._request_generation += 1
        ip._req.turn_records = [("次の起点", "起点")]
        gate.set()
        await asyncio.gather(*ip._background_tasks)
        records = list(ip._req.turn_records)
        await ip.close()
        return records, a._memory.mark_superseded.call_args_list

    records, superseded = asyncio.run(scenario())
    assert records == [("次の起点", "起点")]
    assert superseded, "O の差し替えは世代に関わらず行う"


def test_without_a_detector_the_mark_still_says_where_it_looked(monkeypatch) -> None:
    async def scenario():
        a, ip = _ip(detector=None)
        _patch_vlm(monkeypatch, _never)
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        await ip._run_camera("see", {})
        marks = _marks(a)
        await ip.close()
        return marks

    marks = asyncio.run(scenario())
    assert marks[0][0] == "出入り口を見た。"
