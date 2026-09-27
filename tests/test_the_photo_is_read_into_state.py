"""写真の読み取りを、システムの状態として残す（出-au 段 5-7a・2026-09-27・`設計方針_判定の段` v0.4 §2.2.2）。

Jev は写真を見られないので、調停に写真を渡す形（v0.46）はやめる。

- 写真を読む口（`scene.read_photo`）が、見えたもの（ラベル）と写っている人の見立て（呼び方と確信度）を 1 回で返す。
  家族の見た目の記述（FAMILY.md）を渡す。
- **調停が見ると決めた帰り**は、読み取りを**待って**から完了を返す。見えたものは O の記録と完了の文に、見立ては在席
  （`_apply_seen_people`）に残る。調停は W の中の文として読む（写真は渡さない）。
- 主LLM が見ると決めた帰りは、いままでどおり（読み取りは背景・主LLM が写真を見て話す）。
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from familiar_agent.loop.event_loop import InformationProcessing
from familiar_agent.scene import read_photo

from tests.test_the_seen_mark_keeps_the_picture import _ip, _marks, _pose


def _backend(reply):
    b = MagicMock()
    b.complete_with_image = AsyncMock(return_value=reply)
    return b


def test_reading_returns_things_and_people_with_the_family_described():
    b = _backend(
        '{"entities": [{"label": "机"}, {"label": "ノートパソコン"}],'
        ' "people": [{"name": "パパ", "confidence": 0.7}, {"name": "", "confidence": 0.5}]}'
    )
    labels, people = asyncio.run(read_photo("IMG", b, family_md="パパ：眼鏡をかけている"))
    assert labels == ["机", "ノートパソコン"]
    assert people == [{"name": "パパ", "confidence": 0.7}, {"name": "", "confidence": 0.5}]
    prompt, image = b.complete_with_image.await_args.args
    assert "眼鏡をかけている" in prompt and image == "IMG"


def test_reading_failure_returns_nothing():
    b = _backend("ごめん")
    assert asyncio.run(read_photo("IMG", b, family_md="")) == ([], [])
    b.complete_with_image = AsyncMock(side_effect=RuntimeError("down"))
    assert asyncio.run(read_photo("IMG", b, family_md="")) == ([], [])


def test_an_arbiter_see_waits_for_the_reading_and_keeps_it_as_state(monkeypatch):
    import familiar_agent.loop.event_loop as mod

    monkeypatch.setattr(
        mod,
        "read_photo",
        AsyncMock(return_value=(["机", "ポスター"], [{"name": "パパ", "confidence": 0.7}])),
    )

    async def scenario():
        a, ip = _ip()
        ip._req.see_by = "調停"
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        ip._apply_seen_people = AsyncMock()
        out = await ip._run_camera("see", {})
        refines = len(ip._background_tasks)
        people = ip._apply_seen_people.await_args.args[0]
        marks = _marks(a)
        await ip.close()
        return out, refines, people, marks

    out, refines, people, marks = asyncio.run(scenario())
    assert "机" in out and "ポスター" in out  # 読み取りが完了の文（W）に載る
    assert people == [{"name": "パパ", "confidence": 0.7}]
    assert refines == 0  # 背景の差し替えはしない（待って読んだので）
    assert any("机" in m for m, _ in marks)


def test_a_main_llm_see_keeps_the_background_refinement(monkeypatch):
    import familiar_agent.loop.event_loop as mod

    reader = AsyncMock(return_value=(["机"], []))
    monkeypatch.setattr(mod, "read_photo", reader)

    async def never(*_a, **_kw):
        await asyncio.sleep(3600)

    monkeypatch.setattr(mod, "extract_entities", never)

    async def scenario():
        a, ip = _ip()
        ip._req.see_by = "主LLM"
        monkeypatch.setattr(ip, "_current_pose_name", _pose)
        await ip._run_camera("see", {})
        refines = len(ip._background_tasks)
        await ip.close()
        return refines

    assert asyncio.run(scenario()) == 1
    reader.assert_not_awaited()


def test_the_arbiter_no_longer_gets_the_photo():
    import inspect

    src = inspect.getsource(InformationProcessing._decide)
    assert "image_b64=image_b64" not in src and "_seen_image(" not in src
