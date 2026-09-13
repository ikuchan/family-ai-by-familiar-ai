"""重いモデルは起動時に温める（2026-09-13 実機で露見）。

人検出（YOLO）と見えのエンコーダ（DINOv2）は最初の `see` まで読まれず、初回の see が
5.3 秒かかって「5 秒超え」のつなぎまで出た。読込は起動時に背景で済ませ、YOLO は空の
1 枚を通して融合まで終えておく。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.core.warmup import warm_models


def test_warmup_loads_and_runs_each_model_once() -> None:
    detector = MagicMock()
    encoder = MagicMock()
    asyncio.run(warm_models(detector=detector, encoder=encoder))
    detector.warm.assert_called_once()
    encoder.warm.assert_called_once()


def test_warmup_survives_a_failing_model(caplog) -> None:
    detector = MagicMock()
    detector.warm = MagicMock(side_effect=RuntimeError("no cuda"))
    encoder = MagicMock()
    asyncio.run(warm_models(detector=detector, encoder=encoder))  # 落ちない
    encoder.warm.assert_called_once()
    assert any("温められなかった" in r.getMessage() for r in caplog.records)


def test_nothing_to_warm_is_fine() -> None:
    asyncio.run(warm_models(detector=None, encoder=None))


def test_the_detector_warm_runs_one_blank_inference() -> None:
    from familiar_agent.recognition.person_detector import PersonDetector

    d = PersonDetector()
    model = MagicMock()
    model.names = {0: "person"}
    r = MagicMock()
    r.boxes.cls.tolist = MagicMock(return_value=[])
    model.predict = MagicMock(return_value=[r])
    d._mr_model = model
    d.warm()
    assert model.predict.call_count == 1


def test_the_agent_starts_the_warmup_in_the_background() -> None:
    import inspect

    from familiar_agent.agent import EmbodiedAgent

    src = inspect.getsource(EmbodiedAgent._start_background_services)
    assert "warm_models(" in src


def test_the_one_time_services_are_primed_only_once(monkeypatch) -> None:
    """ウォームアップ・TTS 起こし・Whisper 読み込みは 1 回だけ（環-k・2026-09-14）。

    `_start_background_services` は `_ensure_event_loop` の末尾で呼ばれ、それは起動時と
    **人の発話のたび**（`run()`）に走る。MCP とワーカーは `is_started` で守られていたが、
    温めは守られておらず、実機で `人検出を温めた` が起動時に 2 回・発話ごとに 1 回出て、
    そのたびに YOLO のダミー推論が走っていた（2026-09-13 21:21）。
    """
    import asyncio
    from unittest.mock import MagicMock

    from familiar_agent import agent as agent_mod
    from familiar_agent.agent import EmbodiedAgent

    calls: list[str] = []

    async def fake_warm(**_kw):
        calls.append("warm")

    monkeypatch.setattr("familiar_agent.core.warmup.warm_models", fake_warm)
    a = MagicMock(spec=[])
    a._mcp = None
    a._memory_worker = None
    a.config = MagicMock()
    a.config.stt.engine = "elevenlabs"

    async def scenario():
        for _ in range(3):
            EmbodiedAgent._start_background_services(a)
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert calls == ["warm"], f"温めが {len(calls)} 回走った"
    assert agent_mod is not None
