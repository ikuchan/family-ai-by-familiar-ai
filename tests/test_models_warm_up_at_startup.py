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
