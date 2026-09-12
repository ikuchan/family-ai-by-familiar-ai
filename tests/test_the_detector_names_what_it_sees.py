"""人検出のモデルは、写っているものの名前も返せる（`イベント駆動ループ` v0.43）。

`see` の帰りを VLM で待たないための即席の意味づけ。COCO 80 種・重複はそのまま（個数）。
`count` は変えない。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from familiar_agent.recognition.person_detector import PersonDetector


def _model(classes, names):
    r = MagicMock()
    r.boxes.cls.tolist = MagicMock(return_value=classes)
    m = MagicMock()
    m.predict = MagicMock(return_value=[r])
    m.names = names
    return m


def _detector(model):
    d = PersonDetector()
    d._mr_model = model
    return d


def test_labels_name_every_box_in_order():
    m = _model([0.0, 0.0, 56.0], {0: "person", 56: "chair"})
    assert asyncio.run(_detector(m).labels("/tmp/a.jpg")) == ["person", "person", "chair"]


def test_labels_ask_for_every_class_not_only_person():
    m = _model([], {})
    asyncio.run(_detector(m).labels("/tmp/a.jpg"))
    assert "classes" not in m.predict.call_args.kwargs


def test_no_model_means_no_labels():
    d = PersonDetector()
    d._mr_model = None
    d.ensure = MagicMock(return_value=None)  # type: ignore[method-assign]
    assert asyncio.run(d.labels("/tmp/a.jpg")) == []


def test_a_failed_inference_names_nothing_rather_than_raising():
    m = MagicMock()
    m.predict = MagicMock(side_effect=RuntimeError("cuda oom"))
    assert asyncio.run(_detector(m).labels("/tmp/a.jpg")) == []
