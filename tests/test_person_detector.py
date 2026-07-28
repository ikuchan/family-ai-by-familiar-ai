"""人が居るかの判定（YOLO・在/不在の層）。

`知覚在席` §3-2 は在/不在を **G（T 側・連続）が YOLO（person・GPU）で**担うと定める。
全身ベースなので顔が見えない向きにも強く、**登録が要らない**（誰かは問わない）。

`ultralytics` は依存に入っているが `src/` から一度も呼ばれていなかった。ここで繋ぐ。
モデルの読込は重いので遅延させ、推論は GPU の重い呼び出しとしてイベントループの外へ出す。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from familiar_agent.recognition.person_detector import PersonDetector


def _result(n_boxes):
    r = MagicMock()
    r.boxes = list(range(n_boxes))
    return [r]


def _detector(model):
    d = PersonDetector()
    d._model = model          # 遅延読込を飛ばす
    return d


def test_no_person_in_an_empty_frame():
    model = MagicMock()
    model.predict = MagicMock(return_value=_result(0))
    assert asyncio.run(_detector(model).count("/tmp/a.jpg")) == 0


def test_two_people_are_counted():
    model = MagicMock()
    model.predict = MagicMock(return_value=_result(2))
    assert asyncio.run(_detector(model).count("/tmp/a.jpg")) == 2


def test_only_the_person_class_is_asked_for():
    # 椅子や鞄まで数えると、部屋が常に「誰か居る」になる。COCO の person は class 0。
    model = MagicMock()
    model.predict = MagicMock(return_value=_result(0))
    asyncio.run(_detector(model).count("/tmp/a.jpg"))
    assert model.predict.call_args.kwargs["classes"] == [0]


def test_a_failed_inference_reports_nothing_seen_rather_than_raising():
    # 在席の判定はループの常駐部から呼ばれる。ここで例外を投げると常駐タスクが死ぬ。
    model = MagicMock()
    model.predict = MagicMock(side_effect=RuntimeError("cuda oom"))
    assert asyncio.run(_detector(model).count("/tmp/a.jpg")) == 0


def test_inference_runs_off_the_event_loop():
    """GPU の重い呼び出しでイベントループを止めない（`コード規約` の並行処理）。"""
    model = MagicMock()
    model.predict = MagicMock(return_value=_result(0))
    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy:
        asyncio.run(_detector(model).count("/tmp/a.jpg"))
    assert spy.called


def test_the_model_is_loaded_once_and_reused():
    d = PersonDetector()
    made = MagicMock(return_value=MagicMock(predict=MagicMock(return_value=_result(0))))
    with patch("ultralytics.YOLO", made):
        asyncio.run(d.count("/tmp/a.jpg"))
        asyncio.run(d.count("/tmp/b.jpg"))
    assert made.call_count == 1


def test_a_model_that_cannot_load_degrades_to_no_detection():
    # 重みが取れない環境でも、他の機能は動き続ける。
    d = PersonDetector()
    with patch("ultralytics.YOLO", MagicMock(side_effect=RuntimeError("no weights"))):
        assert asyncio.run(d.count("/tmp/a.jpg")) == 0
