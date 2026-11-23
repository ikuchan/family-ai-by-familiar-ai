"""見えが変わったかの足切りと、前と今の見比べ（知-i 段 2）。

足切りは安く、見比べは高い。足切りが拾うのは「人が映っていない・見えが落ち着いている・前の見えから遠い・
しばらく見比べていない」回だけで、物が変わったか明るさだけかは、前と今を並べた 1 枚を見せて聞く。
"""

from __future__ import annotations

import asyncio
import base64
import io
from unittest.mock import AsyncMock, MagicMock

from PIL import Image

from familiar_agent import view_change as vc

_A = [1.0] + [0.0] * 383
_B = [0.0, 1.0] + [0.0] * 382  # _A から遠い（距離 1）


def _cand(**kw):
    args = dict(
        ref=_A,
        seen=_B,
        surprise=0.001,
        boxes=0,
        checked_at=None,
        now=1_000_000.0,
        threshold=0.15,
        settled=0.02,
        cooldown_sec=6 * 3600,
    )
    args.update(kw)
    return vc.is_candidate(**args)


def test_a_settled_empty_view_far_from_the_earlier_one_is_a_candidate():
    assert _cand() is True


def test_a_view_close_to_the_earlier_one_is_not():
    assert _cand(seen=_A) is False


def test_a_view_with_any_box_is_not():
    """座っている人は動く人に数えないので、枠の数で見る（人が映った見えを前の見えと比べない）。"""
    assert _cand(boxes=1) is False


def test_a_view_still_moving_is_not():
    """移動平均からまだ遠い（いま変わっている最中・明かりが付いた直後）回は、落ち着くまで待つ。"""
    assert _cand(surprise=0.3) is False
    assert _cand(surprise=None) is False


def test_a_pose_checked_recently_is_not_checked_again():
    assert _cand(checked_at=1_000_000.0 - 3600) is False
    assert _cand(checked_at=1_000_000.0 - 7 * 3600) is True


def _jpeg(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_the_two_views_are_put_side_by_side_before_on_the_left():
    b64 = vc.side_by_side(_jpeg((255, 0, 0)), _jpeg((0, 0, 255)))
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    assert img.size[0] >= 2 * 64 and img.size[1] >= 48
    left, right = img.getpixel((10, 24)), img.getpixel((img.size[0] - 10, 24))
    assert left[0] > 200 and left[2] < 60  # 前（赤）が左
    assert right[2] > 200 and right[0] < 60  # 今（青）が右


def test_the_judge_reads_the_kind_and_what_changed():
    backend = MagicMock()
    backend.complete_with_image = AsyncMock(
        return_value='{"kind": "物", "what": "窓のところに花瓶が増えた"}'
    )
    assert asyncio.run(vc.judge(backend, "B64")) == ("物", "窓のところに花瓶が増えた")


def test_an_unreadable_or_unknown_answer_is_no_verdict():
    backend = MagicMock()
    backend.complete_with_image = AsyncMock(return_value="よく分からない")
    assert asyncio.run(vc.judge(backend, "B64")) is None
    backend.complete_with_image = AsyncMock(return_value='{"kind": "猫", "what": ""}')
    assert asyncio.run(vc.judge(backend, "B64")) is None
    backend.complete_with_image = AsyncMock(side_effect=RuntimeError("down"))
    assert asyncio.run(vc.judge(backend, "B64")) is None


def test_a_broken_picture_is_no_verdict_rather_than_an_error():
    backend = MagicMock()
    backend.complete_with_image = AsyncMock(return_value='{"kind": "物", "what": "x"}')
    assert asyncio.run(vc.compare(backend, b"not a jpeg", _jpeg((0, 0, 0)))) is None
    backend.complete_with_image.assert_not_awaited()


def test_large_pictures_are_shrunk_before_they_are_put_side_by_side():
    buf = io.BytesIO()
    Image.new("RGB", (2560, 1440), (0, 0, 0)).save(buf, format="JPEG")
    img = Image.open(io.BytesIO(base64.b64decode(vc.side_by_side(buf.getvalue(), buf.getvalue()))))
    assert img.size == (2048, 576)


def test_a_backend_without_pictures_cannot_judge():
    class Text:
        async def complete(self, prompt):
            return ""

    assert vc.can_judge(Text()) is False
    assert vc.can_judge(MagicMock()) is True
