"""記-ac：自己像の「今日」は家の暦で数える（2026-11-21・仮（本人の確認待ち））。

以前は UTC の日付で、家の 0〜9 時（JST）には前の日になった。家の朝 8 時（UTC ではまだ前の日の 23 時）に種を読んだ行は、
その日の行なのに「（前の日 から）」と古く見えた。書く側（`seed`・REST の `propose`）と読む側（`render`）をそろえる。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from familiar_agent.core import self_image as si

JST = timezone(timedelta(hours=9))
NOW_UTC = datetime(2026, 11, 15, 23, 0, tzinfo=timezone.utc)  # 家の 11/16 08:00


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        return (
            NOW_UTC.astimezone(tz)
            if tz is not None
            else NOW_UTC.astimezone(JST).replace(tzinfo=None)
        )


@pytest.fixture
def morning(monkeypatch):
    monkeypatch.setattr(si, "datetime", _FixedDatetime)
    monkeypatch.setattr(si, "local_tz", lambda: JST)


def test_a_line_seeded_this_morning_is_today(morning, tmp_path):
    seed = tmp_path / "seed.yaml"
    seed.write_text("望み:\n  - みんなと話したい\n気がかり: []\n価値: []\n", encoding="utf-8")
    image = si.seed(seed)
    assert image.hopes[0].since == date(2026, 11, 16)
    assert "から）" not in si.render(image)


def test_a_line_from_yesterday_still_shows_its_age(morning):
    image = si.SelfImage(
        hopes=(si.Line(text="みんなと話したい", since=date(2026, 11, 15), sources=()),),
        concerns=(),
        values=(),
    )
    assert "（11/15 から）" in si.render(image)
