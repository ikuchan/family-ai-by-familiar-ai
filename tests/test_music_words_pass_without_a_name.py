"""音楽が鳴っているあいだ、名前の無い音楽の言葉を入口で通す（出-cp・2026-10-15・本人の決定ア）。

実機 10/14 23:21、マイクの関門は「まず、ストップ。」「音楽を止めて」を通したのに、入口の門が「窓の外の入力」として
捨て、名前をつけて言い直すまで止まらなかった。鳴っているあいだは、音楽の言葉（操作の言葉・プレイリスト名）だけ、名前が
無くても受ける。窓は開けない（開けるとテレビや家族の話も 30 秒受ける）。家族どうしかは聞かない。
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.test_the_wake_window_gates_voice import _heard, _ip

pytestmark = pytest.mark.real_window


def _playing(ip, a, playing=True):
    a._music_state = SimpleNamespace(playing=playing)
    a._music_table = lambda: ()
    ip._between_family = AsyncMock(return_value=True)  # 呼ばれれば「家族どうし」と答える偽物
    return ip, a


@pytest.mark.parametrize("text", ["ストップ", "まず、ストップ。", "音楽を止めて", "次の曲"])
def test_music_words_pass_while_playing_without_opening_the_window(text):
    ip, a = _playing(*_ip())
    assert _heard(ip, text) is True
    assert not ip._wake_window().is_open(time.monotonic())  # 窓は開けない
    ip._between_family.assert_not_awaited()  # 家族どうしかは聞かない


def test_what_is_still_dropped():
    """**反証の側**：鳴っていない・音楽でない言葉・キーボードは、いままでどおり捨てる。"""
    ip, a = _playing(*_ip(), playing=False)
    assert _heard(ip, "ストップ") is False
    ip, a = _playing(*_ip())
    assert _heard(ip, "今日の天気は？") is False
    ip, a = _playing(*_ip())
    assert _heard(ip, "ストップ", source="keyboard") is False


def test_passing_is_logged_without_the_text(caplog):
    import logging

    ip, a = _playing(*_ip())
    with caplog.at_level(logging.INFO, logger="familiar_agent.loop.event_loop"):
        assert _heard(ip, "音楽を止めて") is True
    lines = [r.getMessage() for r in caplog.records if "名前なしで受けた" in r.getMessage()]
    assert len(lines) == 1 and "止めて" not in lines[0]
