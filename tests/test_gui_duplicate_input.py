"""同じ入力が続けてキューへ積まれるのを弾く（実機で「2回答える」が起きた）。

2026-07-30 の実機で、1つの「おはよう。」に2回答えた。GUI のキューに2件入っていた。
入口は2つある。

- `_on_send`（キーボード）… `GUI input queued` を出す
- `_on_realtime_stt_committed`（音声）… **何も出していなかった**

どちらから積まれたか区別できなかったので、まずログを揃える。そのうえで、短い間に
同じ文が続けて来たら落とす。

**言い直しは弾かない。** 人は聞こえなかったと思って同じことを言い直す。窓を短く取り、
それより後の同じ文は通す。
"""

from __future__ import annotations

import pytest

from familiar_agent._ui_helpers import DuplicateInputFilter


def test_the_same_text_twice_in_a_row_is_dropped():
    f = DuplicateInputFilter(window_sec=3.0)
    assert f.accept("おはよう。", now=100.0) is True
    assert f.accept("おはよう。", now=100.5) is False       # 0.5 秒後の重複


def test_a_different_text_is_always_accepted():
    f = DuplicateInputFilter(window_sec=3.0)
    assert f.accept("おはよう。", now=100.0) is True
    assert f.accept("こんばんは。", now=100.1) is True


def test_the_same_text_after_the_window_is_accepted():
    """言い直しを弾かない。窓の外なら同じ文でも通す。"""
    f = DuplicateInputFilter(window_sec=3.0)
    assert f.accept("聞こえた？", now=100.0) is True
    assert f.accept("聞こえた？", now=104.0) is True        # 4 秒後＝言い直し


def test_the_window_is_measured_from_the_accepted_input():
    """弾いた入力で窓を延ばさない。延ばすと、連打のあいだ永久に通らなくなる。"""
    f = DuplicateInputFilter(window_sec=3.0)
    assert f.accept("うん", now=100.0) is True
    assert f.accept("うん", now=101.0) is False
    assert f.accept("うん", now=102.0) is False
    assert f.accept("うん", now=103.5) is True              # 最初の受け入れから 3.5 秒


def test_surrounding_whitespace_does_not_make_it_a_different_text():
    f = DuplicateInputFilter(window_sec=3.0)
    assert f.accept("はい", now=100.0) is True
    assert f.accept("  はい  ", now=100.2) is False


def test_the_window_can_be_disabled():
    """窓を 0 にすると何も弾かない（設定で戻せるようにしておく）。"""
    f = DuplicateInputFilter(window_sec=0.0)
    assert f.accept("おはよう。", now=100.0) is True
    assert f.accept("おはよう。", now=100.1) is True


def test_the_config_default_is_three_seconds():
    import os
    from unittest.mock import patch

    from familiar_agent.config import AgentConfig

    with patch.dict(os.environ, {}, clear=True):
        assert AgentConfig().input_dedupe_window_sec == pytest.approx(3.0)
