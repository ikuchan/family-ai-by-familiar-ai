"""声で名前だけを聞いたら、返事をせずに窓を開け、BOND を動かす（出-ch 段 3・2026-10-16・本人の決定ア・カ・サ・タ）。

名前だけで呼ぶことはあまりなく、ふつうは用件が続く（本人）。名前だけには返事をしない。
- ア：値が足りれば黙って窓を開ける（10 秒）。続けて言った用件は名前なしで受ける。足りなければ開けない。
- カ・サ：いつも BOND へ `voice_name_bond_nudge`（0.05〔仮〕）を足す。居るか・声が当たるか・値は見ない。
- タ：名前だけの繰り返しは窓を開けない（2026-10-07 の決定のまま）。BOND へ足すだけ。
キーボードはいままでどおり。黙っていての依頼が効いているあいだは控えるだけ。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.test_wake_strictness import _gate, _heard

pytestmark = pytest.mark.real_window


def _bond_calls(a):
    return [c for c in a._nudge_drive.await_args_list if c.args[0] == "bond"]


def _closed(ip):
    ip._wake_window().until = time.monotonic() - 60
    return ip


def test_a_bare_name_with_enough_values_opens_the_window_without_a_reply():
    ip, a = _gate()
    _closed(ip)
    assert _heard(ip, "パジュ", ns=0.019, lp=-0.558) is False  # 返事はしない
    assert ip._wake_window().is_open(time.monotonic())
    assert [c.args[1] for c in _bond_calls(a)] == [0.05]


def test_a_bare_name_without_enough_values_keeps_the_window_shut():
    ip, a = _gate(here=False, known_voice=False)
    _closed(ip)
    assert _heard(ip, "パジュー", ns=0.150, lp=-0.723) is False
    assert not ip._wake_window().is_open(time.monotonic())
    assert [c.args[1] for c in _bond_calls(a)] == [0.05]  # 居なくても・声が分からなくても
    assert [c for c in a._nudge_drive.await_args_list if c.args[0] == "safety"] == []


def test_a_bare_name_inside_the_window_reopens_it_without_a_reply():
    ip, a = _gate(here=True, known_voice=True)
    ip._wake_window().open(time.monotonic() - 8)  # あと 2 秒で閉じる
    assert _heard(ip, "パジュー", ns=0.150, lp=-0.723) is False
    assert ip._wake_window().is_open(time.monotonic() + 5)  # 開け直して 10 秒
    assert len(_bond_calls(a)) == 1


def test_a_repeated_name_moves_bond_but_opens_nothing():
    ip, a = _gate(here=True, known_voice=True)
    _closed(ip)
    assert _heard(ip, "パジュ、パジュ、パジュー", ns=0.083, lp=-0.513) is False
    assert not ip._wake_window().is_open(time.monotonic())
    assert [c.args[1] for c in _bond_calls(a)] == [0.05]


def test_the_name_with_words_is_heard_and_does_not_move_bond_by_the_name():
    """**反証の側**：名前＋言葉はいままでどおり受け、名前だけの BOND は足さない。"""
    ip, a = _gate(here=True, known_voice=True)
    _closed(ip)
    assert _heard(ip, "パジュ、いま何時？", ns=0.05, lp=-0.5) is True
    a._nudge_drive.assert_not_awaited()


def test_a_bare_name_typed_on_the_keyboard_is_still_heard():
    """**反証の側**：キーボードはいままでどおり。"""
    ip, a = _gate()
    _closed(ip)
    assert _heard(ip, "パジュ", source="keyboard") is True
    a._nudge_drive.assert_not_awaited()


@pytest.mark.parametrize("ns,opened", [(0.019, True), (0.150, False)])
def test_the_screen_shows_heard_only_when_the_window_opened(ns, opened):
    ip, _ = _gate()
    _closed(ip)
    seen: list = []
    ip._on_heard = lambda text, heard: seen.append(heard)
    out = asyncio.run(ip.push_utterance("パジュー", source="voice", no_speech=ns, logprob=-0.6))
    assert out == ""
    assert seen == [opened]


def test_the_amount_is_a_setting(monkeypatch):
    from familiar_agent.config import DriveConfig

    assert DriveConfig().voice_name_bond_nudge == 0.05
    monkeypatch.setenv("DRIVE_VOICE_NAME_BOND_NUDGE", "0.2")
    assert DriveConfig().voice_name_bond_nudge == 0.2
