"""自律機構接続 AIF：T と I をつなぐ唯一の口（`設計図` ③-2・環-e-ろ）。

設計は口を4つ（IIF・DIF・AIF・OIF）と定め、この4つ以外にコンポーネントどうしが直接
つながる線を置かないとする。ところが **AIF は実装として存在しなかった**。

- T→I の情動発火：`tonic.py` が `InformationProcessing.push_affect()` を直接呼ぶ
  （`push_affect` の docstring は「AIF 経由」と書いているのに、その AIF が無い）
- I→T の Nudge：`agent.py` が `nudge_current_mood()` を直接呼ぶ

**T と I が互いの中身に手を伸ばしている。** 口を1枚挟んで、行き来をそこに集める。

挙動は変えない。キューへの書き方も、mood の計算も、いまのままである。AIF は転送するだけで、
通ったものを debug ログに残す。
"""

from __future__ import annotations

import logging

import pytest

from familiar_agent.io.aif import AIF, Firing, Nudge
from familiar_agent.mood_register import MoodPAD

_LOGGER = "familiar_agent.io.aif"


class _Loop:
    """I のふり。積まれたものを控える。"""

    def __init__(self) -> None:
        self.affects: list[tuple[str, str]] = []

    def push_affect(self, drive_name: str, prompt: str) -> None:
        self.affects.append((drive_name, prompt))


def _aif(loop=None, nudged=None):
    """AIF を組み立てる。mood 側は関数で差し替える（DB を触らない）。"""
    loop = loop or _Loop()
    calls: list = []

    def _nudge(items):
        calls.append(items)
        return MoodPAD(p=0.6, pn=0.4, a=0.5, dom=0.5)

    return AIF(loop, nudge=_nudge), loop, calls


class TestFiring:
    """T → I（情動発火）。"""

    def test_a_firing_reaches_the_loop(self) -> None:
        aif, loop, _ = _aif()
        aif.fire(Firing(axis="seeking", inner_voice="探索したい気持ちが募っている"))
        assert loop.affects == [("SEEKING", "探索したい気持ちが募っている")]

    def test_the_axis_is_upper_cased(self) -> None:
        """I へ渡す軸名は大文字（いまの `push_affect` の呼ばれ方に合わせる）。"""
        aif, loop, _ = _aif()
        aif.fire(Firing(axis="safety", inner_voice="確かめたい"))
        assert loop.affects[0][0] == "SAFETY"

    def test_a_firing_is_frozen(self) -> None:
        import dataclasses

        f = Firing(axis="rest", inner_voice="休みたい")
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.axis = "bond"          # type: ignore[misc]


class TestNudge:
    """I → T（Nudge）。"""

    def test_a_nudge_reaches_the_register(self) -> None:
        aif, _, calls = _aif()
        got = aif.nudge(Nudge(items=[(MoodPAD(p=0.8), 1.0)]))
        assert calls == [[(MoodPAD(p=0.8), 1.0)]]
        assert isinstance(got, MoodPAD)
        assert got.p == pytest.approx(0.6)

    def test_an_empty_nudge_still_goes_through(self) -> None:
        """W が空でも通す（経過による減衰だけが効く）。"""
        aif, _, calls = _aif()
        aif.nudge(Nudge(items=[]))
        assert calls == [[]]


class TestDebugTrail:
    def test_both_directions_leave_a_line(self, caplog) -> None:
        aif, _, _ = _aif()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            aif.fire(Firing(axis="seeking", inner_voice="探索したい"))
            aif.nudge(Nudge(items=[(MoodPAD(), 1.0)]))
        msgs = [r.getMessage() for r in caplog.records]
        assert any("fire" in m for m in msgs), f"発火が残っていない: {msgs}"
        assert any("nudge" in m for m in msgs), f"Nudge が残っていない: {msgs}"

    def test_the_inner_voice_is_not_spelled_out(self, caplog) -> None:
        """内声も記憶の内容と同じ扱いで、先頭だけにする。"""
        voice = "探索したい気持ちが募っている。" * 50
        aif, _, _ = _aif()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            aif.fire(Firing(axis="seeking", inner_voice=voice))
        for m in (r.getMessage() for r in caplog.records):
            assert len(m) < 200, f"ログが長すぎる: {len(m)}字"

    def test_nothing_leaks_at_info(self, caplog) -> None:
        aif, _, _ = _aif()
        with caplog.at_level(logging.INFO, logger=_LOGGER):
            aif.fire(Firing(axis="seeking", inner_voice="探索したい"))
        assert not [r for r in caplog.records if r.levelno >= logging.INFO], (
            "INFO 以上に出ている"
        )


class TestWiring:
    """T も I も、相手の中身を直接呼ばない。"""

    def test_tonic_does_not_call_the_loop_directly(self) -> None:
        import inspect

        from familiar_agent.loop.tonic import Tonic

        src = inspect.getsource(Tonic)
        assert "push_affect" not in src, "T が I の中身を直接呼んでいる"

    def test_the_turn_does_not_call_the_register_directly(self) -> None:
        import inspect

        from familiar_agent.agent import EmbodiedAgent

        src = inspect.getsource(EmbodiedAgent._run_post_response_pipeline)
        assert "nudge_current_mood" not in src, "I が T のレジスタを直接動かしている"
