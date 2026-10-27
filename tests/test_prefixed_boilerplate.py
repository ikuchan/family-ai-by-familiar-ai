"""前置きの付いた幻聴の定型文も捨てる（知-be・2026-10-27・本人の決定ア）。

whisper は無音や物音で「ご視聴ありがとうございました」を出すが、頭に「では、」「それでは、」が付いたり
「最後までご視聴いただき…」と形が崩れたりすると、完全一致の表に当たらなかった。10/20〜10/26 に 27 件が通り、
10/24 07:02 には窓の中で会話の入力になった。「ご視聴」か「ご清聴」を含めば捨てる。whisper の癖なので、
whisper の道（`tools/local_stt.py`）からだけ呼ぶ（STT を替えたら掛からない・本人）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from familiar_agent.core.stt_rules import is_hallucination


@pytest.mark.parametrize(
    "text",
    [
        "では、ご視聴ありがとうございました。",
        "それでは、ご視聴ありがとうございました。",
        "最後までご視聴いただきありがとうございました",
        "はい、ご清聴ありがとうございました",
        "ご視聴いただき、よろしくお願いします",
    ],
)
def test_a_prefixed_boilerplate_is_dropped(text):
    assert is_hallucination(text)


@pytest.mark.parametrize(
    "text", ["教えてくれてありがとうございました", "ありがとうございました、また明日", "こんにちは"]
)
def test_ordinary_thanks_are_kept(text):
    """**反証の側**：「ご視聴」「ご清聴」の無い文は、いままでどおり完全一致でなければ落とさない。"""
    assert not is_hallucination(text)


def test_speech_mixed_with_boilerplate_is_dropped_whole():
    """本当の発話と 1 つの書き起こしに混ざっても、まるごと捨てる（頼みごとが消えてもよい・本人）。"""
    assert is_hallucination("3分測って。ご視聴ありがとうございました")


def test_only_the_whisper_path_drops_boilerplate():
    """**whisper 限定の番人**：表を見るのは whisper の常時集音（`local_stt`）だけ。ほかの STT には掛けない。
    TUI の押して話す（`tools/stt.py`）は家族が使わないので掛けていない（掛けるならこの番人も直す）。

    `local_stt` の口（`drop_if_hallucination`）を通して呼ぶ道も数える。
    """
    src = Path(__file__).resolve().parents[1] / "src" / "familiar_agent"
    names = ("is_hallucination", "drop_if_hallucination")
    callers = sorted(
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p.relative_to(src).as_posix() != "core/stt_rules.py"
        and any(n in p.read_text(encoding="utf-8") for n in names)
    )
    assert callers == ["tools/local_stt.py"]


def test_the_log_does_not_carry_the_text(caplog):
    """捨てたときの INFO のログに本文を出さない（家族の言葉が混ざりうる）。字数だけ残す。"""
    import logging

    from familiar_agent.tools import local_stt

    text = "3分測って。ご視聴ありがとうございました"
    with caplog.at_level(logging.INFO, logger=local_stt.__name__):
        assert local_stt.drop_if_hallucination(text) == ""
    assert "3分測って" not in caplog.text and f"{len(text)} 字" in caplog.text
