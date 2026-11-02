"""長い返事は 100 字以内に分け、話しながら次を作る（出-dh 段 1 の直し・2026-11-02・本人の決定イ）。

声の道具（`tools/tts.py`）は、どの担い手でも 200 字を超えた文を `text[:197] + "..."` で切っていた。深く調べた答え（目標
400 字・800 字以内）は最初の 197 字ほどで切れ、調べた求めの主LLM の返事（240 字まで）も 200 字を超えた分が切れていた。
100 字を超える文は文の終わりで 100 字以内に分け、いまのかたまりを鳴らしているあいだに次のかたまりの声を作っておく
（本人「100 文字以下に分けて流していく形」・TTS の待ちを減らす）。本物の声の API は呼ばない。
"""

from __future__ import annotations

import asyncio

import pytest

from familiar_agent.core.voice_rules import SPEECH_CHUNK_CHARS, split_for_speech
from familiar_agent.tools.tts import TTSTool

LONG = "".join(f"これは{i}番目の文で、少し長めに書いてあります。" for i in range(12))


# ── 分け方 ──────────────────────────────────────────────────────────────


def test_the_chunk_is_100_characters():
    assert SPEECH_CHUNK_CHARS == 100


def test_a_short_reply_is_one_piece():
    assert split_for_speech("こんにちは。元気？") == ["こんにちは。元気？"]


def test_a_long_reply_is_cut_at_sentence_ends_without_loss():
    parts = split_for_speech(LONG)
    assert len(parts) > 1 and "".join(parts) == LONG
    assert all(len(p) <= SPEECH_CHUNK_CHARS for p in parts)
    assert all(p.endswith("。") for p in parts)  # 文の途中で切らない


def test_one_very_long_sentence_is_still_kept_within_the_limit():
    """**反証の側**：句点の無い長い 1 文も、読点か上限で分け、何も落とさない。"""
    text = "、".join(["とても長い文のかたまり"] * 30) + "。"
    parts = split_for_speech(text)
    assert "".join(parts) == text and all(len(p) <= SPEECH_CHUNK_CHARS for p in parts)
    nocomma = "あ" * 250
    parts = split_for_speech(nocomma)
    assert "".join(parts) == nocomma and all(len(p) <= SPEECH_CHUNK_CHARS for p in parts)


# ── 声の道具：分けて、話しながら次を作る ───────────────────────────────────────


def _tool(engine: str = "sbv2", fallback: str = "") -> TTSTool:
    return TTSTool(
        api_key="dummy-key", voice_id="v1", output="local", engine=engine, fallback_engine=fallback
    )


def _fake(tool: TTSTool, *, fail_on: "set[str] | None" = None, play_wait: float = 0.0):
    """作る・鳴らす・1 回の道（1 つ目のかたまり）を偽物にする。起きた順を `log` に残す。"""
    log: list = []
    fail_on = fail_on or set()

    async def synth(engine, text, careful):
        log.append(("作る", engine, text))
        if engine in fail_on:
            raise OSError("作れなかった")
        await asyncio.sleep(0)
        return f"/tmp/{engine}-{len(log)}.wav"

    async def play(path, output):
        log.append(("鳴らす始め", path))
        await asyncio.sleep(play_wait)
        log.append(("鳴らす終わり", path))
        return ["local"]

    async def say_with(engine, text, output, *, careful):
        log.append(("作る", engine, text))
        if engine in fail_on:
            return "話せなかった"
        log.append(("鳴らす始め", "1 回の道"))
        await asyncio.sleep(play_wait)
        log.append(("鳴らす終わり", "1 回の道"))
        return "Said: ok"

    tool._synth_file = synth  # type: ignore[method-assign]
    tool._play_paths = play  # type: ignore[method-assign]
    tool._say_with = say_with  # type: ignore[method-assign]
    return log


def _made(log) -> "list[str]":
    """話したかたまり（作れた担い手のもの）を順に。"""
    return [t for kind, _e, t in (x for x in log if x[0] == "作る")]


def test_a_long_reply_is_spoken_piece_by_piece_in_order():
    tool = _tool()
    log = _fake(tool)
    result = asyncio.run(tool.say(LONG))
    made = _made(log)
    assert result.startswith("Said:")
    assert "".join(made) == LONG and len(made) > 1
    assert len([x for x in log if x[0] == "鳴らす始め"]) == len(made)


def test_the_next_piece_is_made_while_the_current_one_plays():
    tool = _tool()
    log = _fake(tool, play_wait=0.05)
    asyncio.run(tool.say(LONG))
    first_end = log.index(next(x for x in log if x[0] == "鳴らす終わり"))
    second_make = [i for i, x in enumerate(log) if x[0] == "作る"][1]
    assert second_make < first_end  # 1 つ目を鳴らし終える前に、2 つ目を作り始めている


def test_a_short_reply_keeps_the_one_call_path():
    """**反証の側**：100 字以内は、いまの 1 回の道のまま（分けない・先に作らない）。"""
    tool = _tool()
    log = _fake(tool)
    assert asyncio.run(tool.say("こんにちは。")).startswith("Said:")
    assert log[0] == ("作る", "sbv2", "こんにちは。") and len(_made(log)) == 1


def test_a_piece_that_cannot_be_made_falls_back():
    """**反証の側**：主の担い手で作れなかったかたまりは、控えの担い手で作り直す。"""
    tool = _tool("gemini", fallback="sbv2")
    log = _fake(tool, fail_on={"gemini"})
    assert asyncio.run(tool.say(LONG)).startswith("Said:")
    assert {e for kind, e, _t in (x for x in log if x[0] == "作る")} == {"gemini", "sbv2"}


def test_cutting_it_off_also_stops_making_the_next_piece():
    """**反証の側**：途中で打ち切られたら、作りかけの次のかたまりも止める。"""
    tool = _tool()
    stopped: list = []
    first = split_for_speech(LONG)[0]

    async def synth(engine, text, careful):
        if text == first:
            return "/tmp/a.wav"
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            stopped.append(text)
            raise
        return "/tmp/b.wav"

    async def play(path, output):
        await asyncio.sleep(10)
        return ["local"]

    tool._synth_file = synth  # type: ignore[method-assign]
    tool._play_paths = play  # type: ignore[method-assign]

    async def go():
        task = asyncio.ensure_future(tool.say(LONG))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        return list(stopped)  # ループが閉じて残りが止められる前に見る

    assert asyncio.run(go()) == [split_for_speech(LONG)[1]]


# ── 見直しで足した確かめ（1 回目）───────────────────────────────────────────


def test_the_voice_guard_stays_up_for_the_whole_reply():
    """声の門は返事全体で 1 回立て、話し終わるまで下げない（かたまりの境の自分の声を取り違えない）。"""
    from unittest.mock import MagicMock

    tool = _tool()
    _fake(tool)
    guard = MagicMock()
    tool._voice_guard = guard
    asyncio.run(tool.say(LONG))
    guard.on_tts_start.assert_called_once_with(LONG)
    guard.on_tts_end.assert_called_once()


def test_another_voice_waits_until_the_long_reply_ends():
    """長い返事のかたまりのあいだに、ほかの声（タイマーの知らせなど）を挟まない。"""
    tool = _tool()
    log = _fake(tool, play_wait=0.02)

    async def go():
        long_task = asyncio.ensure_future(tool.say(LONG))
        await asyncio.sleep(0.01)
        await tool.say("タイマーが鳴りました。")
        await long_task

    asyncio.run(go())
    made = _made(log)
    assert made[-1] == "タイマーが鳴りました。" and "".join(made[:-1]) == LONG


def test_nothing_spoken_is_a_failure_but_a_lost_piece_is_not():
    """何も話せなければ失敗。1 つでも話せたら成功と返す（全部の言い直しを招かない・欠けた分は警告に残す）。"""
    tool = _tool("sbv2")
    _fake(tool)

    async def none(engine, text, careful):
        raise OSError("作れなかった")

    tool._synth_file = none  # type: ignore[method-assign]
    assert not asyncio.run(tool.say(LONG)).startswith("Said:")
    tool = _tool("sbv2")
    log = _fake(tool)
    second = split_for_speech(LONG)[1]
    real = tool._synth_file

    async def one_lost(engine, text, careful):
        if text == second:
            raise OSError("作れなかった")
        return await real(engine, text, careful)

    tool._synth_file = one_lost  # type: ignore[method-assign]
    assert asyncio.run(tool.say(LONG)).startswith("Said:")
    assert second not in _made(log)


def test_gemini_speaks_the_first_piece_on_its_streaming_path():
    """Gemini は 1 つ目を、届いた分から鳴らす 1 回の道で話す（最初の声を遅らせない）。"""
    tool = _tool("gemini")
    log = _fake(tool)
    asyncio.run(tool.say(LONG))
    assert ("鳴らす始め", "1 回の道") in log
    assert log[0][2] == split_for_speech(LONG)[1] or log[0][2] == split_for_speech(LONG)[0]


def test_sbv2_makes_the_second_piece_only_after_the_first():
    """Style-Bert-VITS2（同じ GPU）は、1 つ目を作り終えてから 2 つ目を頼む（取り合って最初の声を遅らせない）。"""
    tool = _tool("sbv2")
    log = _fake(tool)
    asyncio.run(tool.say(LONG))
    makes = [x[2] for x in log if x[0] == "作る"]
    assert makes[:2] == split_for_speech(LONG)[:2]
    assert ("鳴らす始め", "1 回の道") not in log


def test_an_error_on_the_first_piece_falls_back():
    """**反証の側**：1 つ目で通信の例外が起きても、控えの担い手へ落ちる。"""
    tool = _tool("gemini", fallback="sbv2")
    log = _fake(tool)

    async def say_with(engine, text, output, *, careful):
        log.append(("作る", engine, text))
        if engine == "gemini":
            raise ConnectionError("つながらない")
        return "Said: ok"

    tool._say_with = say_with  # type: ignore[method-assign]
    assert asyncio.run(tool.say(LONG)).startswith("Said:")
    assert ("作る", "sbv2", split_for_speech(LONG)[0]) in log


def test_odd_pieces_are_not_made():
    """「。」だけ・閉じかっこだけ・文頭の引用符だけのかたまりを作らない。数字のカンマで分けない。"""
    for text in ("あ" * 100 + "。", "あ" * 99 + "。」" + "い" * 50, '"' + "い" * 120 + "。"):
        parts = split_for_speech(text)
        assert "".join(parts) == text
        assert all(any(ch.isalnum() for ch in p) for p in parts), parts
    price = "あ" * 60 + "1,000円" + "い" * 60 + "。"  # 読点の無い長い 1 文の中の位取りのカンマ
    assert any("1,000" in p for p in split_for_speech(price))


def test_a_closing_bracket_stays_with_its_sentence():
    text = ("「" + "あ" * 60 + "と言っていました。」") * 3
    parts = split_for_speech(text)
    assert "".join(parts) == text and not any(p.startswith("」") for p in parts)


def test_a_fallback_piece_gets_its_own_engine_reading():
    """控えの ElevenLabs で作るかたまりは、主（SBV2）の読みではなく、ElevenLabs の読み（ひらがな）にする。"""
    tool = _tool("sbv2", fallback="elevenlabs")
    body = tool._elevenlabs_payload("漢字", careful=False)["text"]
    from familiar_agent.core.reading import for_speech

    assert body == for_speech("漢字")
