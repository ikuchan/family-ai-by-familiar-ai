"""沈黙依頼の機械の守り（`core/silence_rules.py`・2026-09-16 実機）。

軽量LLM の読みだけに任せると両方向で外れた：「待てぃ」（11:47）「しなよ」（15:12・STT の断片）
を沈黙の依頼と読んで 60 分黙り、「話していいよ」（15:16）を解除と読まなかった。設計は「名前で
呼ばれたときだけ受ける」（`test_silence_by_name`）だが、プロンプトの指示だけだった。

- 掛ける側：`silence_minutes` が立っても、発話に自分の名前（`ME.md` の名前・呼び方）が無ければ受けない。
- 解く側：黙っているあいだに本人が「話していい／しゃべっていい」と言えば、調停の返りに依らず解く。
- 調停へ、いま黙っていること（誰から・いつまで）を渡す。黙っている前提が無いと「解かれた」と読めない。
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from familiar_agent.core.silence_rules import is_release, names_me, silence_note
from familiar_agent.loop.arbiter import Decision
from familiar_agent.silence_state import SilenceRequest

NAMES = ["パジュ", "ぱじゅ"]


def test_a_request_counts_only_when_she_is_named():
    assert names_me("パジュ、ちょっと静かにして", NAMES)
    assert names_me("静かにしてぱじゅ", NAMES)
    assert not names_me("しなよ", NAMES)
    assert not names_me("待てぃ", NAMES)
    assert not names_me("ちょっと静かにして", NAMES)


def test_a_name_the_stt_mangled_still_counts():
    """STT は「パジュ」を はじゅ／パチュー と書く（2026-09-17 15:32 実機）。ゆるい読みで拾う。"""
    assert names_me("はじゅ、静かにして", NAMES)
    assert names_me("大好きなのはパチュー静かにして", NAMES)
    assert names_me("パジュー、黙って", NAMES)


def test_loose_reading_does_not_open_the_door_to_anything():
    """守りの目的（「待てぃ」「しなよ」で黙らない）は変わらない。別の語も名前にしない。"""
    assert not names_me("待てぃ", NAMES)
    assert not names_me("しなよ", NAMES)
    assert not names_me("体重を静かにして", NAMES)
    assert not names_me("はい、静かにして", NAMES)


def test_short_names_stay_exact():
    """2 文字以下の名前は 1 文字違いを許すと何にでも当たるので、完全一致のまま。"""
    assert names_me("ゆき、静かにして", ["ゆき"])
    assert not names_me("ゆめ、静かにして", ["ゆき"])


def test_saying_she_may_talk_is_a_release():
    for text in ("話していいよ", "もうしゃべっていいよ", "喋ってもいいよ", "話してもいいです"):
        assert is_release(text), text
    for text in ("話して", "静かにして", "もう1週間経ったのにな", "話しかけないで"):
        assert not is_release(text), text


def test_the_note_tells_the_arbiter_who_asked_and_until_when():
    req = SilenceRequest(person="パパ", until=time.time() + 1800)
    note = silence_note(req, now=time.time())
    assert "黙っているよう頼まれている" in note and "パパ" in note and "分" in note
    assert silence_note(None, now=time.time()) == ""
    assert silence_note(SilenceRequest(person="パパ", until=time.time() - 1), now=time.time()) == ""


# ── ループ側の適用 ──────────────────────────────────────────────────────────


def _ip(speaker="パパ"):
    from familiar_agent.loop.event_loop import InformationProcessing

    a = MagicMock()
    a.config.agent_names = NAMES
    a._pmm.presence_status = MagicMock(
        return_value=[{"name": speaker, "is_speaker": True, "confidence": 1.0}]
    )
    ip = InformationProcessing(a)
    calls: list[str] = []
    ip._accept_silence = lambda m: calls.append(f"掛ける{m}")
    ip._release_silence = lambda: calls.append("解く")
    return ip, calls


def test_an_unnamed_request_is_accepted_inside_the_window():
    """名前の関門は入口の窓にまとめた（出-as §2.5・2026-09-26）。ここへ来るのは窓を通った入力だけ。"""
    ip, calls = _ip()
    ip._apply_silence(Decision(branch="light", silence_minutes=-1), utterance="しなよ")
    assert calls == ["掛ける-1"]


def test_a_named_request_is_accepted():
    ip, calls = _ip()
    ip._apply_silence(
        Decision(branch="light", silence_minutes=5), utterance="パジュ、5分静かにして"
    )
    assert calls == ["掛ける5"]


def test_a_release_phrase_lifts_even_if_the_arbiter_missed_it():
    ip, calls = _ip()
    ip._apply_silence(Decision(branch="full"), utterance="話していいよ")
    assert calls == ["解く"]


def test_the_arbiters_release_flag_still_works():
    ip, calls = _ip()
    ip._apply_silence(Decision(branch="light", lift_silence=True), utterance="もういいよ")
    assert calls == ["解く"]
