"""声に出す前に読み（ひらがな）に直す（環-t・2026-09-19）。

ElevenLabs の flash も multilingual_v2 も漢字の読みが使えなかった（出入口・押入・金木犀・九月十九日・三分の 5 語が
全部 ×・`language_code=ja` でも同じ）。v3 は 4/5 だが遅い。**pyopenjtalk（SBV2 と同じ辞書）で読みをひらがなに
してから flash に渡す**と、いまの声と速さのまま自然に読めた（本人が聴いて確認）。画面の文字は変えない。
ElevenLabs のときだけ（SBV2 は自前で読む）。pyopenjtalk が無ければ元の文のまま。
"""

from __future__ import annotations

from familiar_agent.core import reading


def test_kanji_becomes_hiragana_and_punctuation_stays():
    assert reading.to_hiragana("出入口の押入の前で、金木犀の香りがしたよ。") == (
        "でいりぐちのおしいれのまえで、きんもくせーのかおりがしたよ。"
    )
    assert reading.to_hiragana("三分のタイマーを掛けたよ。") == "さんぷんのたいまーをかけたよ。"
    assert reading.to_hiragana("") == ""


def test_the_reading_table_still_wins_for_exceptions(monkeypatch):
    monkeypatch.setitem(reading.READINGS, "パジュ", "ぱじゅ")
    assert reading.for_speech("パジュ、出入口を見て") == "ぱじゅ、でいりぐちをみて"


def test_without_pyopenjtalk_the_text_is_left_alone(monkeypatch):
    monkeypatch.setattr(reading, "_g2p", lambda text: None)
    assert reading.for_speech("出入口へ") == "でいりぐちへ"  # 表の分だけ
    assert reading.for_speech("押入へ") == "押入へ"


def test_elevenlabs_gets_hiragana_but_sbv2_does_not():
    from familiar_agent.tools.tts import TTSTool

    e = TTSTool("k", "v", engine="elevenlabs")
    assert e._clean_for_speech("押入を見てくる（首を回す）") == "押入を見てくる"  # 声の門へは元の文
    assert e._text_for_synth("押入を見てくる") == "おしいれをみてくる"  # 合成器へは読み
    s = TTSTool("k", "v", engine="sbv2")
    assert s._text_for_synth("押入を見てくる") == "押入を見てくる"


def test_tags_latin_and_pure_kana_are_left_alone():
    assert reading.to_hiragana("[cheerful] こんばんは") == "[cheerful] こんばんは"
    assert reading.to_hiragana("hello world") == "hello world"
    assert reading.to_hiragana("MCP の道具で予定を見た") == "MCP のどーぐでよてーをみた"
