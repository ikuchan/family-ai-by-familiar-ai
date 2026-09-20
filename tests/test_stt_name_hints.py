"""知-z（2026-09-19）：「パジュ」が書き起こしから落ちる・化ける。

`hotwords` に `ME.md` の綴りを全部渡し、書き起こしに上がった聞き違いの綴りは先頭の綴り（正しい名前）へ直す。
**例文（`initial_prompt`）は渡さない**——2026-09-20 に入れたが、はっきりしない音でその文がそのまま
書き起こされ、話していないのに「パジュ、3 分測って。」が繰り返し会話として上がった（実機 17:22〜17:26）。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from familiar_agent.config import STTConfig
from familiar_agent.core.stt_rules import normalize_name
from familiar_agent.realtime_stt_session import create_realtime_stt_session
from familiar_agent.tools.local_stt import LocalSttEngine

NAMES = ("パジュ", "はじゅ", "パチュ", "パジュー")


# ── 純関数 ─────────────────────────────────────────────────────────────────


def test_listed_misspellings_are_normalized_to_the_first_name():
    assert normalize_name("パジュー、3 分測って", NAMES) == "パジュ、3 分測って"
    assert normalize_name("はじゅ、静かにして", NAMES) == "パジュ、静かにして"
    assert normalize_name("パチュだよ", NAMES) == "パジュだよ"


def test_normalization_leaves_other_words_and_the_right_name_alone():
    # 体重 は ME.md に無いので当てない（本物の「体重」を壊さない）
    assert normalize_name("体重を測って", NAMES) == "体重を測って"
    assert normalize_name("パジュ、こんにちは", NAMES) == "パジュ、こんにちは"
    assert normalize_name("", NAMES) == ""
    assert normalize_name("パジュー", ()) == "パジュー"


def test_the_longest_spelling_wins_so_no_stray_tail_is_left():
    # 「パジュー」を先に「パジュ」にしないと「パジュー」→「パジュー」のまま残る
    assert normalize_name("パジューさん", ("パジュ", "パジュー")) == "パジュさん"


# ── 集音セッションと書き起こし ──────────────────────────────────────────────


def test_the_session_gives_every_spelling_as_hotwords_and_keeps_the_names(monkeypatch, tmp_path):
    (tmp_path / "ME.md").write_text(
        "# 私について\n\n名前：パジュ、はじゅ、パチュ\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REALTIME_STT", "true")
    monkeypatch.setenv("STT_ENGINE", "whisper")
    session = create_realtime_stt_session()
    assert session is not None
    assert session._stt_config.hotwords == "パジュ はじゅ パチュ"
    assert session._stt_config.names == ("パジュ", "はじゅ", "パチュ")


def _engine_with(text: str, names: tuple[str, ...]) -> tuple[LocalSttEngine, MagicMock]:
    cfg = STTConfig()
    cfg.names = names
    cfg.hotwords = " ".join(names)
    engine = LocalSttEngine(cfg)
    seg = MagicMock()
    seg.text = text
    seg.no_speech_prob = 0.1
    seg.avg_logprob = -0.3
    seg.start, seg.end = 0.0, 1.0
    model = MagicMock()
    model.transcribe.return_value = (iter([seg]), None)
    return engine, model


def test_no_prompt_is_given_and_the_transcript_is_normalized(caplog):
    engine, model = _engine_with("パジュー、3分測って", NAMES)
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        with caplog.at_level("INFO"):
            out = engine._transcribe(b"\x00\x00" * 16000)
    # 例文（`initial_prompt`）は渡さない。渡すと、はっきりしない音に対して Whisper がその文を
    # そのまま書き出す（実機 2026-09-20 17:22〜17:26・10.7 秒の音と 30 秒の窓が例文の後半 11 字に）。
    assert model.transcribe.call_args.kwargs.get("initial_prompt") is None
    assert model.transcribe.call_args.kwargs["hotwords"] == "パジュ はじゅ パチュ パジュー"
    assert out == "パジュ、3分測って"
    assert any("名前を「パジュ」に直した" in r.getMessage() for r in caplog.records)


def test_without_names_nothing_is_rewritten():
    engine, model = _engine_with("パジュー、3分測って", ())
    engine._cfg.hotwords = ""
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        out = engine._transcribe(b"\x00\x00" * 16000)
    assert out == "パジュー、3分測って"
