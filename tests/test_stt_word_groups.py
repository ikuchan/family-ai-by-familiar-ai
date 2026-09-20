"""語の組（2026-09-20）：STT へ渡す語の列と、書き起こしの直しを 1 つの表から導く。

表は **(直すべき語, (あり得る語…))** の並び。直すべき語が `-` の組は「消す」（置き換え先が空）。
STT へ渡す語の列（`hotwords`）はこの表から作り、その列そのものも「消す組」として自動で足す——
列が音の無いところでそのまま書き起こされ、人の発話として積まれたため（実機 2026-09-20 17:22〜17:26）。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from familiar_agent.config import STTConfig
from familiar_agent.core.stt_rules import (
    fix_words,
    hotwords_for,
    parse_word_groups,
    with_hint,
)
from familiar_agent.realtime_stt_session import create_realtime_stt_session
from familiar_agent.tools.local_stt import LocalSttEngine

GROUPS = (("パジュ", ("はじゅ", "パチュ", "パジュー")), ("出入口", ("でいりぐち",)))


# ── 表の読み取り ───────────────────────────────────────────────────────────


def test_the_table_is_read_from_one_line():
    got = parse_word_groups("パジュ：はじゅ、パチュ、パジュー; 出入口：でいりぐち")
    assert got == GROUPS


def test_a_group_whose_target_is_a_dash_means_delete():
    got = parse_word_groups("-：ご視聴ありがとうございました")
    assert got == (("-", ("ご視聴ありがとうございました",)),)


def test_inner_spaces_are_kept_and_the_ends_are_trimmed():
    got = parse_word_groups(" -： パジュ はじゅ パチュ ")
    assert got == (("-", ("パジュ はじゅ パチュ",)),)


def test_a_broken_group_is_skipped():
    assert parse_word_groups("") == ()
    assert parse_word_groups("コロンが無い") == ()
    assert parse_word_groups("パジュ：") == ()


def test_the_config_reads_the_table_from_env(monkeypatch):
    monkeypatch.delenv("STT_WORD_GROUPS", raising=False)
    assert STTConfig().word_groups == ()
    monkeypatch.setenv("STT_WORD_GROUPS", "出入口：でいりぐち")
    assert STTConfig().word_groups == (("出入口", ("でいりぐち",)),)


# ── 語の列と、直し ─────────────────────────────────────────────────────────


def test_the_word_list_for_the_tool_is_built_from_the_table():
    assert hotwords_for(GROUPS) == "パジュ はじゅ パチュ パジュー 出入口 でいりぐち"
    # 消す組は、直すべき語（`-`）を列に入れない
    assert (
        hotwords_for((("-", ("ご視聴ありがとうございました",)),)) == "ご視聴ありがとうございました"
    )
    assert hotwords_for(()) == ""


def test_the_samples_are_replaced_by_the_target():
    assert fix_words("パチュ、3 分測って", GROUPS) == "パジュ、3 分測って"
    assert fix_words("はじゅ、でいりぐちを見て", GROUPS) == "パジュ、出入口を見て"
    # 長い綴りから当てる（「パジュー」を先に直さないと「ー」が残る）
    assert fix_words("パジューさん", GROUPS) == "パジュさん"
    # 表に無い語は触らない
    assert fix_words("体重を測って", GROUPS) == "体重を測って"
    assert fix_words("パジュ、3 分測って", GROUPS) == "パジュ、3 分測って"


def test_a_group_marked_delete_removes_the_words():
    groups = (("-", ("ご視聴ありがとうございました",)),)
    assert fix_words("ご視聴ありがとうございました", groups) == ""
    assert fix_words("ご視聴ありがとうございました、3 分測って", groups) == "3 分測って"


def test_the_word_list_itself_is_added_as_a_group_to_delete():
    groups = with_hint(GROUPS, hotwords_for(GROUPS))
    text = "パジュ はじゅ パチュ パジュー 出入口 でいりぐち"
    assert fix_words(text, groups) == ""
    assert fix_words(text + "、3 分測って", groups) == "3 分測って"
    # 語 1 つは消さない（人が名前を呼んだ言葉）
    assert fix_words("パチュ", groups) == "パジュ"


def test_with_hint_without_a_hint_changes_nothing():
    assert with_hint(GROUPS, "") == GROUPS


# ── 書き起こしの経路 ───────────────────────────────────────────────────────


def _engine_with(text: str, groups) -> tuple[LocalSttEngine, MagicMock]:
    cfg = STTConfig()
    cfg.word_groups = groups
    engine = LocalSttEngine(cfg)
    seg = MagicMock()
    seg.text = text
    seg.no_speech_prob = 0.1
    seg.avg_logprob = -0.3
    seg.start, seg.end = 0.0, 1.0
    model = MagicMock()
    model.transcribe.return_value = (iter([seg]), None)
    return engine, model


def _run(engine, model) -> str:
    with patch("familiar_agent.tools.stt.load_whisper_model", return_value=model):
        return engine._transcribe(b"\x00\x00" * 16000)


def test_the_engine_passes_the_word_list_and_fixes_the_transcript(caplog):
    engine, model = _engine_with("パチュ、3 分測って", GROUPS)
    with caplog.at_level("INFO"):
        out = _run(engine, model)
    assert model.transcribe.call_args.kwargs["hotwords"] == hotwords_for(GROUPS)
    assert model.transcribe.call_args.kwargs.get("initial_prompt") is None
    assert out == "パジュ、3 分測って"
    assert any("語を直した" in r.getMessage() for r in caplog.records)


def test_the_engine_drops_a_transcript_that_is_only_the_word_list():
    engine, model = _engine_with(hotwords_for(GROUPS), GROUPS)
    assert _run(engine, model) == ""


def test_the_engine_keeps_the_words_that_follow_the_echo():
    engine, model = _engine_with(hotwords_for(GROUPS) + "、3 分測って", GROUPS)
    assert _run(engine, model) == "3 分測って"


def test_without_a_table_nothing_is_passed_or_rewritten():
    engine, model = _engine_with("パジュー、3 分測って", ())
    out = _run(engine, model)
    assert model.transcribe.call_args.kwargs.get("hotwords") is None
    assert out == "パジュー、3 分測って"


# ── 集音セッション ─────────────────────────────────────────────────────────


def test_the_session_makes_the_name_the_first_group(monkeypatch, tmp_path):
    (tmp_path / "ME.md").write_text(
        "# 私について\n\n名前：パジュ、はじゅ、パチュ\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REALTIME_STT", "true")
    monkeypatch.setenv("STT_ENGINE", "whisper")
    monkeypatch.setenv("STT_WORD_GROUPS", "出入口：でいりぐち")
    session = create_realtime_stt_session()
    assert session is not None
    assert session._stt_config.word_groups == (
        ("パジュ", ("はじゅ", "パチュ")),
        ("出入口", ("でいりぐち",)),
    )
