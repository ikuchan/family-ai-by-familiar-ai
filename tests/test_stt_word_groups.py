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


def test_the_word_list_is_joined_with_a_comma_not_a_space():
    """**空白でつながない**（知-ah・2026-09-24）。

    faster-whisper は `hotwords` を `<|startofprev|>`（直前に出てきた言葉）の枠へ入れる
    ので、渡した列の**書き方**が書き起こしの書き方になる。空白で区切った列を渡すと、
    書き起こしも空白で切れた——実機 15:57「しょうめん を み て」。

    同じ音で区切りを変えて測った（`根拠台帳` §46）。空白は名前 3/4・空白の歪み 1/3、
    読点・中点・改行はいずれも **4/4・0/3**。読点を採ったのは、`ME.md` の「名前：」が
    読点区切りで、**人が書いた形をそのまま渡す**ことになるからである。
    """
    assert hotwords_for(GROUPS) == "パジュ、出入口"
    assert " " not in hotwords_for(GROUPS)
    # 消す組は何も渡さない
    assert hotwords_for((("-", ("ご視聴ありがとうございました",)),)) == ""
    assert hotwords_for(()) == ""


def test_only_the_target_words_are_passed():
    """**渡すのは直すべき語だけ**（知-z-ろ・2026-10-07）。聞き違いの綴り（はじゅ・パチュ・パジュー）まで渡すと、
    音の無いところで崩れた形の一覧（「パチュ、はじゅ、はじゅー」）が書き出され、直すと「パジュ、パジュ、パジュー」に
    なって、誰も呼んでいないのにパジュが返事をした（実機 15:17〜15:41・11 回）。聞き違いは直す側にだけ使う。"""
    names = (("パジュ", ("はじゅ", "パチュ", "パジュー")),)
    assert hotwords_for(names) == "パジュ"
    # 1 語なので丸ごとの写しを消す組は足さない（正しく取れた名前を消さない）
    assert with_hint(names, hotwords_for(names)) == names
    assert fix_words("パチュ、3 分測って", names) == "パジュ、3 分測って"


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
    """**まるごとの写しは、渡した形そのもの**である（知-ah で区切りが読点になった）。

    音に情報が無いと、渡した列がそのまま書き起こされる（実機 2026-09-20・知-z）。
    落とす対象は「渡した列」なので、区切りを変えたら写しの形も変わる。
    """
    groups = with_hint(GROUPS, hotwords_for(GROUPS))
    text = hotwords_for(
        GROUPS
    )  # 渡した列そのもの（知-z-ろ からは直すべき語だけ：「パジュ、出入口」）
    assert fix_words(text, groups) == ""
    assert fix_words(text + "、3 分測って", groups) == "3 分測って"
    # 語 1 つは消さない（人が名前を呼んだ言葉）
    assert fix_words("パチュ", groups) == "パジュ"


def test_with_hint_without_a_hint_changes_nothing():
    assert with_hint(GROUPS, "") == GROUPS


def test_a_single_word_hint_is_never_deleted():
    """**語 1 つは消さない**（知-ah・2026-09-24）。

    説明文にはそう書いてあったのに、守りが実装されていなかった。名前だけを渡すと、
    正しく取れた名前を `fix_words` が消した——`ねえ、パジュ、聞こえる?` → `ねえ、、聞こえる?`。
    消す組は「**渡した列がまるごと書き起こされた**」ときのためのもので、語 1 つでは
    人が名前を呼んだ言葉と見分けが付かない。
    """
    assert with_hint(GROUPS, "パジュ") == GROUPS
    assert (
        fix_words("ねえ、パジュ、聞こえる?", with_hint(GROUPS, "パジュ"))
        == "ねえ、パジュ、聞こえる?"
    )
    # 2 語以上なら、いままでどおり消す組を足す
    assert with_hint(GROUPS, "パジュ、はじゅ") != GROUPS


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


def test_the_engine_passes_no_words_and_still_fixes_the_transcript(caplog):
    """**名前は渡さない**（知-z-は イ-1・2026-10-07 本人の決定）。頭を付けた声 4 つ（21:44）を書き起こし直すと、
    「パジュ」1 語を渡すと 1 回名前が消え（「今何時?」）、渡さなければ 4 回とも「アジュー」などが取れた。聞き違いは
    いまどおり直す表で直す。"""
    engine, model = _engine_with("パチュ、3 分測って", GROUPS)
    with caplog.at_level("INFO"):
        out = _run(engine, model)
    assert model.transcribe.call_args.kwargs.get("hotwords") is None
    assert model.transcribe.call_args.kwargs.get("initial_prompt") is None
    assert out == "パジュ、3 分測って"
    assert any("語を直した" in r.getMessage() for r in caplog.records)


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
